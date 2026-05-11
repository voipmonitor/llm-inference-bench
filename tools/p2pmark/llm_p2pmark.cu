#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nccl.h>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#define CHECK_CUDA(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

#define CHECK_NCCL(cmd) do { \
  ncclResult_t r = (cmd); \
  if (r != ncclSuccess) { \
    std::fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    std::exit(1); \
  } \
} while (0)

#define CHECK(cmd) CHECK_CUDA(cmd)

// Custom PCIe allreduce kernel, adapted from p2pmark / pcie_allreduce.cu.
// It is intentionally limited to <=8 GPUs, matching the upstream diagnostic.
namespace pcie_ar {

constexpr int kMaxBlocks = 36;
using FlagType = uint32_t;
constexpr int kFlagStride = 32;

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][16 * kFlagStride];
};

struct __align__(16) RankData {
  const void* __restrict__ ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

DINLINE float upcast_s(half val) { return __half2float(val); }
template <typename T> DINLINE T downcast_s(float val);
template <> DINLINE half downcast_s(float val) { return __float2half(val); }
DINLINE half& assign_add(half& a, half b) { a = __hadd(a, b); return a; }
DINLINE float& assign_add(float& a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <> DINLINE nv_bfloat16 downcast_s(float val) { return __float2bfloat16(val); }
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) { a = __hadd(a, b); return a; }
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]);
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) out.data[i] = upcast_s(val.data[i]);
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) out.data[i] = downcast_s<typename O::type>(val.data[i]);
    return out;
  }
}

static DINLINE void st_flag_relaxed(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_relaxed(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

template <int ngpus, bool is_start>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  if (threadIdx.x < ngpus) {
    __threadfence_system();
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank * kFlagStride];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x * kFlagStride];
    st_flag_relaxed(peer_counter_ptr, val);
    while (ld_flag_relaxed(self_counter_ptr) != val);
  }
  __syncthreads();
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(ptrs[i][idx]));
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_kernel(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  const P* rotated[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    rotated[i] = (const P*)dp.ptrs[(rank + i) % ngpus];
  }
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>(rotated, idx);
  }
}

template <typename T>
static void launch_kernel(int ngpus, int blocks, int threads, cudaStream_t stream,
                          RankData* rd, RankSignals rs, Signal* self_sg,
                          T* output, int rank, int packed_size) {
#define KL(ng) pcie_allreduce_kernel<T, ng><<<blocks, threads, 0, stream>>>(rd, rs, self_sg, output, rank, packed_size)
  switch (ngpus) {
    case 2: KL(2); break;
    case 3: KL(3); break;
    case 4: KL(4); break;
    case 5: KL(5); break;
    case 6: KL(6); break;
    case 7: KL(7); break;
    case 8: KL(8); break;
  }
#undef KL
}

#undef DINLINE

}  // namespace pcie_ar

struct Options {
  std::string mode = "all";
  int size_mb = 64;
  int iters = 20;
  int warmup = 5;
  int latency_iters = 10000;
  int max_gpus = 0;
  std::vector<size_t> allreduce_sizes_bytes;
};

static void usage(const char* prog) {
  std::printf(
      "Usage: %s [--mode all|bandwidth|latency|allreduce] [--size-mb N] [--iters N]\\n"
      "          [--warmup N] [--latency-iters N] [--max-gpus N]\\n"
      "          [--allreduce-sizes-mb 1,2,4,8,16]\\n"
      "Default allreduce sweep: 256 B .. 1 MiB. Use --allreduce-sizes-mb for larger MiB sizes.\\n",
      prog);
}

static std::vector<size_t> default_allreduce_sizes() {
  std::vector<size_t> sizes;
  for (size_t s = 256; s <= 1ULL * 1024 * 1024; s *= 2) {
    sizes.push_back(s);
    if (s >= 32 * 1024 && s < 64 * 1024) {
      sizes.push_back(s + s / 4);
      sizes.push_back(s + s / 2);
      sizes.push_back(s + 3 * s / 4);
    }
  }
  return sizes;
}

static std::vector<size_t> parse_csv_mib_as_bytes(const char* s) {
  std::vector<size_t> out;
  const char* p = s;
  while (*p) {
    char* end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v <= 0) {
      std::fprintf(stderr, "Invalid positive integer list: %s\n", s);
      std::exit(2);
    }
    out.push_back(static_cast<size_t>(v) * 1024 * 1024);
    p = end;
    if (*p == ',') p++;
  }
  return out;
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      usage(argv[0]);
      std::exit(0);
    } else if (!std::strcmp(argv[i], "--mode")) {
      opt.mode = need_value(argv[i]);
    } else if (!std::strcmp(argv[i], "--size-mb")) {
      opt.size_mb = std::atoi(need_value(argv[i]));
    } else if (!std::strcmp(argv[i], "--iters")) {
      opt.iters = std::atoi(need_value(argv[i]));
    } else if (!std::strcmp(argv[i], "--warmup")) {
      opt.warmup = std::atoi(need_value(argv[i]));
    } else if (!std::strcmp(argv[i], "--latency-iters")) {
      opt.latency_iters = std::atoi(need_value(argv[i]));
    } else if (!std::strcmp(argv[i], "--max-gpus")) {
      opt.max_gpus = std::atoi(need_value(argv[i]));
    } else if (!std::strcmp(argv[i], "--allreduce-sizes-mb")) {
      opt.allreduce_sizes_bytes = parse_csv_mib_as_bytes(need_value(argv[i]));
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(2);
    }
  }
  if (opt.mode != "all" && opt.mode != "bandwidth" && opt.mode != "latency" && opt.mode != "allreduce") {
    std::fprintf(stderr, "--mode must be all, bandwidth, latency, or allreduce\n");
    std::exit(2);
  }
  if (opt.size_mb <= 0 || opt.iters <= 0 || opt.warmup < 0) {
    std::fprintf(stderr, "--size-mb and --iters must be >0; --warmup must be >=0\n");
    std::exit(2);
  }
  if (opt.latency_iters <= 0) {
    std::fprintf(stderr, "--latency-iters must be >0\n");
    std::exit(2);
  }
  if (opt.allreduce_sizes_bytes.empty()) {
    opt.allreduce_sizes_bytes = default_allreduce_sizes();
  }
  return opt;
}

static std::vector<std::vector<int>> peer_matrix(int n) {
  std::vector<std::vector<int>> can(n, std::vector<int>(n, 0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        can[i][j] = 1;
      } else {
        int ok = 0;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&ok, i, j));
        can[i][j] = ok ? 1 : 0;
      }
    }
  }
  return can;
}

static void enable_peers(const std::vector<std::vector<int>>& can) {
  int n = static_cast<int>(can.size());
  for (int i = 0; i < n; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int j = 0; j < n; j++) {
      if (i == j || !can[i][j]) continue;
      cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
      if (e == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (e != cudaSuccess) {
        std::fprintf(stderr, "Peer enable %d -> %d failed: %s\n", i, j, cudaGetErrorString(e));
      }
    }
  }
}

static std::vector<std::vector<double>> measure_bandwidth(
    int n, size_t bytes, int warmup, int iters) {
  std::vector<void*> src(n, nullptr);
  std::vector<void*> dst(n, nullptr);
  std::vector<cudaStream_t> streams(n, nullptr);
  std::vector<cudaEvent_t> start(n, nullptr), stop(n, nullptr);
  for (int d = 0; d < n; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaMalloc(&src[d], bytes));
    CHECK_CUDA(cudaMalloc(&dst[d], bytes));
    CHECK_CUDA(cudaMemset(src[d], 1, bytes));
    CHECK_CUDA(cudaMemset(dst[d], 0, bytes));
    CHECK_CUDA(cudaStreamCreate(&streams[d]));
    CHECK_CUDA(cudaEventCreate(&start[d]));
    CHECK_CUDA(cudaEventCreate(&stop[d]));
  }

  std::vector<std::vector<double>> bw(n, std::vector<double>(n, 0.0));
  for (int from = 0; from < n; from++) {
    for (int to = 0; to < n; to++) {
      CHECK_CUDA(cudaSetDevice(to));
      for (int k = 0; k < warmup; k++) {
        CHECK_CUDA(cudaMemcpyPeerAsync(dst[to], to, src[from], from, bytes, streams[to]));
      }
      CHECK_CUDA(cudaStreamSynchronize(streams[to]));
      CHECK_CUDA(cudaEventRecord(start[to], streams[to]));
      for (int k = 0; k < iters; k++) {
        CHECK_CUDA(cudaMemcpyPeerAsync(dst[to], to, src[from], from, bytes, streams[to]));
      }
      CHECK_CUDA(cudaEventRecord(stop[to], streams[to]));
      CHECK_CUDA(cudaEventSynchronize(stop[to]));
      float ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, start[to], stop[to]));
      double seconds = static_cast<double>(ms) / 1000.0;
      bw[from][to] = (static_cast<double>(bytes) * iters) / seconds / 1e9;
    }
  }

  for (int d = 0; d < n; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    cudaEventDestroy(start[d]);
    cudaEventDestroy(stop[d]);
    cudaStreamDestroy(streams[d]);
    cudaFree(src[d]);
    cudaFree(dst[d]);
  }
  return bw;
}

struct ConcurrentCopyResult {
  std::vector<std::pair<int, int>> pairs;
  std::vector<double> pair_gbps;
  double avg_gbps = 0.0;
  double total_gbps = 0.0;
};

static ConcurrentCopyResult measure_concurrent_copies(
    int n,
    size_t bytes,
    int warmup,
    int iters,
    const std::vector<std::pair<int, int>>& pairs) {
  ConcurrentCopyResult result;
  result.pairs = pairs;
  if (pairs.empty()) return result;

  std::vector<void*> src(n, nullptr);
  std::vector<void*> dst(pairs.size(), nullptr);
  std::vector<cudaStream_t> streams(pairs.size(), nullptr);
  std::vector<cudaEvent_t> start(pairs.size(), nullptr), stop(pairs.size(), nullptr);

  for (int d = 0; d < n; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaMalloc(&src[d], bytes));
    CHECK_CUDA(cudaMemset(src[d], d + 1, bytes));
  }
  for (size_t p = 0; p < pairs.size(); p++) {
    int to = pairs[p].second;
    CHECK_CUDA(cudaSetDevice(to));
    CHECK_CUDA(cudaMalloc(&dst[p], bytes));
    CHECK_CUDA(cudaMemset(dst[p], 0, bytes));
    CHECK_CUDA(cudaStreamCreate(&streams[p]));
    CHECK_CUDA(cudaEventCreate(&start[p]));
    CHECK_CUDA(cudaEventCreate(&stop[p]));
  }

  for (int k = 0; k < warmup; k++) {
    for (size_t p = 0; p < pairs.size(); p++) {
      int from = pairs[p].first;
      int to = pairs[p].second;
      CHECK_CUDA(cudaSetDevice(to));
      CHECK_CUDA(cudaMemcpyPeerAsync(dst[p], to, src[from], from, bytes, streams[p]));
    }
  }
  for (size_t p = 0; p < pairs.size(); p++) {
    CHECK_CUDA(cudaSetDevice(pairs[p].second));
    CHECK_CUDA(cudaStreamSynchronize(streams[p]));
  }

  for (size_t p = 0; p < pairs.size(); p++) {
    CHECK_CUDA(cudaSetDevice(pairs[p].second));
    CHECK_CUDA(cudaEventRecord(start[p], streams[p]));
  }
  for (int k = 0; k < iters; k++) {
    for (size_t p = 0; p < pairs.size(); p++) {
      int from = pairs[p].first;
      int to = pairs[p].second;
      CHECK_CUDA(cudaSetDevice(to));
      CHECK_CUDA(cudaMemcpyPeerAsync(dst[p], to, src[from], from, bytes, streams[p]));
    }
  }
  for (size_t p = 0; p < pairs.size(); p++) {
    CHECK_CUDA(cudaSetDevice(pairs[p].second));
    CHECK_CUDA(cudaEventRecord(stop[p], streams[p]));
  }

  result.pair_gbps.resize(pairs.size(), 0.0);
  double max_seconds = 0.0;
  double sum_gbps = 0.0;
  for (size_t p = 0; p < pairs.size(); p++) {
    CHECK_CUDA(cudaSetDevice(pairs[p].second));
    CHECK_CUDA(cudaEventSynchronize(stop[p]));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start[p], stop[p]));
    double seconds = static_cast<double>(ms) / 1000.0;
    max_seconds = std::max(max_seconds, seconds);
    double gbps = (static_cast<double>(bytes) * iters) / seconds / 1e9;
    result.pair_gbps[p] = gbps;
    sum_gbps += gbps;
  }
  result.avg_gbps = sum_gbps / pairs.size();
  result.total_gbps = max_seconds > 0.0
      ? (static_cast<double>(bytes) * iters * pairs.size()) / max_seconds / 1e9
      : 0.0;

  for (size_t p = 0; p < pairs.size(); p++) {
    CHECK_CUDA(cudaSetDevice(pairs[p].second));
    cudaEventDestroy(start[p]);
    cudaEventDestroy(stop[p]);
    cudaStreamDestroy(streams[p]);
    cudaFree(dst[p]);
  }
  for (int d = 0; d < n; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    cudaFree(src[d]);
  }
  return result;
}

struct BandwidthTopology {
  std::vector<ConcurrentCopyResult> staggered;
  ConcurrentCopyResult ring;
  std::vector<double> single_writer_gbps;
  std::vector<double> all_to_all_gpu_gbps;
  double all_to_all_total_gbps = 0.0;
  double pcie_link_score = 0.0;
  double dense_interconnect_score = 0.0;
};

static BandwidthTopology measure_bandwidth_topology(
    int n, size_t bytes, int warmup, int iters, const std::vector<std::vector<double>>& sequential_bw) {
  BandwidthTopology topo;

  for (int offset = 1; offset < n; offset++) {
    std::vector<std::pair<int, int>> pairs;
    for (int from = 0; from < n; from++) {
      pairs.push_back({from, (from + offset) % n});
    }
    topo.staggered.push_back(measure_concurrent_copies(n, bytes, warmup, iters, pairs));
  }
  if (!topo.staggered.empty()) {
    topo.ring = topo.staggered[0];
  }

  topo.single_writer_gbps.resize(n, 0.0);
  for (int from = 0; from < n; from++) {
    std::vector<std::pair<int, int>> pairs;
    for (int to = 0; to < n; to++) {
      if (to != from) pairs.push_back({from, to});
    }
    auto res = measure_concurrent_copies(n, bytes, warmup, iters, pairs);
    topo.single_writer_gbps[from] = res.total_gbps;
  }

  std::vector<std::pair<int, int>> all_pairs;
  for (int from = 0; from < n; from++) {
    for (int to = 0; to < n; to++) {
      if (from != to) all_pairs.push_back({from, to});
    }
  }
  auto all = measure_concurrent_copies(n, bytes, warmup, iters, all_pairs);
  topo.all_to_all_gpu_gbps.resize(n, 0.0);
  for (int from = 0; from < n; from++) {
    double max_seconds = 0.0;
    int count = 0;
    for (size_t p = 0; p < all.pairs.size(); p++) {
      if (all.pairs[p].first != from || all.pair_gbps[p] <= 0.0) continue;
      double seconds = (static_cast<double>(bytes) * iters) / all.pair_gbps[p] / 1e9;
      max_seconds = std::max(max_seconds, seconds);
      count++;
    }
    topo.all_to_all_gpu_gbps[from] = max_seconds > 0.0
        ? (static_cast<double>(bytes) * iters * count) / max_seconds / 1e9
        : 0.0;
    topo.all_to_all_total_gbps += topo.all_to_all_gpu_gbps[from];
  }

  double off_sum = 0.0;
  int off_count = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      off_sum += sequential_bw[i][j];
      off_count++;
    }
  }
  double avg_offdiag = off_count ? off_sum / off_count : 0.0;
  topo.pcie_link_score = avg_offdiag / 63.0;
  double ideal_dense = avg_offdiag * n;
  topo.dense_interconnect_score = ideal_dense > 0.0 ? topo.all_to_all_total_gbps / ideal_dense : 0.0;
  return topo;
}

__global__ void remote_read_latency_kernel(
    const uint32_t* __restrict__ remote,
    uint32_t* __restrict__ sink,
    int iters) {
  uint32_t idx = 0;
  volatile const uint32_t* ptr = remote;
  for (int i = 0; i < iters; i++) {
    idx = ptr[idx];
  }
  sink[0] = idx;
}

static std::vector<double> measure_concurrent_remote_reads(
    int n,
    int latency_iters,
    const std::vector<std::pair<int, int>>& pairs) {
  const int slots = 1024;
  const int stride_words = 32;  // 128-byte spacing, matching the upstream remote-read probe.
  const int words = slots * stride_words;
  const size_t bytes = words * sizeof(uint32_t);
  std::vector<void*> remote(n, nullptr);
  std::vector<void*> sink(pairs.size(), nullptr);
  std::vector<cudaStream_t> streams(pairs.size(), nullptr);
  std::vector<cudaEvent_t> start(pairs.size(), nullptr), stop(pairs.size(), nullptr);

  std::vector<uint32_t> chain(words, 0);
  for (int i = 0; i < slots; i++) {
    chain[i * stride_words] = ((i + 1) % slots) * stride_words;
  }
  for (int d = 0; d < n; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaMalloc(&remote[d], bytes));
    CHECK_CUDA(cudaMemcpy(remote[d], chain.data(), bytes, cudaMemcpyHostToDevice));
  }
  for (size_t p = 0; p < pairs.size(); p++) {
    int reader = pairs[p].first;
    CHECK_CUDA(cudaSetDevice(reader));
    CHECK_CUDA(cudaMalloc(&sink[p], sizeof(uint32_t)));
    CHECK_CUDA(cudaStreamCreate(&streams[p]));
    CHECK_CUDA(cudaEventCreate(&start[p]));
    CHECK_CUDA(cudaEventCreate(&stop[p]));
  }

  for (size_t p = 0; p < pairs.size(); p++) {
    int reader = pairs[p].first;
    int peer = pairs[p].second;
    CHECK_CUDA(cudaSetDevice(reader));
    for (int w = 0; w < 3; w++) {
      remote_read_latency_kernel<<<1, 1, 0, streams[p]>>>(
          (const uint32_t*)remote[peer], (uint32_t*)sink[p], latency_iters);
    }
  }
  for (size_t p = 0; p < pairs.size(); p++) {
    CHECK_CUDA(cudaSetDevice(pairs[p].first));
    CHECK_CUDA(cudaStreamSynchronize(streams[p]));
  }

  for (size_t p = 0; p < pairs.size(); p++) {
    CHECK_CUDA(cudaSetDevice(pairs[p].first));
    CHECK_CUDA(cudaEventRecord(start[p], streams[p]));
    remote_read_latency_kernel<<<1, 1, 0, streams[p]>>>(
        (const uint32_t*)remote[pairs[p].second], (uint32_t*)sink[p], latency_iters);
    CHECK_CUDA(cudaEventRecord(stop[p], streams[p]));
  }

  std::vector<double> us(pairs.size(), 0.0);
  for (size_t p = 0; p < pairs.size(); p++) {
    CHECK_CUDA(cudaSetDevice(pairs[p].first));
    CHECK_CUDA(cudaEventSynchronize(stop[p]));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start[p], stop[p]));
    us[p] = (double)ms * 1000.0 / latency_iters;
  }

  for (size_t p = 0; p < pairs.size(); p++) {
    CHECK_CUDA(cudaSetDevice(pairs[p].first));
    cudaEventDestroy(start[p]);
    cudaEventDestroy(stop[p]);
    cudaStreamDestroy(streams[p]);
    cudaFree(sink[p]);
  }
  for (int d = 0; d < n; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    cudaFree(remote[d]);
  }
  return us;
}

struct LatencyTopology {
  std::vector<std::vector<double>> sequential_us;
  std::vector<double> staggered_avg_us;
  std::vector<double> single_reader_all_peers_us;
  std::vector<double> all_read_all_peers_us;
  double min_sequential_us = 0.0;
  double avg_sequential_us = 0.0;
  double mean_full_load_us = 0.0;
  double effective_full_load_us = 0.0;
};

static LatencyTopology measure_latency_topology(int n, int latency_iters) {
  LatencyTopology topo;
  topo.sequential_us.assign(n, std::vector<double>(n, 0.0));

  double seq_sum = 0.0;
  int seq_count = 0;
  topo.min_sequential_us = 1e30;
  for (int reader = 0; reader < n; reader++) {
    for (int peer = 0; peer < n; peer++) {
      if (reader == peer) continue;
      auto values = measure_concurrent_remote_reads(n, latency_iters, {{reader, peer}});
      topo.sequential_us[reader][peer] = values[0];
      topo.min_sequential_us = std::min(topo.min_sequential_us, values[0]);
      seq_sum += values[0];
      seq_count++;
    }
  }
  topo.avg_sequential_us = seq_count ? seq_sum / seq_count : 0.0;

  for (int offset = 1; offset < n; offset++) {
    std::vector<std::pair<int, int>> pairs;
    for (int reader = 0; reader < n; reader++) {
      pairs.push_back({reader, (reader + offset) % n});
    }
    auto values = measure_concurrent_remote_reads(n, latency_iters, pairs);
    topo.staggered_avg_us.push_back(std::accumulate(values.begin(), values.end(), 0.0) / values.size());
  }

  topo.single_reader_all_peers_us.resize(n, 0.0);
  for (int reader = 0; reader < n; reader++) {
    std::vector<std::pair<int, int>> pairs;
    for (int peer = 0; peer < n; peer++) {
      if (peer != reader) pairs.push_back({reader, peer});
    }
    auto values = measure_concurrent_remote_reads(n, latency_iters, pairs);
    topo.single_reader_all_peers_us[reader] = *std::max_element(values.begin(), values.end());
  }

  std::vector<std::pair<int, int>> all_pairs;
  for (int reader = 0; reader < n; reader++) {
    for (int peer = 0; peer < n; peer++) {
      if (reader != peer) all_pairs.push_back({reader, peer});
    }
  }
  auto all_values = measure_concurrent_remote_reads(n, latency_iters, all_pairs);
  topo.all_read_all_peers_us.resize(n, 0.0);
  for (int reader = 0; reader < n; reader++) {
    double max_us = 0.0;
    for (size_t p = 0; p < all_pairs.size(); p++) {
      if (all_pairs[p].first == reader) max_us = std::max(max_us, all_values[p]);
    }
    topo.all_read_all_peers_us[reader] = max_us;
  }
  topo.mean_full_load_us = std::accumulate(
      topo.all_read_all_peers_us.begin(), topo.all_read_all_peers_us.end(), 0.0) / n;
  topo.effective_full_load_us = *std::max_element(
      topo.all_read_all_peers_us.begin(), topo.all_read_all_peers_us.end());
  return topo;
}

struct AllreduceRow {
  size_t size_bytes;
  double custom_us;
  double nccl_us;
  double custom_bus_bw_gbps;
  double nccl_bus_bw_gbps;
  const char* winner;
  double ratio;
};

static double allreduce_bus_bw(int n, size_t bytes, double us) {
  if (us <= 0.0) return 0.0;
  return bytes * 2.0 * (n - 1) / n / (us / 1e6) / 1e9;
}

static double bench_custom_ar(
    int n,
    size_t bytes,
    std::vector<void*>& input,
    std::vector<void*>& output,
    std::vector<pcie_ar::Signal*>& sigs,
    std::vector<void*>& rd_dev,
    pcie_ar::RankSignals& rs,
    std::vector<cudaStream_t>& streams,
    int iters_override) {
  using T = half;
  constexpr int d = pcie_ar::packed_t<T>::P::size;
  int num_elements = bytes / sizeof(T);
  int packed_size = num_elements / d;
  int threads = 512;
  int blocks = std::min(36, (packed_size + threads - 1) / threads);
  blocks = std::max(blocks, 1);
  int iters = iters_override > 0 ? iters_override : ((bytes <= 1024 * 1024) ? 2000 : 200);

  for (int i = 0; i < n; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemset(sigs[i], 0, sizeof(pcie_ar::Signal)));
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  {
    std::barrier bar(n);
    std::vector<std::thread> tv;
    for (int i = 0; i < n; i++) {
      tv.emplace_back([&, i]() {
        CHECK_CUDA(cudaSetDevice(i));
        bar.arrive_and_wait();
        for (int w = 0; w < 20; w++) {
          pcie_ar::launch_kernel<T>(
              n, blocks, threads, streams[i],
              (pcie_ar::RankData*)rd_dev[i], rs, sigs[i], (T*)output[i], i, packed_size);
        }
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
      });
    }
    for (auto& t : tv) t.join();
  }

  std::vector<cudaGraph_t> graphs(n);
  std::vector<cudaGraphExec_t> execs(n);
  for (int i = 0; i < n; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeThreadLocal));
    pcie_ar::launch_kernel<T>(
        n, blocks, threads, streams[i],
        (pcie_ar::RankData*)rd_dev[i], rs, sigs[i], (T*)output[i], i, packed_size);
    CHECK_CUDA(cudaStreamEndCapture(streams[i], &graphs[i]));
    CHECK_CUDA(cudaGraphInstantiate(&execs[i], graphs[i], 0));
  }

  std::barrier tbar(n);
  std::vector<double> per_gpu_us(n);
  std::vector<std::thread> tv;
  for (int i = 0; i < n; i++) {
    tv.emplace_back([&, i]() {
      CHECK_CUDA(cudaSetDevice(i));
      cudaEvent_t e0, e1;
      CHECK_CUDA(cudaEventCreate(&e0));
      CHECK_CUDA(cudaEventCreate(&e1));
      tbar.arrive_and_wait();
      CHECK_CUDA(cudaEventRecord(e0, streams[i]));
      for (int it = 0; it < iters; it++) {
        CHECK_CUDA(cudaGraphLaunch(execs[i], streams[i]));
      }
      CHECK_CUDA(cudaEventRecord(e1, streams[i]));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
      float ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
      per_gpu_us[i] = (double)ms * 1000.0 / iters;
      CHECK_CUDA(cudaEventDestroy(e0));
      CHECK_CUDA(cudaEventDestroy(e1));
    });
  }
  for (auto& t : tv) t.join();

  for (int i = 0; i < n; i++) {
    CHECK_CUDA(cudaGraphExecDestroy(execs[i]));
    CHECK_CUDA(cudaGraphDestroy(graphs[i]));
  }

  double max_us = 0.0;
  for (double us : per_gpu_us) max_us = std::max(max_us, us);
  return max_us;
}

static double bench_nccl_ar(
    int n,
    size_t bytes,
    std::vector<void*>& input,
    std::vector<void*>& output,
    std::vector<ncclComm_t>& comms,
    std::vector<cudaStream_t>& streams,
    int iters_override) {
  int count = bytes / sizeof(half);
  int iters = iters_override > 0 ? iters_override : ((bytes <= 1024 * 1024) ? 2000 : 200);

  for (int w = 0; w < 20; w++) {
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < n; i++) {
      CHECK_NCCL(ncclAllReduce(input[i], output[i], count, ncclFloat16, ncclSum, comms[i], streams[i]));
    }
    CHECK_NCCL(ncclGroupEnd());
  }
  for (int i = 0; i < n; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));
  }

  std::vector<cudaEvent_t> e0(n), e1(n);
  for (int i = 0; i < n; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaEventCreate(&e0[i]));
    CHECK_CUDA(cudaEventCreate(&e1[i]));
    CHECK_CUDA(cudaEventRecord(e0[i], streams[i]));
  }
  for (int it = 0; it < iters; it++) {
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < n; i++) {
      CHECK_NCCL(ncclAllReduce(input[i], output[i], count, ncclFloat16, ncclSum, comms[i], streams[i]));
    }
    CHECK_NCCL(ncclGroupEnd());
  }

  double max_us = 0.0;
  for (int i = 0; i < n; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaEventRecord(e1[i], streams[i]));
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, e0[i], e1[i]));
    double us = (double)ms * 1000.0 / iters;
    max_us = std::max(max_us, us);
    CHECK_CUDA(cudaEventDestroy(e0[i]));
    CHECK_CUDA(cudaEventDestroy(e1[i]));
  }
  return max_us;
}

static std::vector<AllreduceRow> measure_allreduce(
    int n, const std::vector<size_t>& sizes_bytes, int iters_override) {
  using T = half;
  constexpr int d = pcie_ar::packed_t<T>::P::size;
  if (n < 2 || n > 8) {
    std::fprintf(stderr, "Allreduce compare mode requires 2-8 GPUs, got %d\n", n);
    std::exit(2);
  }

  std::vector<int> devices(n);
  std::iota(devices.begin(), devices.end(), 0);
  std::vector<ncclComm_t> comms(n);
  CHECK_NCCL(ncclCommInitAll(comms.data(), n, devices.data()));

  size_t max_bytes = *std::max_element(sizes_bytes.begin(), sizes_bytes.end());
  std::vector<void*> input(n, nullptr), output(n, nullptr);
  std::vector<pcie_ar::Signal*> sigs(n, nullptr);
  std::vector<void*> rd_dev(n, nullptr);
  std::vector<cudaStream_t> streams(n, nullptr);
  for (int d = 0; d < n; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaMalloc(&input[d], max_bytes));
    CHECK_CUDA(cudaMalloc(&output[d], max_bytes));
    CHECK_CUDA(cudaMemset(input[d], d + 1, max_bytes));
    CHECK_CUDA(cudaMemset(output[d], 0, max_bytes));
    CHECK_CUDA(cudaMalloc((void**)&sigs[d], sizeof(pcie_ar::Signal)));
    CHECK_CUDA(cudaMemset(sigs[d], 0, sizeof(pcie_ar::Signal)));
    CHECK_CUDA(cudaStreamCreate(&streams[d]));
  }

  pcie_ar::RankData rd;
  for (int i = 0; i < n; i++) rd.ptrs[i] = input[i];
  for (int i = 0; i < n; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&rd_dev[i], sizeof(pcie_ar::RankData)));
    CHECK_CUDA(cudaMemcpy(rd_dev[i], &rd, sizeof(pcie_ar::RankData), cudaMemcpyHostToDevice));
  }
  pcie_ar::RankSignals rs;
  for (int i = 0; i < n; i++) rs.signals[i] = sigs[i];

  std::vector<AllreduceRow> rows;
  for (size_t bytes : sizes_bytes) {
    int num_elements = bytes / sizeof(T);
    if (num_elements % d != 0) continue;
    double custom_us = bench_custom_ar(n, bytes, input, output, sigs, rd_dev, rs, streams, iters_override);
    double nccl_us = bench_nccl_ar(n, bytes, input, output, comms, streams, iters_override);
    const char* winner = custom_us < nccl_us ? "custom" : "nccl";
    double ratio = custom_us < nccl_us ? nccl_us / custom_us : custom_us / nccl_us;
    rows.push_back({
        bytes,
        custom_us,
        nccl_us,
        allreduce_bus_bw(n, bytes, custom_us),
        allreduce_bus_bw(n, bytes, nccl_us),
        winner,
        ratio,
    });
  }

  for (int d = 0; d < n; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    cudaStreamDestroy(streams[d]);
    cudaFree(input[d]);
    cudaFree(output[d]);
    cudaFree(sigs[d]);
    cudaFree(rd_dev[d]);
    ncclCommDestroy(comms[d]);
  }
  return rows;
}

static void print_matrix_int(const std::vector<std::vector<int>>& m) {
  std::printf("[");
  for (size_t i = 0; i < m.size(); i++) {
    if (i) std::printf(",");
    std::printf("[");
    for (size_t j = 0; j < m[i].size(); j++) {
      if (j) std::printf(",");
      std::printf("%d", m[i][j]);
    }
    std::printf("]");
  }
  std::printf("]");
}

static void print_matrix_double(const std::vector<std::vector<double>>& m) {
  std::printf("[");
  for (size_t i = 0; i < m.size(); i++) {
    if (i) std::printf(",");
    std::printf("[");
    for (size_t j = 0; j < m[i].size(); j++) {
      if (j) std::printf(",");
      std::printf("%.3f", m[i][j]);
    }
    std::printf("]");
  }
  std::printf("]");
}

static void print_vector_double(const std::vector<double>& v) {
  std::printf("[");
  for (size_t i = 0; i < v.size(); i++) {
    if (i) std::printf(",");
    std::printf("%.3f", v[i]);
  }
  std::printf("]");
}

static void print_pairs(const std::vector<std::pair<int, int>>& pairs) {
  std::printf("[");
  for (size_t i = 0; i < pairs.size(); i++) {
    if (i) std::printf(",");
    std::printf("[%d,%d]", pairs[i].first, pairs[i].second);
  }
  std::printf("]");
}

static void print_concurrent_copy_result(const ConcurrentCopyResult& r) {
  std::printf("{\"pairs\":");
  print_pairs(r.pairs);
  std::printf(",\"pair_gbps\":");
  print_vector_double(r.pair_gbps);
  std::printf(",\"avg_gbps\":%.3f,\"total_gbps\":%.3f}", r.avg_gbps, r.total_gbps);
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  int visible = 0;
  CHECK_CUDA(cudaGetDeviceCount(&visible));
  if (visible <= 0) {
    std::fprintf(stderr, "No CUDA devices visible\n");
    return 1;
  }
  int n = visible;
  if (opt.max_gpus > 0) n = std::min(n, opt.max_gpus);
  if (n > 8) n = 8;
  if (n < 2) {
    std::fprintf(stderr, "Need at least two visible GPUs\n");
    return 1;
  }

  auto can = peer_matrix(n);
  enable_peers(can);

  std::vector<std::vector<double>> bw;
  BandwidthTopology bw_topology;
  bool have_bw_topology = false;
  if (opt.mode == "all" || opt.mode == "bandwidth") {
    size_t bytes = static_cast<size_t>(opt.size_mb) * 1024 * 1024;
    bw = measure_bandwidth(n, bytes, opt.warmup, opt.iters);
    bw_topology = measure_bandwidth_topology(n, bytes, opt.warmup, opt.iters, bw);
    have_bw_topology = true;
  }

  LatencyTopology latency_topology;
  bool have_latency = false;
  if (opt.mode == "all" || opt.mode == "latency") {
    latency_topology = measure_latency_topology(n, opt.latency_iters);
    have_latency = true;
  }

  std::vector<AllreduceRow> ar;
  if (opt.mode == "all" || opt.mode == "allreduce") {
    ar = measure_allreduce(n, opt.allreduce_sizes_bytes, opt.iters);
  }

  double off_sum = 0.0, off_min = 0.0, off_max = 0.0;
  int off_count = 0;
  if (!bw.empty()) {
    off_min = 1e30;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (i == j) continue;
        off_sum += bw[i][j];
        off_min = std::min(off_min, bw[i][j]);
        off_max = std::max(off_max, bw[i][j]);
        off_count++;
      }
    }
  }

  std::printf("{");
  std::printf("\"tool\":\"llm_p2pmark\",");
  std::printf("\"version\":1,");
  std::printf("\"visible_gpu_count\":%d,", visible);
  std::printf("\"gpu_count\":%d,", n);
  std::printf("\"gpu_count_clamped\":%s,", (visible != n && opt.max_gpus <= 0) ? "true" : "false");
  std::printf("\"mode\":\"%s\",", opt.mode.c_str());
  std::printf("\"size_mb\":%d,", opt.size_mb);
  std::printf("\"iters\":%d,", opt.iters);
  std::printf("\"warmup\":%d,", opt.warmup);
  std::printf("\"latency_iters\":%d,", opt.latency_iters);
  std::printf("\"peer_access\":");
  print_matrix_int(can);
  std::printf(",");
  if (!bw.empty()) {
    std::printf("\"bandwidth_gbps\":");
    print_matrix_double(bw);
    std::printf(",\"bandwidth_summary\":{");
    std::printf("\"avg_offdiag_gbps\":%.3f,", off_count ? off_sum / off_count : 0.0);
    std::printf("\"min_offdiag_gbps\":%.3f,", off_count ? off_min : 0.0);
    std::printf("\"max_offdiag_gbps\":%.3f", off_count ? off_max : 0.0);
    std::printf("},");
  } else {
    std::printf("\"bandwidth_gbps\":[],\"bandwidth_summary\":{},");
  }
  std::printf("\"bandwidth_topology\":");
  if (have_bw_topology) {
    std::printf("{\"staggered\":[");
    for (size_t i = 0; i < bw_topology.staggered.size(); i++) {
      if (i) std::printf(",");
      std::printf("{\"offset\":%zu,\"result\":", i + 1);
      print_concurrent_copy_result(bw_topology.staggered[i]);
      std::printf("}");
    }
    std::printf("],\"ring\":");
    print_concurrent_copy_result(bw_topology.ring);
    std::printf(",\"single_writer_gbps\":");
    print_vector_double(bw_topology.single_writer_gbps);
    std::printf(",\"all_to_all_gpu_gbps\":");
    print_vector_double(bw_topology.all_to_all_gpu_gbps);
    std::printf(",\"all_to_all_total_gbps\":%.3f", bw_topology.all_to_all_total_gbps);
    std::printf(",\"pcie_link_score\":%.6f", bw_topology.pcie_link_score);
    std::printf(",\"dense_interconnect_score\":%.6f", bw_topology.dense_interconnect_score);
    std::printf("},");
  } else {
    std::printf("{},");
  }
  std::printf("\"latency\":");
  if (have_latency) {
    std::printf("{\"sequential_us\":");
    print_matrix_double(latency_topology.sequential_us);
    std::printf(",\"staggered_avg_us\":");
    print_vector_double(latency_topology.staggered_avg_us);
    std::printf(",\"single_reader_all_peers_us\":");
    print_vector_double(latency_topology.single_reader_all_peers_us);
    std::printf(",\"all_read_all_peers_us\":");
    print_vector_double(latency_topology.all_read_all_peers_us);
    std::printf(",\"min_sequential_us\":%.3f", latency_topology.min_sequential_us);
    std::printf(",\"avg_sequential_us\":%.3f", latency_topology.avg_sequential_us);
    std::printf(",\"mean_full_load_us\":%.3f", latency_topology.mean_full_load_us);
    std::printf(",\"effective_full_load_us\":%.3f", latency_topology.effective_full_load_us);
    std::printf("},");
  } else {
    std::printf("{},");
  }
  std::printf("\"allreduce\":[");
  for (size_t i = 0; i < ar.size(); i++) {
    if (i) std::printf(",");
    double best_bus = std::strcmp(ar[i].winner, "custom") == 0
        ? ar[i].custom_bus_bw_gbps : ar[i].nccl_bus_bw_gbps;
    std::printf(
        "{\"size_bytes\":%zu,\"custom_us\":%.3f,\"nccl_us\":%.3f,"
        "\"custom_bus_bw_gbps\":%.3f,\"nccl_bus_bw_gbps\":%.3f,"
        "\"best_bus_bw_gbps\":%.3f,\"winner\":\"%s\",\"ratio\":%.3f}",
        ar[i].size_bytes, ar[i].custom_us, ar[i].nccl_us,
        ar[i].custom_bus_bw_gbps, ar[i].nccl_bus_bw_gbps,
        best_bus, ar[i].winner, ar[i].ratio);
  }
  std::printf("]}\n");
  return 0;
}
