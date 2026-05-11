#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <x86intrin.h>

struct Options {
  int size_mb = 512;
  int latency_mb = 256;
  int iters = 5;
  int warmup = 1;
  int threads = 0;
  int latency_iters = 5000000;
  int max_nodes = 0;
};

static void die(const char* msg) {
  std::fprintf(stderr, "%s\n", msg);
  std::exit(1);
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      std::printf(
          "Usage: %s [--size-mb N] [--latency-mb N] [--iters N] [--warmup N]\n"
          "          [--threads N] [--latency-iters N] [--max-nodes N]\n",
          argv[0]);
      std::exit(0);
    } else if (!std::strcmp(argv[i], "--size-mb")) {
      opt.size_mb = std::atoi(need(argv[i]));
    } else if (!std::strcmp(argv[i], "--latency-mb")) {
      opt.latency_mb = std::atoi(need(argv[i]));
    } else if (!std::strcmp(argv[i], "--iters")) {
      opt.iters = std::atoi(need(argv[i]));
    } else if (!std::strcmp(argv[i], "--warmup")) {
      opt.warmup = std::atoi(need(argv[i]));
    } else if (!std::strcmp(argv[i], "--threads")) {
      opt.threads = std::atoi(need(argv[i]));
    } else if (!std::strcmp(argv[i], "--latency-iters")) {
      opt.latency_iters = std::atoi(need(argv[i]));
    } else if (!std::strcmp(argv[i], "--max-nodes")) {
      opt.max_nodes = std::atoi(need(argv[i]));
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      std::exit(2);
    }
  }
  if (opt.size_mb <= 0 || opt.latency_mb <= 0 || opt.iters <= 0 || opt.warmup < 0 ||
      opt.threads < 0 || opt.latency_iters <= 0 || opt.max_nodes < 0) {
    die("invalid numeric option");
  }
  return opt;
}

static std::vector<int> cpus_for_node(int node) {
  std::vector<int> cpus;
  struct bitmask* mask = numa_allocate_cpumask();
  if (numa_node_to_cpus(node, mask) != 0) {
    numa_free_cpumask(mask);
    return cpus;
  }
  for (unsigned i = 0; i < mask->size; i++) {
    if (numa_bitmask_isbitset(mask, i)) cpus.push_back((int)i);
  }
  numa_free_cpumask(mask);
  return cpus;
}

static void pin_cpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    std::perror("pthread_setaffinity_np");
    std::exit(1);
  }
}

static void* alloc_node(size_t bytes, int node) {
  void* p = numa_alloc_onnode(bytes, node);
  if (!p) die("numa_alloc_onnode failed");
  std::memset(p, 1, bytes);
  return p;
}

static void flush_range(void* ptr, size_t bytes) {
  if (!ptr || bytes == 0) return;
  auto* p = static_cast<char*>(ptr);
  for (size_t off = 0; off < bytes; off += 64) {
    _mm_clflush(p + off);
  }
  _mm_mfence();
}

enum class Op { Read, Write, Copy };

static void flush_for_op(Op op, uint8_t* src, uint8_t* dst, size_t bytes) {
  if (op == Op::Read) {
    flush_range(src, bytes);
  } else if (op == Op::Write) {
    flush_range(dst, bytes);
  } else {
    flush_range(src, bytes);
    flush_range(dst, bytes);
  }
}

struct Worker {
  Op op;
  int cpu;
  uint8_t* src;
  uint8_t* dst;
  size_t begin;
  size_t end;
  int loops;
  pthread_barrier_t* barrier;
  double seconds = 0.0;
  uint64_t sink = 0;
};

static void* worker_main(void* arg) {
  Worker* w = static_cast<Worker*>(arg);
  pin_cpu(w->cpu);
  pthread_barrier_wait(w->barrier);
  auto t0 = std::chrono::steady_clock::now();
  if (w->op == Op::Read) {
    uint64_t sum = 0;
    for (int it = 0; it < w->loops; it++) {
      for (size_t p = w->begin; p < w->end; p += 64) {
        sum += *reinterpret_cast<volatile uint64_t*>(w->src + p);
      }
    }
    w->sink = sum;
  } else if (w->op == Op::Write) {
    for (int it = 0; it < w->loops; it++) {
      uint64_t value = 0x9e3779b97f4a7c15ULL + (uint64_t)it;
      for (size_t p = w->begin; p < w->end; p += 64) {
        auto* q = reinterpret_cast<volatile uint64_t*>(w->dst + p);
        q[0] = value;
        q[1] = value + 1;
        q[2] = value + 2;
        q[3] = value + 3;
        q[4] = value + 4;
        q[5] = value + 5;
        q[6] = value + 6;
        q[7] = value + 7;
      }
    }
  } else {
    for (int it = 0; it < w->loops; it++) {
      std::memcpy(w->dst + w->begin, w->src + w->begin, w->end - w->begin);
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  w->seconds = std::chrono::duration<double>(t1 - t0).count();
  return nullptr;
}

static double run_op(
    Op op,
    const std::vector<int>& cpus,
    int threads,
    uint8_t* src,
    uint8_t* dst,
    size_t bytes,
    int loops) {
  threads = std::max(1, std::min(threads, (int)cpus.size()));
  std::vector<Worker> workers(threads);
  std::vector<pthread_t> tids(threads);
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, nullptr, threads);
  size_t chunk = bytes / threads;
  chunk = (chunk / 4096) * 4096;
  for (int i = 0; i < threads; i++) {
    size_t begin = i * chunk;
    size_t end = (i == threads - 1) ? bytes : (i + 1) * chunk;
    workers[i] = Worker{op, cpus[i % cpus.size()], src, dst, begin, end, loops, &barrier};
    pthread_create(&tids[i], nullptr, worker_main, &workers[i]);
  }
  double max_seconds = 0.0;
  for (int i = 0; i < threads; i++) {
    pthread_join(tids[i], nullptr);
    max_seconds = std::max(max_seconds, workers[i].seconds);
  }
  pthread_barrier_destroy(&barrier);
  return max_seconds;
}

static double measure_op(
    Op op,
    const std::vector<int>& cpus,
    int threads,
    uint8_t* src,
    uint8_t* dst,
    size_t bytes,
    int warmup,
    int iters) {
  flush_for_op(op, src, dst, bytes);
  if (warmup > 0) run_op(op, cpus, threads, src, dst, bytes, warmup);
  flush_for_op(op, src, dst, bytes);
  double seconds = run_op(op, cpus, threads, src, dst, bytes, iters);
  return seconds > 0.0 ? (double)bytes * iters / seconds / 1e9 : 0.0;
}

static double measure_bidirectional_remote(
    Op op,
    const std::vector<int>& cpus0,
    const std::vector<int>& cpus1,
    int threads,
    uint8_t* mem0,
    uint8_t* mem1,
    uint8_t* dst0,
    uint8_t* dst1,
    size_t bytes,
    int warmup,
    int iters) {
  auto run = [&](int loops) {
    int t = std::max(1, threads);
    int total_threads = t * 2;
    std::vector<Worker> workers(total_threads);
    std::vector<pthread_t> tids(total_threads);
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, total_threads);
    size_t chunk = bytes / t;
    chunk = (chunk / 4096) * 4096;
    for (int i = 0; i < t; i++) {
      size_t begin = i * chunk;
      size_t end = (i == t - 1) ? bytes : (i + 1) * chunk;
      if (op == Op::Read) {
        workers[i] = Worker{op, cpus0[i % cpus0.size()], mem1, nullptr, begin, end, loops, &barrier};
        workers[i + t] = Worker{op, cpus1[i % cpus1.size()], mem0, nullptr, begin, end, loops, &barrier};
      } else if (op == Op::Write) {
        workers[i] = Worker{op, cpus0[i % cpus0.size()], nullptr, mem1, begin, end, loops, &barrier};
        workers[i + t] = Worker{op, cpus1[i % cpus1.size()], nullptr, mem0, begin, end, loops, &barrier};
      } else {
        workers[i] = Worker{op, cpus0[i % cpus0.size()], mem1, dst0, begin, end, loops, &barrier};
        workers[i + t] = Worker{op, cpus1[i % cpus1.size()], mem0, dst1, begin, end, loops, &barrier};
      }
      pthread_create(&tids[i], nullptr, worker_main, &workers[i]);
      pthread_create(&tids[i + t], nullptr, worker_main, &workers[i + t]);
    }
    double max_seconds = 0.0;
    for (int i = 0; i < total_threads; i++) {
      pthread_join(tids[i], nullptr);
      max_seconds = std::max(max_seconds, workers[i].seconds);
    }
    pthread_barrier_destroy(&barrier);
    return max_seconds;
  };
  auto flush_pair = [&]() {
    if (op == Op::Read) {
      flush_range(mem0, bytes);
      flush_range(mem1, bytes);
    } else if (op == Op::Write) {
      flush_range(mem0, bytes);
      flush_range(mem1, bytes);
    } else {
      flush_range(mem0, bytes);
      flush_range(mem1, bytes);
      flush_range(dst0, bytes);
      flush_range(dst1, bytes);
    }
  };
  flush_pair();
  if (warmup > 0) run(warmup);
  flush_pair();
  double seconds = run(iters);
  return seconds > 0.0 ? (double)bytes * iters * 2.0 / seconds / 1e9 : 0.0;
}

static double measure_latency_us(const std::vector<int>& cpus, int mem_node, size_t bytes, int iters) {
  const size_t stride_words = 16;  // 64-byte spacing.
  size_t slots = std::max<size_t>(1024, bytes / 64);
  size_t words = slots * stride_words;
  uint32_t* arr = static_cast<uint32_t*>(alloc_node(words * sizeof(uint32_t), mem_node));
  std::vector<uint32_t> perm(slots);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 rng(0x5eed1234u + mem_node);
  std::shuffle(perm.begin(), perm.end(), rng);
  for (size_t i = 0; i < slots; i++) {
    arr[(size_t)perm[i] * stride_words] = perm[(i + 1) % slots] * stride_words;
  }

  struct LatArg {
    int cpu;
    volatile uint32_t* arr;
    int iters;
    double seconds;
    uint32_t sink;
  } arg{cpus.front(), arr, iters, 0.0, 0};
  auto fn = [](void* p) -> void* {
    auto* a = static_cast<LatArg*>(p);
    pin_cpu(a->cpu);
    uint32_t idx = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < a->iters; i++) {
      idx = a->arr[idx];
    }
    auto t1 = std::chrono::steady_clock::now();
    a->sink = idx;
    a->seconds = std::chrono::duration<double>(t1 - t0).count();
    return nullptr;
  };
  pthread_t tid;
  flush_range(arr, words * sizeof(uint32_t));
  pthread_create(&tid, nullptr, fn, &arg);
  pthread_join(tid, nullptr);
  numa_free(arr, words * sizeof(uint32_t));
  return arg.seconds * 1e6 / iters;
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

static void print_cpu_lists(const std::vector<std::vector<int>>& cpus) {
  std::printf("[");
  for (size_t i = 0; i < cpus.size(); i++) {
    if (i) std::printf(",");
    std::printf("[");
    for (size_t j = 0; j < cpus[i].size(); j++) {
      if (j) std::printf(",");
      std::printf("%d", cpus[i][j]);
    }
    std::printf("]");
  }
  std::printf("]");
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  if (numa_available() < 0) die("NUMA is not available");
  int max_node = numa_max_node();
  int nodes = max_node + 1;
  if (opt.max_nodes > 0) nodes = std::min(nodes, opt.max_nodes);
  if (nodes < 1) die("no NUMA nodes found");

  std::vector<std::vector<int>> node_cpus(nodes);
  int min_cpus = 1 << 30;
  for (int n = 0; n < nodes; n++) {
    node_cpus[n] = cpus_for_node(n);
    if (node_cpus[n].empty()) {
      std::fprintf(stderr, "node %d has no CPUs\n", n);
      return 1;
    }
    min_cpus = std::min(min_cpus, (int)node_cpus[n].size());
  }
  int threads = opt.threads > 0 ? opt.threads : std::min(64, min_cpus);
  threads = std::max(1, threads);

  size_t bytes = (size_t)opt.size_mb * 1024 * 1024;
  size_t latency_bytes = (size_t)opt.latency_mb * 1024 * 1024;

  std::vector<uint8_t*> buffers(nodes, nullptr);
  std::vector<uint8_t*> local_dst(nodes, nullptr);
  for (int n = 0; n < nodes; n++) {
    buffers[n] = static_cast<uint8_t*>(alloc_node(bytes, n));
    local_dst[n] = static_cast<uint8_t*>(alloc_node(bytes, n));
  }

  std::vector<std::vector<int>> distances(nodes, std::vector<int>(nodes, 0));
  std::vector<std::vector<double>> read_gbps(nodes, std::vector<double>(nodes, 0.0));
  std::vector<std::vector<double>> write_gbps(nodes, std::vector<double>(nodes, 0.0));
  std::vector<std::vector<double>> copy_gbps(nodes, std::vector<double>(nodes, 0.0));
  std::vector<std::vector<double>> latency_us(nodes, std::vector<double>(nodes, 0.0));

  for (int cpu_node = 0; cpu_node < nodes; cpu_node++) {
    for (int mem_node = 0; mem_node < nodes; mem_node++) {
      distances[cpu_node][mem_node] = numa_distance(cpu_node, mem_node);
      read_gbps[cpu_node][mem_node] = measure_op(
          Op::Read, node_cpus[cpu_node], threads, buffers[mem_node], nullptr, bytes, opt.warmup, opt.iters);
      write_gbps[cpu_node][mem_node] = measure_op(
          Op::Write, node_cpus[cpu_node], threads, nullptr, buffers[mem_node], bytes, opt.warmup, opt.iters);
      copy_gbps[cpu_node][mem_node] = measure_op(
          Op::Copy, node_cpus[cpu_node], threads, buffers[mem_node], local_dst[cpu_node], bytes, opt.warmup, opt.iters);
      latency_us[cpu_node][mem_node] = measure_latency_us(node_cpus[cpu_node], mem_node, latency_bytes, opt.latency_iters);
    }
  }

  double bidir_remote_read_gbps = 0.0;
  double bidir_remote_write_gbps = 0.0;
  double bidir_remote_copy_gbps = 0.0;
  if (nodes >= 2) {
    bidir_remote_read_gbps = measure_bidirectional_remote(
        Op::Read, node_cpus[0], node_cpus[1], threads, buffers[0], buffers[1],
        local_dst[0], local_dst[1], bytes, opt.warmup, opt.iters);
    bidir_remote_write_gbps = measure_bidirectional_remote(
        Op::Write, node_cpus[0], node_cpus[1], threads, buffers[0], buffers[1],
        local_dst[0], local_dst[1], bytes, opt.warmup, opt.iters);
    bidir_remote_copy_gbps = measure_bidirectional_remote(
        Op::Copy, node_cpus[0], node_cpus[1], threads, buffers[0], buffers[1],
        local_dst[0], local_dst[1], bytes, opt.warmup, opt.iters);
  }

  std::printf("{");
  std::printf("\"tool\":\"llm_amd_fabric\",");
  std::printf("\"version\":2,");
  std::printf("\"numa_nodes\":%d,", nodes);
  std::printf("\"size_mb\":%d,", opt.size_mb);
  std::printf("\"latency_mb\":%d,", opt.latency_mb);
  std::printf("\"iters\":%d,", opt.iters);
  std::printf("\"warmup\":%d,", opt.warmup);
  std::printf("\"threads_per_node\":%d,", threads);
  std::printf("\"latency_iters\":%d,", opt.latency_iters);
  std::printf("\"node_cpus\":");
  print_cpu_lists(node_cpus);
  std::printf(",\"distance\":");
  print_matrix_int(distances);
  std::printf(",\"read_gbps\":");
  print_matrix_double(read_gbps);
  std::printf(",\"write_gbps\":");
  print_matrix_double(write_gbps);
  std::printf(",\"copy_gbps\":");
  print_matrix_double(copy_gbps);
  std::printf(",\"latency_us\":");
  print_matrix_double(latency_us);
  std::printf(",\"bidirectional_remote_read_gbps\":%.3f", bidir_remote_read_gbps);
  std::printf(",\"bidirectional_remote_write_gbps\":%.3f", bidir_remote_write_gbps);
  std::printf(",\"bidirectional_remote_copy_gbps\":%.3f", bidir_remote_copy_gbps);
  std::printf("}\n");

  for (int n = 0; n < nodes; n++) {
    numa_free(buffers[n], bytes);
    numa_free(local_dst[n], bytes);
  }
  return 0;
}
