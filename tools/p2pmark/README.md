# llm_p2pmark

Small CUDA/NCCL fabric diagnostic used by `llm_decode_bench.py`.

It is intentionally narrower than upstream `p2pmark`: it emits stable JSON that
the Python benchmark can embed in its startup report. It measures:

- CUDA peer-access availability matrix
- GPU-to-GPU `cudaMemcpyPeerAsync` bandwidth matrix
- staggered peer-distance writes, single-writer fan-out, ring bandwidth, and
  all-to-all fabric stress
- dependent 128-byte-stride remote-read latency in isolated and full-load modes
- NCCL allreduce latency and bandwidth for a configurable size list
- custom PCIe allreduce vs NCCL latency/bandwidth for the same size list

Build:

```bash
make -C tools/p2pmark
```

Run standalone:

```bash
tools/p2pmark/llm_p2pmark --mode all --size-mb 64 --iters 20
```

The default allreduce mode compares custom PCIe allreduce against NCCL over a
small-message sweep from 256 B to 1 MiB and reports the winner. Use
`--allreduce-sizes-mb 1,2,4,8,16,32,64` for a larger MiB-only sweep.

The benchmark wrapper can run it before inference:

```bash
python3 llm_decode_bench.py --p2pmark --p2pmark-mode all --port 8000
python3 llm_decode_bench.py --p2pmark-only
```

`llm_decode_bench.py` also contains a compressed Linux x86_64 CUDA/NCCL fallback
binary. A raw single-file download can therefore run `--p2pmark-only` without
the sidecar file, as long as compatible `libcudart.so.13` and `libnccl.so.2`
are available. If the runtime is incompatible, rebuild the sidecar with `make`
and pass `--p2pmark-bin`.

Do not run this while a large vLLM/SGLang instance already occupies most GPU
memory. The diagnostic allocates buffers on every visible GPU and can perturb
fabric traffic.
