# llm_amd_fabric

Small NUMA CPU-fabric diagnostic used by `llm_decode_bench.py`.

It measures practical socket-to-socket behavior on AMD EPYC systems:

- NUMA node distance matrix from libnuma
- CPU-read bandwidth for every CPU-node -> memory-node pair
- CPU-write bandwidth for every CPU-node -> memory-node pair
- memcpy bandwidth for every CPU-node -> source-memory-node pair
- dependent pointer-chase latency for every CPU-node -> memory-node pair
- dual-socket bidirectional remote read/write/memcpy saturation when at least two NUMA nodes exist

Build:

```bash
make -C tools/amd_fabric
```

Run standalone:

```bash
tools/amd_fabric/llm_amd_fabric --size-mb 512 --threads 0
```

`--threads 0` auto-selects up to 64 CPUs per NUMA node, which is usually needed
to approach cross-socket fabric saturation on high-core-count EPYC systems.
Small values such as 2 or 4 are only smoke tests and should not be interpreted
as maximum xGMI bandwidth.

The helper reports JSON only. The Python wrapper renders the human-readable
tables and keeps the raw JSON in benchmark output.

Linux does not expose a portable "active xGMI link count" sysfs value for EPYC
socket-to-socket links. The wrapper therefore reports board/CPU expectation
separately from measured local/remote NUMA bandwidth. When Linux perf exposes
`data_fabric` `link_N` counter slots, the wrapper prints the slot count as a
hint, but this is not a decoded active/trained xGMI link count. Treat measured
remote bandwidth as the authoritative fabric result.
