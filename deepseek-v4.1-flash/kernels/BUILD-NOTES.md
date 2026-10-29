# Build notes

On 2026-10-23 all six build scripts were run in the pinned image on a CPU-only host (no GPU):

```text
w4-boundary   ok   libw4boundary.so
w6-moe        ok   libw6mk.so, libw6tp.so, libw6combine.so
w8-persist    ok   libw8oproj.so, libw8topkg.so
w1-attn       ok   w1_attn_ext.so
w3-preattn    ok   w3_ext.so
peer3         ok   peer3_gb300_sm103.so, peer3_rtx_sm120.so
```

The shipped binaries are the production ones; their hashes are in `../fastpath/SHA256SUMS`.  Rebuilds are
functionally equivalent but not byte-identical: `nvcc` embeds the source paths and the JIT builds embed
their temporary build directories.  Compare behaviour with the peer3 tests and the end-to-end `verify/`
run, not by hash.

`libw8topkg.so` is built with `-DW8TG_SORT=0`, matching the production binary.  The determinism variant
(`W8TG_SORT=1`) is not shipped.  The peer3 build produces both `sm_103` (GB300) and `sm_120` (RTX PRO
6000) device halves; no GPU is needed to compile either.
