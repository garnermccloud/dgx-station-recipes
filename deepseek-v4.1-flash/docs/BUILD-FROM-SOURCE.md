# Building the image is not part of this release

The release runs the stock `vllm/vllm-openai:v0.31.0` image.  `ds41 install` pulls it; there is no
`Dockerfile`, `build.sh`, or image build step.  At launch the recipe mounts the overlays,
`sitecustomize.py`, `fastpath/`, the hook and rowmaps/tuning files, and b12x from this checkout
read-only.

The prebuilt kernel `.so` files under `fastpath/` are the ones that ship.  Rebuilding them is a
development task for the kernel owner; it is not required to run the release.  The build runs inside
the same official image, needs no GPU, and is not byte-reproducible (the compiler embeds source and
temporary paths), so a rebuild is verified by behaviour, never by hash:

```bash
cd kernels
./build_all.sh              # writes ../fastpath/*.so; IMAGE=... to build against another image
```

See `../kernels/BUILD-NOTES.md` for the per-kernel mapping and `../fastpath/THIRD_PARTY.md` and
`../NOTICE` for licences.

If you need to change the recipe, edit the mounted trees and restart; do not build a serving image.
