# The one compatibility assertion.  Called once by preflight / serve start, never per component.
# The image is the official v0.31.0 release; its vendored DeepGEMM files must be the exact bases
# the MegaMoE fork was written for.
DS41_SUPPORTED_IMAGE='vllm/vllm-openai:v0.31.0'
DS41_PKG='/usr/local/lib/python3.12/dist-packages'
DS41_SITE_TARGET='/usr/lib/python3.12/sitecustomize.py'
DS41_DEEPGEMM_BASES=(
  '2e725484419819a1e87bfe76ed5c90799b980701fb8f9f2816910a4d2082ca9b /usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh'
  'f001c934026af52aac90bf162707b6f7454813f794a279aa15dd5a3d6efa490d /usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh'
  '43b0f07bed73a6ed0a7c53b42308d709ccd42b824ad8eb2738875207c097bb68 /usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh'
)

ds41_assert_compat() {
  [ "${DS41_IMAGE:-}" = "$DS41_SUPPORTED_IMAGE" ] || {
    echo "unsupported image ${DS41_IMAGE:-unset}; this release is built for $DS41_SUPPORTED_IMAGE" >&2
    return 1
  }
  docker image inspect "$DS41_IMAGE" >/dev/null 2>&1 || {
    echo "image $DS41_IMAGE is not present; run: ds41 install" >&2
    return 1
  }
  local got expected
  got=$(docker run --rm --network none --entrypoint sha256sum "$DS41_IMAGE" \
    /usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh \
    /usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh \
    /usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh | awk '{print $1" "$2}') || {
    echo "could not hash the image's vendored DeepGEMM files" >&2
    return 1
  }
  expected=$(printf '%s\n' "${DS41_DEEPGEMM_BASES[@]}")
  [ "$got" = "$expected" ] || {
    echo "image $DS41_IMAGE vendored DeepGEMM files do not match the fork's base" >&2
    return 1
  }
}

ds41_image_layout_ok() {
  docker run --rm --network none --entrypoint sh "$DS41_IMAGE" -c '
    for p in /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41 \
             /usr/local/lib/python3.12/dist-packages/vllm/config/cache.py \
             /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic \
             /usr/local/lib/python3.12/dist-packages/vllm/renderers \
             /usr/local/lib/python3.12/dist-packages/vllm/tokenizers \
             /usr/lib/python3.12/sitecustomize.py; do
      [ -e "$p" ] || exit 1
    done' >/dev/null 2>&1
}

ds41_image_numactl_ok() {
  docker run --rm --network none --entrypoint sh "$DS41_IMAGE" \
    -c 'command -v numactl >/dev/null 2>&1' >/dev/null 2>&1
}
