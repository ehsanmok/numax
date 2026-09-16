#!/usr/bin/env bash
# Build the examples concurrently, then run them.
#
#   tools/run_examples.sh              # CPU examples: build in parallel, run each
#   tools/run_examples.sh --gpu-build  # GPU examples: type-check only, in parallel
#
# Examples cannot be aggregated the way tests are -- each one is its own
# `main` and its printed output is the point -- but they can share the
# *build* phase. `examples-cpu` was 19 sequential `mojo -I .` invocations
# and the cost is compilation, the same as it was for the test suites.
# Building concurrently and then running the binaries keeps one process per
# example while cutting the wall clock by roughly the job count.

set -uo pipefail

BUILD_DIR="${BUILD_DIR:-build/examples}"

# See the cap rationale in tools/run_test_aggregates.sh: each job is a full
# elaboration of the source graph and the macOS runner is memory-bound.
_ncpu="$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null) || echo 2 )"
[ "$_ncpu" -gt 4 ] 2>/dev/null && _ncpu=4
JOBS="${EX_JOBS:-$_ncpu}"

# The examples that launch GPU kernels alongside their CPU paths. They are
# compiled against a named accelerator and never run here, because CI has no
# device. Keep this list in step with `examples-gpu-build` in pixi.toml.
GPU=(
  examples/advanced/gaussian_gpu.mojo
  examples/advanced/unified_tensor_gpu.mojo
  examples/advanced/ode.mojo
  examples/intermediate/softmax.mojo
  examples/intermediate/random_ensemble.mojo
  examples/advanced/quantum_well.mojo
  examples/advanced/batched_solve.mojo
)

is_gpu() {
  local f="$1"
  for g in "${GPU[@]}"; do [ "$f" = "$g" ] && return 0; done
  return 1
}

mkdir -p "$BUILD_DIR"

if [ "${1:-}" = "--gpu-build" ]; then
  # `--target-accelerator sm_80` is not optional and not arbitrary; the long
  # explanation lives with `examples-gpu-build` in pixi.toml.
  echo "── type-checking ${#GPU[@]} GPU examples (jobs=$JOBS) ──"
  gpu_build_one() {
    local src="$1" out="$BUILD_DIR/gpu_$(basename "${1%.mojo}")"
    if ! mojo build --target-accelerator sm_80 -I . "$src" -o "$out" 2>"$out.log"; then
      echo "BUILD FAILED: $src"; sed -n '1,25p' "$out.log"; return 1
    fi
  }
  export -f gpu_build_one; export BUILD_DIR
  printf '%s\n' "${GPU[@]}" | xargs -P "$JOBS" -n1 -I FF bash -c 'gpu_build_one FF'
  bad=()
  for g in "${GPU[@]}"; do
    [ -x "$BUILD_DIR/gpu_$(basename "${g%.mojo}")" ] || bad+=("$g")
  done
  rm -f "$BUILD_DIR"/gpu_*
  if [ "${#bad[@]}" -ne 0 ]; then
    echo "── ${#bad[@]} GPU example(s) failed to compile ──"
    printf '  %s\n' "${bad[@]}"
    exit 1
  fi
  echo "── all ${#GPU[@]} GPU examples type-check ──"
  exit 0
fi

# `git ls-files` emits sorted paths already. Do NOT pipe through `sort`:
# under `pixi run`, LD_LIBRARY_PATH points at the env's own libs and a
# system binary can fail to load, which inside `$(...)` is silent -- the
# list comes back empty and every example is "run" with a green result.
# The count check below is the backstop for that whole class of bug.
CPU=()
while IFS= read -r e; do
  [ -n "$e" ] || continue
  is_gpu "$e" || CPU+=("$e")
done < <(git ls-files 'examples/*/*.mojo')

if [ "${#CPU[@]}" -lt 15 ]; then
  echo "ERROR: found only ${#CPU[@]} CPU examples; expected 19." >&2
  echo "       Refusing to report a pass over a truncated list." >&2
  exit 1
fi

echo "── building ${#CPU[@]} CPU examples (jobs=$JOBS) ──"
build_one() {
  local src="$1" out="$BUILD_DIR/ex_${1//\//_}"; out="${out%.mojo}"
  if ! mojo build -I . "$src" -o "$out" 2>"$out.log"; then
    echo "BUILD FAILED: $src"; sed -n '1,25p' "$out.log"; return 1
  fi
}
export -f build_one; export BUILD_DIR
printf '%s\n' "${CPU[@]}" | xargs -P "$JOBS" -n1 -I FF bash -c 'build_one FF'

failed=()
echo "── running ${#CPU[@]} CPU examples ──"
for e in "${CPU[@]}"; do
  bin="$BUILD_DIR/ex_${e//\//_}"; bin="${bin%.mojo}"
  if [ ! -x "$bin" ]; then failed+=("$e (build)"); continue; fi
  echo "── $e ──"
  if ! "$bin" >/dev/null; then failed+=("$e"); fi
done

if [ "${#failed[@]}" -ne 0 ]; then
  echo
  echo "── ${#failed[@]} example(s) failed ──"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "── all ${#CPU[@]} CPU examples passed ──"
