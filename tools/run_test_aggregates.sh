#!/usr/bin/env bash
# Build and run the per-area aggregate test binaries.
#
# The old `tests` task ran one `mojo -I .` per test file: 87 invocations,
# each elaborating the whole `numax` source graph before running a single
# assertion. The cost is compilation, not the test bodies, so aggregating
# an area into one binary pays that elaboration once per area instead of
# once per file -- and the build phase, being 11 independent compiler
# invocations, then parallelises.
#
# Aggregates are generated; see tools/gen_test_aggregates.py. `--check`
# below is the drift gate, and it is not optional: an aggregate that is out
# of date silently stops running whatever was added, and a `TestSuite` with
# no registered tests still exits 0, so nothing downstream would notice.

set -uo pipefail

AGG_DIR="tests/_agg"
BUILD_DIR="${BUILD_DIR:-build/agg}"

# Default to the core count. The build phase dominates and the aggregates
# are independent, so this is where the wall clock goes. Capped at 4
# because each job is a full elaboration of the source graph and the macOS
# CI runner has 7GB across 3 cores -- the cap is what keeps a parallel
# build from becoming an OOM. Override with AGG_JOBS.
_ncpu="$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null) || echo 2 )"
[ "$_ncpu" -gt 4 ] 2>/dev/null && _ncpu=4
JOBS="${AGG_JOBS:-$_ncpu}"

# Tests that cannot share a process with their neighbours. Empty by
# design -- see the note on EXCLUDE in tools/gen_test_aggregates.py, which
# records why the two candidates (the `/tmp` paths in tests/io, the global
# RNG seed) turned out to be safe. Keep this list matching that one.
STANDALONE=()

echo "── checking aggregates are up to date ──"
python3 tools/gen_test_aggregates.py --check || {
  echo "ERROR: aggregates are stale; run \`pixi run tests-gen\`" >&2
  exit 1
}

mkdir -p "$BUILD_DIR"
# macOS ships bash 3.2, which has no `mapfile`.
AGGS=()
for a in "$AGG_DIR"/agg_*.mojo; do [ -f "$a" ] && AGGS+=("$a"); done
if [ "${#AGGS[@]}" -eq 0 ]; then
  echo "ERROR: no aggregates found in $AGG_DIR" >&2
  exit 1
fi

echo "── building ${#AGGS[@]} aggregates (jobs=$JOBS) ──"
build_one() {
  local src="$1" out="$BUILD_DIR/$(basename "${1%.mojo}")"
  # Each aggregate imports bare module names, resolved against its own
  # area directory.
  local area="${src##*/agg_}"; area="${area%.mojo}"
  local inc="tests/$area"
  [ "$area" = "_root" ] && inc="tests"
  # Line tables so a crash names the function it died in rather than eight
  # raw addresses -- the reason the per-file CI loop built before running.
  if ! mojo build -debug-level=line-tables -I . -I "$inc" "$src" -o "$out" 2>"$out.log"; then
    echo "BUILD FAILED: $src"; sed -n '1,25p' "$out.log"; return 1
  fi
}
export -f build_one; export BUILD_DIR
build_failed=()
if [ "$JOBS" -gt 1 ]; then
  printf '%s\n' "${AGGS[@]}" | xargs -P "$JOBS" -n1 -I FF bash -c 'build_one FF'
  # xargs hides which item failed, so re-check the artifacts.
  for a in "${AGGS[@]}"; do
    [ -x "$BUILD_DIR/$(basename "${a%.mojo}")" ] || build_failed+=("$a")
  done
else
  # Keep going after a failure: stopping at the first one turns a run into
  # a one-bug-per-iteration loop.
  for a in "${AGGS[@]}"; do build_one "$a" || build_failed+=("$a"); done
fi
if [ "${#build_failed[@]}" -ne 0 ]; then
  echo "── ${#build_failed[@]} aggregate(s) failed to build ──"
  printf '  %s\n' "${build_failed[@]}"
  exit 1
fi

# Run everything and collect failures rather than stopping at the first:
# with one binary per area, the areas after a failure still carry signal.
#
# Three attempts per area, because the Mojo runtime segfaults transiently
# inside `libKGENCompilerRTShared` on a fresh process before any user code
# runs. The retry is per *area*, not per run: retrying the whole set would
# re-roll every start to recover from one flake. Aggregation already cut
# the exposure from 89 process starts to 11; this keeps the guarantee that
# a genuinely broken area still fails three times and fails the job.
ATTEMPTS="${AGG_ATTEMPTS:-3}"
failed=()
echo "── running ${#AGGS[@]} aggregates (up to $ATTEMPTS attempts each) ──"
for a in "${AGGS[@]}"; do
  bin="$BUILD_DIR/$(basename "${a%.mojo}")"
  ok=0
  for attempt in $(seq 1 "$ATTEMPTS"); do
    if "$bin"; then ok=1; break; fi
    [ "$attempt" -lt "$ATTEMPTS" ] && echo "── $(basename "$a") failed, attempt $attempt of $ATTEMPTS ──"
  done
  [ "$ok" -eq 1 ] || failed+=("$a")
done

if [ "${#STANDALONE[@]}" -ne 0 ]; then
  echo "── running ${#STANDALONE[@]} standalone tests ──"
  for t in "${STANDALONE[@]}"; do
    [ -f "$t" ] || { echo "MISSING: $t"; failed+=("$t"); continue; }
    if ! mojo -I . "$t"; then failed+=("$t"); fi
  done
fi

if [ "${#failed[@]}" -ne 0 ]; then
  echo
  echo "── ${#failed[@]} area(s) failed ──"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "── all ${#AGGS[@]} areas passed ──"
