//! Rust `thermite` 0.2 CPU baseline for the same gaussian(x) = exp(-x^2)
//! sweep the numax and NumPy/MLX/PyTorch benchmarks in ../../ run, at the
//! same sizes -- see ../../README.md.
//!
//! `thermite` is the library `numax`'s composable-numeric-type pattern was
//! originally ported from (see the top-level README's introduction), so
//! this isn't picking an arbitrary Rust SIMD crate -- it's the closest thing
//! to a direct ancestor comparison available. `thermite`'s NEON backend is
//! "Complete. Mandatory on the architecture" for aarch64 (this machine's
//! architecture), so this is a genuinely vectorized run, not a scalar
//! fallback standing in for one.
//!
//! `gaussian` runs in place (`Self = &mut [f32]`), the same shape as the
//! `sigmoid` example in `thermite`'s own crate docs -- there's no
//! two-slice (input-to-output) SIMD iterator in the public API, only
//! single-slice in-place transforms, so an in-place run is the idiomatic
//! way to use this crate, not a benchmark-specific shortcut.
//!
//! Run: `pixi run bench-thermite` (from the repo root)

use std::time::Instant;

use thermite::isa::InstructionSet;
use thermite::math::TranscendentalMath;
use thermite::prelude::*;

const SIZES: [usize; 6] = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26];
const WARMUP_ITERS: usize = 3;
const TIMED_ITERS: usize = 10;

// The kernel: written once against trait bounds, no ISA or lane count
// named. `#[thermite::dispatch]` is what makes this compile with target
// features and get inlined into the dispatched-to backend body -- without
// it, this would still be correct, just "catastrophically slow" per
// thermite's own docs.
#[thermite::dispatch(V)]
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V {
    (-(x * x)).exp()
}

fn run_gaussian_inplace(data: &mut [f32]) {
    thermite::dispatch_dyn!(for<S> |data: &mut [f32]| {
        let (head, chunks, tail) = data.try_aligned_simd_iter_mut::<f32xN>();

        for v in chunks {
            *v = gaussian(*v);
        }

        // Ragged ends run through the same `gaussian` on the 1-lane scalar
        // backend -- the same pattern thermite's own docs use.
        for x in head.iter_mut().chain(tail) {
            *x = gaussian(Vector::<f32>::splat(*x)).extract::<0>();
        }
    });
}

fn bench_at_size(n: usize) {
    let original: Vec<f32> = (0..n).map(|i| i as f32 * 0.0001 - 50.0).collect();
    let mut data = original.clone();

    for _ in 0..WARMUP_ITERS {
        run_gaussian_inplace(&mut data);
    }

    let t0 = Instant::now();
    for _ in 0..TIMED_ITERS {
        run_gaussian_inplace(&mut data);
    }
    let elapsed = t0.elapsed();

    let avg_ms = elapsed.as_secs_f64() * 1000.0 / TIMED_ITERS as f64;
    let elems_per_sec_m = (n as f64) / (avg_ms / 1000.0) / 1e6;
    println!("n={n:>9}  CPU={avg_ms:8.4} ms  ({elems_per_sec_m:9.1} M elem/s)");

    // Separate, untimed correctness check against an f64 reference -- not
    // the mutated `data` from the timed loop above, which has had
    // `gaussian` applied to it `TIMED_ITERS` times over, not once.
    let mut check = original.clone();
    run_gaussian_inplace(&mut check);
    let max_err = original
        .iter()
        .zip(check.iter())
        .map(|(&x, &y)| {
            let x = x as f64;
            let reference = (-(x * x)).exp();
            (y as f64 - reference).abs()
        })
        .fold(0.0_f64, f64::max);
    if max_err > 1e-5 {
        println!("  WARNING: max |thermite - f64 reference| = {max_err}");
    }
}

fn main() {
    println!(
        "thermite 0.2  dtype=f32  ISA={:?}",
        InstructionSet::get()
    );
    for &n in SIZES.iter() {
        bench_at_size(n);
    }
}
