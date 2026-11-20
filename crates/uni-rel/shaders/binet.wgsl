// rel/binet.wgsl: Schwarzschild null geodesics in Binet/Cartesian form. GPU twin of
// `uni_rel::binet::{trace, static_observer_ray}` (crates/uni-rel/src/binet.rs): the same
// RK4, step rule and termination tests, in f32.
//
// Units: G = c = M = 1, hole at the origin, horizon at r = 2 (Schwarzschild 1916). The ray
// moves in flat Cartesian 3-space under
//     d²x/dλ² = −3·h²·x / r⁵,   h = |x × dx/dλ| (conserved),
// the Binet orbit equation u'' + u = 3M·u² (Chandrasekhar 1983, ch. 3) in Cartesian form. λ is
// the affine parameter; v = dx/dλ is never renormalised. Reference integrator and step rule:
// `deflect()` in docs/research/m0-golden-values.py.
//
// The selftest (Task 11) imports this file and compares `binet_trace` with the CPU `trace`.

const BINET_CAPTURED: u32 = 0u;
const BINET_ESCAPED: u32 = 1u;
const BINET_MAX_STEPS: u32 = 2u;

// Capture margin ε: captured once r < 2(1 + ε) (as in m0-golden-values.py).
const BINET_CAPTURE_RADIUS: f32 = 2.000002;

// How a ray ended. Function-local only; never stored in a uniform or storage buffer.
struct BinetEnd {
    // BINET_CAPTURED, BINET_ESCAPED or BINET_MAX_STEPS.
    status: u32,
    // Final position [M].
    x: vec3<f32>,
    // Final tangent dx/dλ; normalize(v) is the exit direction when escaped.
    v: vec3<f32>,
}

// Affine step dλ = clamp(0.02·r, 1e-3, 5.0) [M] (physics research §1.3: step ∝ r).
fn binet_step_size(r: f32) -> f32 {
    return clamp(0.02 * r, 1e-3, 5.0);
}

// a = −3·h²·x/r⁵ with h from the current state; (h²/r²)/r³ avoids f32 overflow at large r.
fn binet_accel(x: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let r2 = dot(x, x);
    let r = sqrt(r2);
    let hv = cross(x, v);
    let h2 = dot(hv, hv);
    return (-3.0 * (h2 / r2) / (r2 * r)) * x;
}

// Integrates from x0 with tangent v0. Before each step: captured if r < 2(1 + ε); escaped if
// r > r_far moving outwards (x·v > 0). After max_steps steps without either: BINET_MAX_STEPS.
fn binet_trace(x0: vec3<f32>, v0: vec3<f32>, r_far: f32, max_steps: u32) -> BinetEnd {
    var x = x0;
    var v = v0;
    var i = 0u;
    loop {
        let r = length(x);
        if (r < BINET_CAPTURE_RADIUS) {
            return BinetEnd(BINET_CAPTURED, x, v);
        }
        if (r > r_far && dot(x, v) > 0.0) {
            return BinetEnd(BINET_ESCAPED, x, v);
        }
        if (i >= max_steps) {
            break;
        }
        // Classical 4th-order Runge–Kutta.
        let dl = binet_step_size(r);
        let k1x = v;
        let k1v = binet_accel(x, v);
        let k2x = v + k1v * (0.5 * dl);
        let k2v = binet_accel(x + k1x * (0.5 * dl), k2x);
        let k3x = v + k2v * (0.5 * dl);
        let k3v = binet_accel(x + k2x * (0.5 * dl), k3x);
        let k4x = v + k3v * dl;
        let k4v = binet_accel(x + k3x * dl, k4x);
        x += (k1x + 2.0 * k2x + 2.0 * k3x + k4x) * (dl / 6.0);
        v += (k1v + 2.0 * k2v + 2.0 * k3v + k4v) * (dl / 6.0);
        i += 1u;
    }
    return BinetEnd(BINET_MAX_STEPS, x, v);
}

// Initial tangent (E = 1) for a ray leaving a static observer at `cam` in the local direction
// `local_dir`: the radial part is kept and the tangential part divided by √(1 − 2/r), giving
// b = r·sin δ/√(1 − 2/r) (Synge 1966, MNRAS 131, 463). The caller guarantees r > 2(1 + ε);
// the lapse is floored so a violation cannot divide by zero.
fn static_observer_ray(cam: vec3<f32>, local_dir: vec3<f32>) -> vec3<f32> {
    let r = length(cam);
    let n = cam / r;
    let d = normalize(local_dir);
    let radial = n * dot(d, n);
    let lapse = sqrt(max(1.0 - 2.0 / r, 1e-12));
    return radial + (d - radial) / lapse;
}
