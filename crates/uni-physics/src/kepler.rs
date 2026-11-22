//! Two-body conics (M1 Task A1): elliptic and hyperbolic Kepler solvers, universal-variable
//! propagation and conversion between elements and state vectors.
//!
//! Sources: Kepler's equation and the anomaly ↔ state formulas (Meeus, *Astronomical
//! Algorithms* 2nd ed., ch. 30 and 33; Vallado, *Fundamentals of Astrodynamics and
//! Applications* 4th ed., 2013, §2.2–2.3); Barker's equation for the parabola (Vallado §2.2,
//! solved in closed form); universal variables with the Stumpff functions C(z), S(z) (Bate,
//! Mueller & White, *Fundamentals of Astrodynamics*, 1971, ch. 4; Vallado Algorithm 8);
//! Laguerre's method for Kepler's equation (Conway 1986, Celest. Mech. 39, 199).
//!
//! Numerics. Near-parabolic orbits lose every digit to cancellation in the textbook forms, so
//! the code never forms them: E − sin E and sinh H − H come from their Taylor series below 1,
//! 1 − cos E = 2 sin²(E/2), 1 − e cos E = (1 − e) + 2e sin²(E/2) (and the hyperbolic twins),
//! and 1 − e is exact for e ≥ 0.5 (Sterbenz). The Kepler solvers run Newton from a proven
//! upper bound of the root, where the residual is convex, so the iterates fall monotonically.
//!
//! Domain. Every function returns finite values for finite input whose true answer is
//! representable; NaN in gives NaN out. A conic with q ≤ 0 or μ ≤ 0 (a radial or invalid
//! orbit) has no periapsis: [`Conic::state`] returns the centre at rest.

use std::f64::consts::{PI, TAU};

use glam::{DMat3, DVec3};
use uni_core::Epoch;

/// Iteration cap for the solvers, a safety net: each starts from a bound of its root.
const MAX_ITER: usize = 100;

/// A two-body orbit. Angles are in the conic's reference frame (whatever frame the state
/// vectors are in); any e ≥ 0.
///
/// Conventions (used by [`Conic::from_state`]): i ∈ [0, π]; node and argp in (−π, π]. An
/// equatorial orbit (angular momentum along ±z) has node = 0, so argp is measured from +x. A
/// circular orbit (e = 0) has argp = 0, so periapsis is at the ascending node (at +x if it is
/// also equatorial) and `tp` is the time of that passage. Nearly circular orbits keep their
/// computed tiny e and the periapsis it implies; their states are reproduced either way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conic {
    /// Periapsis distance [m].
    pub q_m: f64,
    /// Eccentricity.
    pub e: f64,
    /// Inclination [rad].
    pub i_rad: f64,
    /// Longitude of the ascending node [rad].
    pub node_rad: f64,
    /// Argument of periapsis [rad].
    pub argp_rad: f64,
    /// Time of periapsis passage.
    pub tp: Epoch,
    /// Gravitational parameter of the central body [m³/s²].
    pub mu_m3s2: f64,
}

impl Conic {
    /// From classical elements with mean anomaly `m0` [rad] at `epoch` (e < 1); `a_m` [m],
    /// angles [rad], `mu` [m³/s²].
    ///
    /// `tp` is the periapsis passage nearest `epoch` (m0 is reduced to [−π, π]). A hyperbola
    /// may be given with a < 0 and e > 1 (M = n(t − tp), no reduction). A parabola cannot be
    /// given through a: set `q_m` directly. Other combinations give q ≤ 0, which has no state.
    #[allow(clippy::too_many_arguments)]
    pub fn from_mean_anomaly(
        a_m: f64,
        e: f64,
        i: f64,
        node: f64,
        argp: f64,
        m0: f64,
        epoch: Epoch,
        mu: f64,
    ) -> Self {
        let size = a_m.abs();
        let n = (mu / size).sqrt() / size; // mean motion √(μ/|a|³)
        let m = if e < 1.0 {
            m0 - TAU * (m0 / TAU).round()
        } else {
            m0
        };
        let since_periapsis = if n > 0.0 { m / n } else { 0.0 };
        Conic {
            q_m: a_m * (1.0 - e),
            e,
            i_rad: i,
            node_rad: node,
            argp_rad: argp,
            tp: epoch.add_seconds(-since_periapsis),
            mu_m3s2: mu,
        }
    }

    /// The osculating conic of state `r` [m], `v` [m/s] at `t` (review I9: used by C5, A4,
    /// A1's Pluto and C1).
    ///
    /// Angle conventions are on [`Conic`]. A radial state (r × v = 0), r = 0 or μ ≤ 0 has no
    /// periapsis and gives q = 0, e = 1 with `tp = t`.
    pub fn from_state(r: DVec3, v: DVec3, mu: f64, t: Epoch) -> Self {
        let rn = r.length();
        let h = r.cross(v);
        let hn = h.length();
        if rn == 0.0 || hn == 0.0 || mu <= 0.0 {
            return Conic {
                q_m: 0.0,
                e: 1.0,
                i_rad: 0.0,
                node_rad: 0.0,
                argp_rad: 0.0,
                tp: t,
                mu_m3s2: mu,
            };
        }
        // Eccentricity vector (Laplace–Runge–Lenz / μ), pointing at periapsis.
        let e_vec = ((v.length_squared() - mu / rn) * r - r.dot(v) * v) / mu;
        let e = e_vec.length();
        // q = p/(1 + e) with p = h²/μ: no cancellation for any e.
        let q = hn * hn / mu / (1.0 + e);
        let w = h / hn;
        let i = w.x.hypot(w.y).atan2(w.z);
        let node = if w.x == 0.0 && w.y == 0.0 {
            0.0
        } else {
            w.x.atan2(-w.y)
        };
        // In-plane axes: toward the ascending node, and 90° ahead of it.
        let n_hat = DVec3::new(node.cos(), node.sin(), 0.0);
        let m_hat = w.cross(n_hat);
        let argp = if e > 0.0 {
            e_vec.dot(m_hat).atan2(e_vec.dot(n_hat))
        } else {
            0.0
        };
        let u = r.dot(m_hat).atan2(r.dot(n_hat)); // argument of latitude
        let nu = (u - argp + PI).rem_euclid(TAU) - PI; // true anomaly
        let half = 0.5 * nu;
        let since_periapsis = if e < 1.0 {
            // tan(E/2) = √((1 − e)/(1 + e)) tan(ν/2), quadrant kept by atan2.
            let ea = 2.0 * ((1.0 - e).sqrt() * half.sin()).atan2((1.0 + e).sqrt() * half.cos());
            let m = x_minus_sin(ea) + (1.0 - e) * ea.sin(); // E − e sin E
            let a = q / (1.0 - e);
            m / ((mu / a).sqrt() / a)
        } else if e > 1.0 {
            // tanh(H/2) = √((e − 1)/(e + 1)) tan(ν/2); |·| < 1 inside the asymptotes.
            let limit = 1.0 - f64::EPSILON;
            let ratio = (((e - 1.0) / (e + 1.0)).sqrt() * half.tan()).clamp(-limit, limit);
            let hh = 2.0 * ratio.atanh();
            let m = (e - 1.0) * hh.sinh() + sinh_minus_x(hh); // e sinh H − H
            let a = q / (e - 1.0);
            m / ((mu / a).sqrt() / a)
        } else {
            // Barker: D + D³/3 = √(μ/(2q³)) (t − tp), D = tan(ν/2).
            let d = half.tan();
            (d + d * d * d / 3.0) / ((mu / (2.0 * q)).sqrt() / q)
        };
        Conic {
            q_m: q,
            e,
            i_rad: i,
            node_rad: node,
            argp_rad: argp,
            tp: t.add_seconds(-since_periapsis),
            mu_m3s2: mu,
        }
    }

    /// Position [m] and velocity [m/s] at `t`, in the reference frame; any e ≥ 0.
    ///
    /// One Kepler solve: elliptic (e < 1), hyperbolic (e > 1) or Barker's cubic in closed
    /// form (e = 1). The velocity is the analytic time derivative of the conic.
    pub fn state(&self, t: Epoch) -> (DVec3, DVec3) {
        let (q, e, mu) = (self.q_m, self.e, self.mu_m3s2);
        if e.is_nan() {
            return (DVec3::NAN, DVec3::NAN);
        }
        if q <= 0.0 || mu <= 0.0 {
            return (DVec3::ZERO, DVec3::ZERO);
        }
        let e = e.max(0.0);
        let dt = t.seconds_since(self.tp);
        let (pos, vel) = if e < 1.0 {
            let a = q / (1.0 - e);
            let n = (mu / a).sqrt() / a;
            let period = TAU / n;
            let dt = dt - period * (dt / period).round();
            elliptic_perifocal(a, e, n * dt, n)
        } else if e > 1.0 {
            hyperbolic_perifocal(q, e, mu, dt)
        } else {
            parabolic_perifocal(q, mu, dt)
        };
        let rot = perifocal_to_reference(self.i_rad, self.node_rad, self.argp_rad);
        (rot * pos, rot * vel)
    }

    /// Orbital period [s]; `None` for e ≥ 1.
    pub fn period_s(&self) -> Option<f64> {
        if self.e < 1.0 && self.q_m > 0.0 && self.mu_m3s2 > 0.0 {
            let a = self.q_m / (1.0 - self.e.max(0.0));
            Some(TAU * a * (a / self.mu_m3s2).sqrt())
        } else {
            None
        }
    }
}

/// Perifocal position [m] and velocity [m/s] on an ellipse of semi-major axis `a` [m] and
/// eccentricity `e` < 1 at mean anomaly `m` [rad], with mean motion `n` [rad/s] for the
/// velocity (the Standish planets pass their tables' dM/dt here).
pub(crate) fn elliptic_perifocal(a: f64, e: f64, m: f64, n: f64) -> (DVec3, DVec3) {
    let ea = solve_kepler(m, e);
    let (sin_e, cos_e) = ea.sin_cos();
    let half = (0.5 * ea).sin();
    let one_minus_cos = 2.0 * half * half;
    let q = a * (1.0 - e);
    let b = (a * q * (1.0 + e)).sqrt(); // a√(1 − e²)
    let e_dot = n / ((1.0 - e) + e * one_minus_cos); // n/(1 − e cos E)
    (
        DVec3::new(q - a * one_minus_cos, b * sin_e, 0.0), // a(cos E − e), b sin E
        DVec3::new(-a * sin_e * e_dot, b * cos_e * e_dot, 0.0),
    )
}

/// Perifocal state on a hyperbola of periapsis `q` and e > 1, `dt` seconds after periapsis.
fn hyperbolic_perifocal(q: f64, e: f64, mu: f64, dt: f64) -> (DVec3, DVec3) {
    let a = q / (e - 1.0); // |a|
    let n = (mu / a).sqrt() / a;
    let h = solve_kepler_hyperbolic(n * dt, e);
    let half = (0.5 * h).sinh();
    let cosh_minus_1 = 2.0 * half * half;
    let b = (a * q * (e + 1.0)).sqrt(); // |a|√(e² − 1)
    let h_dot = n / ((e - 1.0) + e * cosh_minus_1); // n/(e cosh H − 1)
    (
        DVec3::new(q - a * cosh_minus_1, b * h.sinh(), 0.0), // |a|(e − cosh H), b sinh H
        DVec3::new(-a * h.sinh() * h_dot, b * h.cosh() * h_dot, 0.0),
    )
}

/// Perifocal state on a parabola of periapsis `q`, `dt` seconds after periapsis. Barker's
/// equation D + D³/3 = k·dt (D = tan(ν/2), k = √(μ/(2q³))) has the closed-form root
/// D = 2 sinh(asinh(1.5 k dt)/3), since D + D³/3 = (2/3) sinh 3φ for D = 2 sinh φ.
fn parabolic_perifocal(q: f64, mu: f64, dt: f64) -> (DVec3, DVec3) {
    let k = (mu / (2.0 * q)).sqrt() / q;
    let d = 2.0 * ((1.5 * k * dt).asinh() / 3.0).sinh();
    let d_dot = k / (1.0 + d * d);
    (
        DVec3::new(q * (1.0 - d * d), 2.0 * q * d, 0.0),
        DVec3::new(-2.0 * q * d * d_dot, 2.0 * q * d_dot, 0.0),
    )
}

/// Eccentric anomaly E [rad] for mean anomaly `m_rad`, e < 1 (E − e sin E = M).
///
/// Any finite M (reduced exactly to [−π, π], the turns added back); e is clamped to [0, 1]
/// (e = 1 solves the parabolic limit E − sin E = M). Converged to rounding.
pub fn solve_kepler(m_rad: f64, e: f64) -> f64 {
    if !m_rad.is_finite() || e.is_nan() {
        return f64::NAN;
    }
    let e = e.clamp(0.0, 1.0);
    // `%` is exact; the whole turns removed are added back as m_rad − m.
    let mut m = m_rad % TAU;
    if m > PI {
        m -= TAU;
    } else if m < -PI {
        m += TAU;
    }
    let turns = m_rad - m;
    turns + kepler_reduced(m.abs(), e).copysign(m)
}

/// E for M ∈ [0, π]. f(E) = (E − sin E) + (1 − e) sin E − M is increasing and convex on
/// [0, π], so Newton from any point where f ≥ 0 falls monotonically onto the root. Each start
/// below is such a point: f(π) = π − M; f(M + e) = e(1 − sin(M + e)); f(M/(1 − e)) ≥ 0 since
/// sin E ≤ E; and E − sin E ≥ (E³/6)(1 − π²/20) on [0, π], so f(∛(12M)) ≥ 0.013 M.
fn kepler_reduced(m: f64, e: f64) -> f64 {
    if m == 0.0 {
        return 0.0;
    }
    let one_minus_e = 1.0 - e;
    let mut x = PI.min(m + e).min((12.0 * m).cbrt());
    if one_minus_e > 0.0 {
        x = x.min(m / one_minus_e);
    }
    for _ in 0..MAX_ITER {
        let half = (0.5 * x).sin();
        let f = x_minus_sin(x) + one_minus_e * x.sin() - m;
        let slope = one_minus_e + 2.0 * e * half * half; // 1 − e cos E
        let step = f / slope;
        let next = x - step;
        // f ≤ 0 within rounding (or no slope left): x is the root.
        if !(next < x) || next <= 0.0 {
            break;
        }
        x = next;
        if step <= 4.0 * f64::EPSILON * x {
            break;
        }
    }
    x
}

/// Hyperbolic anomaly H for mean anomaly `m`, e > 1 (e sinh H − H = M).
///
/// Any finite M; e is clamped to ≥ 1 (e = 1 solves sinh H − H = M). Converged to rounding.
pub fn solve_kepler_hyperbolic(m: f64, e: f64) -> f64 {
    if !m.is_finite() || e.is_nan() {
        return f64::NAN;
    }
    let e = e.max(1.0);
    let target = m.abs();
    if target == 0.0 {
        return 0.0;
    }
    let e_minus_1 = e - 1.0;
    // f(H) = (e − 1) sinh H + (sinh H − H) − M: increasing and convex for H ≥ 0. Upper
    // bounds of the root: sinh H − H ≥ H³/6 gives ∛(6M); (e − 1) sinh H ≤ M gives
    // asinh(M/(e − 1)); and with any bound h, the root ≤ asinh((M + h)/e).
    let mut x = (6.0 * target).cbrt();
    if e_minus_1 > 0.0 {
        x = x.min((target / e_minus_1).asinh()).min(target / e_minus_1);
    }
    x = x.min(((target + x) / e).asinh());
    for _ in 0..MAX_ITER {
        let half = (0.5 * x).sinh();
        let f = e_minus_1 * x.sinh() + sinh_minus_x(x) - target;
        let slope = e_minus_1 + 2.0 * e * half * half; // e cosh H − 1
        let step = f / slope;
        let next = x - step;
        if !(next < x) || next <= 0.0 {
            break;
        }
        x = next;
        if step <= 4.0 * f64::EPSILON * x {
            break;
        }
    }
    x.copysign(m)
}

/// Propagates state `r0` [m], `v0` [m/s] by `dt_s` with universal variables (Stumpff C, S);
/// any conic.
///
/// The universal Kepler equation √μ Δt = σ₀χ²C + (1 − αr₀)χ³S + r₀χ (σ₀ = r₀·v₀/√μ,
/// α = 1/a, z = αχ²) is solved for χ by Newton, falling back to Laguerre's step (Conway 1986)
/// and then to bisection whenever a step leaves the bracket of the root (the residual
/// increases with χ, its slope being r). For ellipses Δt is first reduced modulo the period.
/// The state follows from the Lagrange f, g coefficients. r₀ = 0 returns the input; μ ≤ 0 is
/// free motion.
pub fn propagate_universal(r0: DVec3, v0: DVec3, mu: f64, dt_s: f64) -> (DVec3, DVec3) {
    let r0n = r0.length();
    if r0n == 0.0 {
        return (r0, v0);
    }
    if mu <= 0.0 {
        return (r0 + v0 * dt_s, v0);
    }
    let sqrt_mu = mu.sqrt();
    let sigma0 = r0.dot(v0) / sqrt_mu;
    let alpha = 2.0 / r0n - v0.length_squared() / mu;
    let mut dt = dt_s;
    if alpha > 0.0 {
        let period = TAU / (sqrt_mu * alpha * alpha.sqrt());
        if period.is_finite() && dt.abs() > 0.5 * period {
            dt -= period * (dt / period).round();
        }
    }
    let kepler = UniversalKepler {
        r0: r0n,
        sigma0,
        alpha,
        target: sqrt_mu * dt,
    };
    let chi = kepler.solve(kepler.first_guess(r0.dot(v0), mu, dt));
    let z = alpha * chi * chi;
    let (c, s) = stumpff(z);
    let chi2 = chi * chi;
    let f = 1.0 - chi2 * c / r0n;
    // g = Δt − χ³S/√μ, written without the cancellation against Δt.
    let g = (sigma0 * chi2 * c + r0n * chi * (1.0 - z * s)) / sqrt_mu;
    let r = f * r0 + g * v0;
    let rn = match kepler.radius(chi, z, c, s) {
        rn if rn > 0.0 => rn,
        _ => r.length(),
    };
    let f_dot = sqrt_mu / (rn * r0n) * chi * (z * s - 1.0);
    let g_dot = 1.0 - chi2 * c / rn;
    (r, f_dot * r0 + g_dot * v0)
}

/// The universal Kepler equation F(χ) = 0 for one propagation.
struct UniversalKepler {
    r0: f64,
    sigma0: f64,
    alpha: f64,
    /// √μ Δt.
    target: f64,
}

impl UniversalKepler {
    /// The radius at χ, which is dF/dχ.
    fn radius(&self, chi: f64, z: f64, c: f64, s: f64) -> f64 {
        self.sigma0 * chi * (1.0 - z * s) + (1.0 - self.alpha * self.r0) * chi * chi * c + self.r0
    }

    /// F, F′ and F″ at χ.
    fn eval(&self, chi: f64) -> (f64, f64, f64) {
        let z = self.alpha * chi * chi;
        let (c, s) = stumpff(z);
        let chi2 = chi * chi;
        let k = 1.0 - self.alpha * self.r0;
        let f = self.sigma0 * chi2 * c + k * chi2 * chi * s + self.r0 * chi - self.target;
        let f1 = self.radius(chi, z, c, s);
        let f2 = self.sigma0 * (1.0 - z * c) + k * chi * (1.0 - z * s);
        (f, f1, f2)
    }

    /// Vallado's starting values (Algorithm 8): √μ Δt α for ellipses, the logarithmic form for
    /// hyperbolas, else the first-order √μ Δt / r₀. `rv` = r₀·v₀.
    fn first_guess(&self, rv: f64, mu: f64, dt: f64) -> f64 {
        let simple = self.target / self.r0;
        let guess = if self.alpha > 0.0 {
            self.target * self.alpha
        } else if self.alpha < 0.0 {
            let a = 1.0 / self.alpha;
            let sign = dt.signum();
            let num = -2.0 * mu * self.alpha * dt;
            let den = rv + sign * (-mu * a).sqrt() * (1.0 - self.r0 * self.alpha);
            sign * (-a).sqrt() * (num / den).ln()
        } else {
            simple
        };
        if guess.is_finite() && guess * self.target > 0.0 {
            guess
        } else {
            simple
        }
    }

    /// The root χ of F, from `guess`: bracket by doubling, then Newton → Laguerre → bisection.
    fn solve(&self, guess: f64) -> f64 {
        if self.target == 0.0 {
            return 0.0;
        }
        let dir = self.target.signum();
        // F(0) = −target. Walk outward until F changes sign (or overflows to ±∞/NaN, which
        // only happens far past the root).
        let (mut near, mut far) = (0.0, guess);
        for _ in 0..2100 {
            let (f, _, _) = self.eval(far);
            if !(f * dir < 0.0) {
                break;
            }
            near = far;
            far *= 2.0;
            if !far.is_finite() {
                break;
            }
        }
        let (mut lo, mut hi) = if dir > 0.0 { (near, far) } else { (far, near) };
        let mut x = if guess > lo && guess < hi {
            guess
        } else {
            0.5 * (lo + hi)
        };
        for _ in 0..MAX_ITER {
            let (f, f1, f2) = self.eval(x);
            if f == 0.0 {
                return x;
            }
            if f < 0.0 {
                lo = x;
            } else if f > 0.0 || dir > 0.0 {
                hi = x; // past the root (a NaN residual means overflow far past it)
            } else {
                lo = x;
            }
            let mut next = x - f / f1;
            if !(next > lo && next < hi) {
                // Laguerre with n = 5 (Conway 1986).
                let n = 5.0;
                let disc = ((n - 1.0) * (n - 1.0) * f1 * f1 - n * (n - 1.0) * f * f2)
                    .abs()
                    .sqrt();
                next = x - n * f / (f1 + disc.copysign(f1));
                if !(next > lo && next < hi) {
                    next = 0.5 * (lo + hi);
                }
            }
            let tol = 4.0 * f64::EPSILON * next.abs();
            if (next - x).abs() <= tol || hi - lo <= 4.0 * f64::EPSILON * lo.abs().max(hi.abs()) {
                return next;
            }
            x = next;
        }
        x
    }
}

/// Stumpff functions C(z) = (1 − cos √z)/z and S(z) = (√z − sin √z)/√z³, continued to z < 0
/// with cosh and sinh (Bate, Mueller & White 1971, §4.4). Their Taylor series
/// C = Σ (−z)ᵏ/(2k + 2)!, S = Σ (−z)ᵏ/(2k + 3)! are used for |z| < 1, not only the |z| < 1e-6
/// the brief requires: the closed forms lose up to |log₁₀ z| digits to cancellation there.
fn stumpff(z: f64) -> (f64, f64) {
    if z.abs() < 1.0 {
        let (mut c, mut s) = (0.0, 0.0);
        let (mut term_c, mut term_s) = (0.5, 1.0 / 6.0);
        for k in 0..12 {
            c += term_c;
            s += term_s;
            let j = (2 * k + 3) as f64;
            term_c *= -z / (j * (j + 1.0));
            term_s *= -z / ((j + 1.0) * (j + 2.0));
        }
        (c, s)
    } else if z > 0.0 {
        let root = z.sqrt();
        let half = (0.5 * root).sin();
        (2.0 * half * half / z, x_minus_sin(root) / (z * root))
    } else {
        let root = (-z).sqrt();
        let half = (0.5 * root).sinh();
        (2.0 * half * half / -z, sinh_minus_x(root) / (-z * root))
    }
}

/// x − sin x without cancellation: its Taylor series x³/3! − x⁵/5! + … below |x| = 1.
fn x_minus_sin(x: f64) -> f64 {
    if x.abs() >= 1.0 {
        return x - x.sin();
    }
    let x2 = x * x;
    let mut term = x * x2 / 6.0;
    let mut sum = term;
    for n in 1..12 {
        let k = (2 * n + 2) as f64;
        term *= -x2 / (k * (k + 1.0));
        sum += term;
    }
    sum
}

/// sinh x − x without cancellation: x³/3! + x⁵/5! + … below |x| = 1.
fn sinh_minus_x(x: f64) -> f64 {
    if x.abs() >= 1.0 {
        return x.sinh() - x;
    }
    let x2 = x * x;
    let mut term = x * x2 / 6.0;
    let mut sum = term;
    for n in 1..12 {
        let k = (2 * n + 2) as f64;
        term *= x2 / (k * (k + 1.0));
        sum += term;
    }
    sum
}

/// Rotation from the perifocal frame (x to periapsis, z along the angular momentum) to the
/// reference frame: R_z(node) · R_x(i) · R_z(argp).
///
/// Active rotations; the matrix equals the element formula on the JPL "Approximate Positions
/// of the Planets" page.
pub fn perifocal_to_reference(i: f64, node: f64, argp: f64) -> DMat3 {
    DMat3::from_rotation_z(node) * DMat3::from_rotation_x(i) * DMat3::from_rotation_z(argp)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use uni_core::{AU, DAY};

    use super::*;
    use crate::bodies::GM_SUN_DE440_M3S2 as GM_SUN;

    /// 1 mm, the brief's position tolerance for 1 AU orbits.
    const MM: f64 = 1e-3;

    fn at_day(d: f64) -> Epoch {
        Epoch::from_days_since_j2000(d)
    }

    /// Meeus, *Astronomical Algorithms* 2nd ed., Example 30.a: e = 0.1, M = 5° → E = 5.554 589°.
    #[test]
    fn kepler_meeus_30a() {
        assert!((solve_kepler(5f64.to_radians(), 0.1).to_degrees() - 5.554589).abs() < 1e-6);
    }

    /// |E − e sin E − M| < 1e-13 over 1000 mean anomalies in [−4π, 4π] plus tiny and special
    /// ones, for e up to 0.999 (the near-parabolic cube-root regime included).
    #[test]
    fn kepler_residual_all_e() {
        let specials = [
            0.0,
            1e-300,
            -1e-300,
            1e-15,
            1e-9,
            -1e-6,
            1e-3,
            PI,
            -PI,
            2.0 * PI,
        ];
        for e in [0.0, 0.1, 0.5, 0.9, 0.99, 0.999] {
            let sweep = (0..1000).map(|k| -4.0 * PI + 8.0 * PI * k as f64 / 999.0);
            for m in sweep.chain(specials) {
                let ea = solve_kepler(m, e);
                let residual = ea - e * ea.sin() - m;
                assert!(
                    residual.abs() < 1e-13,
                    "e {e} M {m}: E {ea}, residual {residual:e}"
                );
            }
        }
    }

    /// |e sinh H − H − M| < 1e-12·max(1, |M|) for |M| log-spaced over 1e-8 … 1e4, both signs.
    #[test]
    fn hyperbolic_residual() {
        for e in [1.001, 1.5, 5.0] {
            assert_eq!(solve_kepler_hyperbolic(0.0, e), 0.0);
            for k in 0..1000 {
                let size = 10f64.powf(-8.0 + 12.0 * (k / 2) as f64 / 499.0);
                let m = if k % 2 == 0 { size } else { -size };
                let h = solve_kepler_hyperbolic(m, e);
                let residual = e * h.sinh() - h - m;
                assert!(
                    residual.abs() < 1e-12 * m.abs().max(1.0),
                    "e {e} M {m}: H {h}, residual {residual:e}"
                );
            }
        }
    }

    /// Vallado, *Fundamentals of Astrodynamics and Applications* 4th ed. (2013), Example 2-4.
    #[test]
    fn universal_vallado_2_4() {
        let r0 = DVec3::new(1131.340, -2282.343, 6672.423); // km
        let v0 = DVec3::new(-5.64305, 4.30333, 2.42879); // km/s
        let (r, v) = propagate_universal(r0, v0, 398_600.4418, 40.0 * 60.0);
        let r_book = DVec3::new(-4219.7527, 4363.0292, -3958.7666);
        let v_book = DVec3::new(3.689866, -1.916735, -6.112511);
        assert!((r - r_book).abs().max_element() < 1e-3, "{r:?}");
        assert!((v - v_book).abs().max_element() < 1e-6, "{v:?}");
    }

    /// One propagation over ±100 periods (or ±1e9 s) keeps the specific energy and the angular
    /// momentum to 1e-10. The energy error is relative to v₀²/2 + μ/r₀, because a parabola's
    /// energy is zero.
    #[test]
    fn universal_conserves_energy_and_h() {
        for e in [0.3, 0.9999, 1.0, 1.7] {
            let q = AU;
            let rot = perifocal_to_reference(0.4, 1.0, 2.0);
            let r0 = rot * DVec3::new(q, 0.0, 0.0);
            let v0 = rot * DVec3::new(0.0, (GM_SUN * (1.0 + e) / q).sqrt(), 0.0);
            let energy = |r: DVec3, v: DVec3| v.length_squared() / 2.0 - GM_SUN / r.length();
            let scale = v0.length_squared() / 2.0 + GM_SUN / r0.length();
            let h0 = r0.cross(v0);
            let mut spans = vec![1e9, -1e9, 1.234_567e5];
            if e < 1.0 {
                let a = q / (1.0 - e);
                let period = 2.0 * PI * (a * a * a / GM_SUN).sqrt();
                if 100.0 * period < 1e11 {
                    spans.extend([100.0 * period, -100.0 * period, 100.37 * period]);
                }
            }
            for dt in spans {
                let (r, v) = propagate_universal(r0, v0, GM_SUN, dt);
                assert!(r.is_finite() && v.is_finite(), "e {e} dt {dt}");
                let de = (energy(r, v) - energy(r0, v0)).abs() / scale;
                let dh = (r.cross(v) - h0).length() / h0.length();
                assert!(
                    de < 1e-10 && dh < 1e-10,
                    "e {e} dt {dt}: dE {de:e}, dh {dh:e}"
                );
            }
        }
    }

    /// Conic::state (classical anomalies) against propagate_universal from the periapsis
    /// state: two independent methods agree to 1 mm on q = 1 AU orbits over ±1 yr.
    ///
    /// Beyond that, for orbits of period < 3 yr, the check runs to 2 periods with 1 mm plus
    /// 4 mm per orbit: the periapsis state handed to propagate_universal is rounded to
    /// ≈ 1e-16, which fixes its a, and so its mean motion, only to ≈ 1e-15 (measured
    /// 3.3 mm after 1.3 orbits at a = 2 AU). Any error in either method's period reduction
    /// would be of order the orbit itself.
    #[test]
    fn conic_state_matches_universal() {
        let es = [0.0, 0.2, 0.5, 0.9, 0.97, 0.9999, 1.0, 1.0001, 1.5, 3.0];
        let mut worst = (0.0f64, 0.0f64);
        let mut worst_multi = 0.0f64;
        for e in es {
            let c = Conic {
                q_m: AU,
                e,
                i_rad: 0.4,
                node_rad: 1.2,
                argp_rad: 2.1,
                tp: at_day(1000.25),
                mu_m3s2: GM_SUN,
            };
            let rot = perifocal_to_reference(c.i_rad, c.node_rad, c.argp_rad);
            let (rp, vp) = c.state(c.tp);
            let vp_exact = (GM_SUN * (1.0 + e) / AU).sqrt();
            assert!(
                (rp - rot * DVec3::new(AU, 0.0, 0.0)).length() < MM,
                "e {e}: {rp:?}"
            );
            assert!(
                (vp - rot * DVec3::new(0.0, vp_exact, 0.0)).length() < 1e-9,
                "e {e}"
            );
            for days in [-365.0, -100.0, -1.0, 0.25, 10.0, 77.7, 200.0, 365.0] {
                let dt = days * DAY;
                let (r1, v1) = c.state(c.tp.add_seconds(dt));
                let (r2, v2) = propagate_universal(rp, vp, GM_SUN, dt);
                let (dr, dv) = ((r1 - r2).length(), (v1 - v2).length());
                worst = (worst.0.max(dr), worst.1.max(dv));
                assert!(dr < MM, "e {e} dt {dt}: |Δr| {dr} m");
                assert!(dv < 1e-9, "e {e} dt {dt}: |Δv| {dv} m/s");
            }
            if let Some(p) = c.period_s().filter(|&p| p < 3.0 * 365.25 * DAY) {
                for orbits in [1.3, -0.5, 2.0, -1.7] {
                    let dt = orbits * p;
                    let (r1, _) = c.state(c.tp.add_seconds(dt));
                    let (r2, _) = propagate_universal(rp, vp, GM_SUN, dt);
                    let dr = (r1 - r2).length();
                    worst_multi = worst_multi.max(dr);
                    let bound = MM * (1.0 + 4.0 * f64::abs(orbits));
                    assert!(dr < bound, "e {e}, {orbits} orbits: |Δr| {dr} m");
                }
            }
        }
        println!(
            "conic vs universal ±1 yr: worst |Δr| {:.3e} m, |Δv| {:.3e} m/s",
            worst.0, worst.1
        );
        println!("conic vs universal to 2 orbits: worst |Δr| {worst_multi:.3e} m");
    }

    /// from_state(state(t0)) reproduces the conic: the same elements where they are defined
    /// and the same states at other times to 1 mm. i = 0 gives node = 0 exactly; e = 0 keeps
    /// the state (the documented convention puts periapsis at the node).
    #[test]
    fn conic_from_state_roundtrip() {
        let tp = at_day(-512.5);
        for e in [0.0, 0.2, 0.97, 1.0, 1.5] {
            for i in [0.0, 0.5, PI] {
                let c0 = Conic {
                    q_m: AU,
                    e,
                    i_rad: i,
                    node_rad: 0.7,
                    argp_rad: 1.9,
                    tp,
                    mu_m3s2: GM_SUN,
                };
                for d0 in [-20.0, 5.0, 90.0] {
                    let t0 = tp.add_seconds(d0 * DAY);
                    let (r, v) = c0.state(t0);
                    let c1 = Conic::from_state(r, v, GM_SUN, t0);
                    assert!((c1.e - e).abs() < 1e-12, "e {e} i {i}: e1 {}", c1.e);
                    assert!((c1.q_m - AU).abs() < MM, "e {e} i {i}: q1 {}", c1.q_m);
                    assert!((c1.i_rad - i).abs() < 1e-12, "e {e} i {i}: i1 {}", c1.i_rad);
                    if i == 0.0 {
                        assert_eq!(c1.node_rad, 0.0, "e {e}: equatorial node convention");
                    }
                    if e == 0.2 && i == 0.5 {
                        assert!((c1.node_rad - 0.7).abs() < 1e-13);
                        assert!((c1.argp_rad - 1.9).abs() < 1e-12);
                        assert!(c1.tp.seconds_since(tp).abs() < 1e-6);
                    }
                    for d1 in [-60.0, 1.0, 200.0] {
                        let t1 = tp.add_seconds(d1 * DAY);
                        let (r0, v0) = c0.state(t1);
                        let (r1, v1) = c1.state(t1);
                        let (dr, dv) = ((r1 - r0).length(), (v1 - v0).length());
                        assert!(dr < MM, "e {e} i {i} t0 {d0} t1 {d1}: |Δr| {dr} m");
                        assert!(dv < 1e-9, "e {e} i {i} t0 {d0} t1 {d1}: |Δv| {dv} m/s");
                    }
                }
            }
        }
        // An exactly circular equatorial state: e = 0, i = 0, node = 0 and argp = 0 by
        // convention, so periapsis is the current position and tp = t.
        let t = at_day(3.0);
        let c = Conic::from_state(DVec3::X, DVec3::Y, 1.0, t);
        assert_eq!((c.e, c.i_rad, c.node_rad, c.argp_rad), (0.0, 0.0, 0.0, 0.0));
        assert_eq!(c.tp, t);
        assert_eq!(c.q_m, 1.0);
    }

    /// The perifocal rotation equals the element formula of the JPL "Approximate Positions of
    /// the Planets" page (x_ecl, y_ecl, z_ecl in terms of ω, Ω, I).
    #[test]
    fn perifocal_matches_jpl_formula() {
        for (i, node, w) in [(0.1, 0.2, 0.3), (2.9, -1.0, 4.0), (0.0, 1.0, 1.0)] {
            let m = perifocal_to_reference(i, node, w);
            let (ci, si, co, so, cw, sw) =
                (i.cos(), i.sin(), node.cos(), node.sin(), w.cos(), w.sin());
            let x_col = DVec3::new(cw * co - sw * so * ci, cw * so + sw * co * ci, sw * si);
            let y_col = DVec3::new(-sw * co - cw * so * ci, -sw * so + cw * co * ci, cw * si);
            assert!((m.x_axis - x_col).length() < 1e-15);
            assert!((m.y_axis - y_col).length() < 1e-15);
            assert!((m.z_axis - x_col.cross(y_col)).length() < 1e-15);
        }
    }

    /// from_mean_anomaly sets tp so that state(epoch) sits at mean anomaly m0, and the period
    /// is 2π√(a³/μ); one period later the state repeats.
    #[test]
    fn from_mean_anomaly_and_period() {
        let epoch = at_day(100.0);
        let a = 2.5 * AU;
        let c = Conic::from_mean_anomaly(a, 0.3, 0.2, 0.4, 0.6, 7.0, epoch, GM_SUN);
        let period = c.period_s().expect("elliptic");
        assert!((period / (2.0 * PI * (a * a * a / GM_SUN).sqrt()) - 1.0).abs() < 1e-14);
        // Mean anomaly at epoch, reduced to (−π, π]: 7 − 2π.
        let m_at_epoch = 2.0 * PI * epoch.seconds_since(c.tp) / period;
        assert!(
            (m_at_epoch - (7.0 - 2.0 * PI)).abs() < 1e-12,
            "{m_at_epoch}"
        );
        assert!((c.q_m - a * 0.7).abs() < MM);
        let (r0, _) = c.state(epoch);
        let (r1, _) = c.state(epoch.add_seconds(period));
        assert!((r1 - r0).length() < MM);
        let hyperbola = Conic { e: 1.2, ..c };
        assert_eq!(hyperbola.period_s(), None);
    }

    /// Every function returns finite values for extreme finite input; NaN only from NaN.
    #[test]
    fn finite_for_extreme_input() {
        for e in [0.0, 0.5, 0.999_999_999, 1.0] {
            for m in [1e300, -1e300, 1e15, -7.5] {
                assert!(solve_kepler(m, e).is_finite(), "e {e} M {m}");
            }
        }
        for e in [1.0, 1.0 + 1e-15, 2.0] {
            for m in [1e300, -1e300, 1e-300] {
                assert!(solve_kepler_hyperbolic(m, e).is_finite(), "e {e} M {m}");
            }
        }
        assert!(solve_kepler(f64::NAN, 0.1).is_nan());
        let rot = perifocal_to_reference(0.4, 1.0, 2.0);
        for (e, dt) in [(0.5, 1e15), (0.5, -3e13), (1.0, 1e15), (3.0, 1e15)] {
            let r0 = rot * DVec3::new(AU, 0.0, 0.0);
            let v0 = rot * DVec3::new(0.0, (GM_SUN * (1.0 + e) / AU).sqrt(), 0.0);
            let (r, v) = propagate_universal(r0, v0, GM_SUN, dt);
            assert!(r.is_finite() && v.is_finite(), "e {e} dt {dt}: {r:?} {v:?}");
        }
        // A radial state has no periapsis distance: q = 0 and the conic stays finite.
        let radial = Conic::from_state(
            DVec3::new(AU, 0.0, 0.0),
            DVec3::new(1e4, 0.0, 0.0),
            GM_SUN,
            at_day(0.0),
        );
        assert_eq!(radial.q_m, 0.0);
        let (r, v) = radial.state(at_day(10.0));
        assert!(r.is_finite() && v.is_finite());
        for e in [0.0, 0.5, 0.97, 1.0, 1.5] {
            let c = Conic {
                q_m: AU,
                e,
                i_rad: 0.4,
                node_rad: 1.0,
                argp_rad: 2.0,
                tp: at_day(0.0),
                mu_m3s2: GM_SUN,
            };
            for d in [3.65e9, -3.65e9, 1e-9] {
                let (r, v) = c.state(at_day(d));
                assert!(r.is_finite() && v.is_finite(), "e {e} day {d}");
            }
        }
    }
}
