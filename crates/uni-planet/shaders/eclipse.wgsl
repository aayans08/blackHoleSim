// planet/eclipse.wgsl: partial eclipses of the Sun's disk (Task B0, shared by the bodies pass B1
// and the ring pass B2). No bindings, so any pass can import it.
//
// Twin of `uni_planet::sun_visible_fraction` (crates/uni-planet/src/eclipse.rs): keep the order
// of operations identical. B1's GPU probe compares the two.
//
// Fraction of the Sun's disk visible past one occluder, both uniform planar disks: angular
// radii sun_ang_r and occ_ang_r [rad], centre separation sep [rad]. The covered area of a
// partial overlap is the lens of two circles, r_s^2 (a - sin a cos a) + r_o^2 (b - sin b cos b),
// with cos a = (d^2 + r_s^2 - r_o^2) / (2 d r_s) and cos b = (d^2 + r_o^2 - r_s^2) / (2 d r_o)
// (Weisstein, "Circle-Circle Intersection", MathWorld), worked in units of r_s.

#import "common/constants.wgsl"

fn sun_visible_fraction(sun_ang_r: f32, occ_ang_r: f32, sep: f32) -> f32 {
    let d = abs(sep);
    if (!(occ_ang_r > 0.0)) {
        return 1.0;
    }
    if (!(sun_ang_r > 0.0)) {
        return select(1.0, 0.0, d < occ_ang_r);
    }
    // Apart or touching; covering; inside (annular).
    if (d >= sun_ang_r + occ_ang_r) {
        return 1.0;
    }
    if (occ_ang_r >= sun_ang_r + d) {
        return 0.0;
    }
    let rho = occ_ang_r / sun_ang_r;
    if (d + occ_ang_r <= sun_ang_r) {
        return 1.0 - rho * rho;
    }
    // Partial: d > |r_s - r_o| >= 0 here.
    let delta = d / sun_ang_r;
    let cos_a = clamp((delta * delta + 1.0 - rho * rho) / (2.0 * delta), -1.0, 1.0);
    let cos_b = clamp((delta * delta + rho * rho - 1.0) / (2.0 * delta * rho), -1.0, 1.0);
    let seg_a = acos(cos_a) - cos_a * sqrt(1.0 - cos_a * cos_a);
    let seg_b = acos(cos_b) - cos_b * sqrt(1.0 - cos_b * cos_b);
    return clamp(1.0 - (seg_a + rho * rho * seg_b) / PI, 0.0, 1.0);
}
