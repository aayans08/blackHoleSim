//! Integer hashes, the Rust twins of `shaders/common/hash.wgsl`.
//! Source: Jarzynski & Olano 2020, "Hash Functions for GPU Rendering", JCGT 9(3).

use glam::UVec3;

/// PCG3D hash (Jarzynski & Olano 2020); bit-identical to `pcg3d` in hash.wgsl.
pub fn pcg3d(v: UVec3) -> UVec3 {
    // One LCG step per lane (Numerical Recipes constants), then cross-lane mixing around an
    // xorshift. All arithmetic wraps mod 2^32, as u32 does in WGSL.
    let mut v = v.wrapping_mul(UVec3::splat(1_664_525)).wrapping_add(UVec3::splat(1_013_904_223));
    v = pcg3d_mix(v);
    v ^= v >> 16_u32;
    pcg3d_mix(v)
}

/// PCG3D's cross-lane step: each lane, in order, adds the product of the other two.
fn pcg3d_mix(mut v: UVec3) -> UVec3 {
    v.x = v.x.wrapping_add(v.y.wrapping_mul(v.z));
    v.y = v.y.wrapping_add(v.z.wrapping_mul(v.x));
    v.z = v.z.wrapping_add(v.x.wrapping_mul(v.y));
    v
}

/// SplitMix64 (Steele, Lea & Flood 2014); mixes `x + 0x9E3779B97F4A7C15`.
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Folds `parts` into one 64-bit hash with [`splitmix64`]: `h ← splitmix64(h ^ part)` from
/// `h = 0`. Order matters, and the empty slice hashes to 0.
pub fn hash_u64s(parts: &[u64]) -> u64 {
    parts.iter().fold(0, |h, &part| splitmix64(h ^ part))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg3d_golden() {
        assert_eq!(pcg3d(UVec3::new(0, 0, 0)), UVec3::new(0x9bafd7c6, 0xa8e88a6b, 0x3f15482c));
        assert_eq!(pcg3d(UVec3::new(1, 2, 3)), UVec3::new(0xfa9f79a6, 0x48f2f44c, 0x596f5ab1));
        assert_eq!(
            pcg3d(UVec3::new(0xFFFF_FFFF, 7, 42)),
            UVec3::new(0xe387d2d4, 0x47aa5f4a, 0xd51ae662)
        );
    }

    #[test]
    fn splitmix64_golden() {
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
        // Second output of the SplitMix64 generator seeded with 0.
        assert_eq!(splitmix64(0x9E37_79B9_7F4A_7C15), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn hash_u64s_golden() {
        // Pinned: procedural content is keyed by this fold, so changing it re-rolls the universe.
        assert_eq!(hash_u64s(&[1, 2, 3]), 0xD073_4750_FDE3_62B3);
        assert_eq!(hash_u64s(&[3, 2, 1]), 0xD62A_7F6E_291F_EF64);
        assert_eq!(hash_u64s(&[0]), splitmix64(0));
        assert_ne!(hash_u64s(&[0]), hash_u64s(&[0, 0]));
        assert_eq!(hash_u64s(&[]), 0);
    }
}
