//! Integer hashes, the Rust twins of `shaders/common/hash.wgsl`.
//! Source: Jarzynski & Olano 2020, "Hash Functions for GPU Rendering", JCGT 9(3).

use glam::UVec3;

/// PCG3D hash (Jarzynski & Olano 2020); bit-identical to `pcg3d` in hash.wgsl.
pub fn pcg3d(v: UVec3) -> UVec3 {
    todo!()
}

/// SplitMix64 (Steele, Lea & Flood 2014); mixes `x + 0x9E3779B97F4A7C15`.
pub fn splitmix64(x: u64) -> u64 {
    todo!()
}

/// Folds `parts` into one 64-bit hash with [`splitmix64`].
pub fn hash_u64s(parts: &[u64]) -> u64 {
    todo!()
}
