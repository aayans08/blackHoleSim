//! WGSL library with `#import "path"` expansion, plus naga validation.

use std::collections::HashMap;

/// Errors from composing or validating WGSL.
#[derive(Debug, thiserror::Error)]
pub enum ShaderError {
    #[error("missing import {0}")]
    MissingImport(String),
    #[error("naga: {0}")]
    Naga(String),
}

/// A set of WGSL sources keyed by path (for example `"common/camera.wgsl"`).
pub struct ShaderLib {
    sources: HashMap<&'static str, &'static str>,
}

impl ShaderLib {
    /// A library with every `common/*.wgsl` file registered.
    pub fn with_common() -> Self {
        todo!()
    }

    /// Registers `src` under `path`.
    pub fn add(&mut self, path: &'static str, src: &'static str) {
        todo!()
    }

    /// Expands `#import "path"` recursively, including each file at most once.
    pub fn compose(&self, entry: &str) -> Result<String, ShaderError> {
        todo!()
    }

    /// Composes `entry` and creates a shader module from it.
    pub fn module(
        &self,
        device: &wgpu::Device,
        entry: &str,
    ) -> Result<wgpu::ShaderModule, ShaderError> {
        todo!()
    }
}

/// Parses and validates WGSL with naga.
pub fn validate_wgsl(src: &str) -> Result<naga::Module, ShaderError> {
    todo!()
}
