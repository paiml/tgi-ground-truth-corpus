//! Quantization patterns.
//!
//! Model quantization and dequantization patterns.
//!
//! # TGI Source
//!
//! Patterns derived from `backends/llamacpp/src/quantize.rs`:
//! - GGUF format support
//! - Dequantization kernels
//! - Mixed precision inference
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `aprender::quantize` for model quantization.

/// Quantization types supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // Match GGUF naming conventions
pub enum QuantType {
    /// 4-bit quantization.
    Q4_0,
    /// 4-bit quantization with K-quants.
    Q4_K,
    /// 5-bit quantization with K-quants.
    Q5_K,
    /// 6-bit quantization with K-quants.
    Q6_K,
    /// 8-bit quantization.
    Q8_0,
    /// 16-bit float.
    F16,
    /// 32-bit float.
    F32,
}

impl QuantType {
    /// Bits per weight.
    pub const fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Q4_0 | Self::Q4_K => 4.5,
            Self::Q5_K => 5.5,
            Self::Q6_K => 6.5,
            Self::Q8_0 => 8.0,
            Self::F16 => 16.0,
            Self::F32 => 32.0,
        }
    }

    /// Compression ratio vs F32.
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bits_per_weight()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_type_bits() {
        assert!((QuantType::Q4_0.bits_per_weight() - 4.5).abs() < 0.01);
        assert!((QuantType::F16.bits_per_weight() - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_compression_ratio() {
        assert!((QuantType::Q4_0.compression_ratio() - 7.1).abs() < 0.1);
        assert!((QuantType::F16.compression_ratio() - 2.0).abs() < 0.01);
    }
}
