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
    fn test_quant_type_bits_all_variants() {
        // Test all quantization types for complete coverage
        assert!((QuantType::Q4_0.bits_per_weight() - 4.5).abs() < 0.01);
        assert!((QuantType::Q4_K.bits_per_weight() - 4.5).abs() < 0.01);
        assert!((QuantType::Q5_K.bits_per_weight() - 5.5).abs() < 0.01);
        assert!((QuantType::Q6_K.bits_per_weight() - 6.5).abs() < 0.01);
        assert!((QuantType::Q8_0.bits_per_weight() - 8.0).abs() < 0.01);
        assert!((QuantType::F16.bits_per_weight() - 16.0).abs() < 0.01);
        assert!((QuantType::F32.bits_per_weight() - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_compression_ratio_all_variants() {
        // Q4 variants: 32/4.5 ≈ 7.1
        assert!((QuantType::Q4_0.compression_ratio() - 7.1).abs() < 0.1);
        assert!((QuantType::Q4_K.compression_ratio() - 7.1).abs() < 0.1);

        // Q5_K: 32/5.5 ≈ 5.8
        assert!((QuantType::Q5_K.compression_ratio() - 5.8).abs() < 0.1);

        // Q6_K: 32/6.5 ≈ 4.9
        assert!((QuantType::Q6_K.compression_ratio() - 4.9).abs() < 0.1);

        // Q8_0: 32/8 = 4.0
        assert!((QuantType::Q8_0.compression_ratio() - 4.0).abs() < 0.01);

        // F16: 32/16 = 2.0
        assert!((QuantType::F16.compression_ratio() - 2.0).abs() < 0.01);

        // F32: 32/32 = 1.0 (no compression)
        assert!((QuantType::F32.compression_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_quant_type_equality() {
        assert_eq!(QuantType::Q4_0, QuantType::Q4_0);
        assert_ne!(QuantType::Q4_0, QuantType::Q4_K);
        assert_ne!(QuantType::F16, QuantType::F32);
    }

    #[test]
    fn test_quant_type_clone() {
        let q = QuantType::Q8_0;
        let q_clone = q;
        assert_eq!(q, q_clone);
    }

    #[test]
    fn test_quant_type_debug() {
        let debug_str = format!("{:?}", QuantType::Q4_K);
        assert!(debug_str.contains("Q4_K"));
    }
}
