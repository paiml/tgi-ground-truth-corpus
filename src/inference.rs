//! Inference backend patterns.
//!
//! Backend inference engine patterns extracted from TGI's backend layer.
//!
//! # TGI Source
//!
//! Patterns derived from `backends/v3/src/backend.rs`:
//! - Backend initialization
//! - Model loading
//! - Inference execution
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `realizar::inference` for model inference.

use crate::{Error, Result};

/// Backend configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendConfig {
    /// Model path or identifier.
    pub model_id: String,

    /// Device to run on (cpu, cuda:0, etc.).
    pub device: String,

    /// Data type for inference.
    pub dtype: DataType,

    /// Maximum sequence length.
    pub max_sequence_length: usize,

    /// Whether to use flash attention.
    pub use_flash_attention: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            device: "cpu".to_string(),
            dtype: DataType::Float16,
            max_sequence_length: 4096,
            use_flash_attention: false,
        }
    }
}

impl BackendConfig {
    /// Create a new config with model ID.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    /// Set the device.
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    /// Set the data type.
    pub const fn dtype(mut self, dtype: DataType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set max sequence length.
    pub const fn max_sequence_length(mut self, len: usize) -> Self {
        self.max_sequence_length = len;
        self
    }

    /// Enable flash attention.
    pub const fn flash_attention(mut self, enabled: bool) -> Self {
        self.use_flash_attention = enabled;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.model_id.is_empty() {
            return Err(Error::config("model_id is required"));
        }
        if self.max_sequence_length == 0 {
            return Err(Error::config("max_sequence_length must be > 0"));
        }
        Ok(())
    }
}

/// Data type for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataType {
    /// 32-bit float.
    Float32,
    /// 16-bit float.
    #[default]
    Float16,
    /// Brain float 16.
    BFloat16,
    /// 8-bit integer.
    Int8,
}

impl DataType {
    /// Bytes per element.
    pub const fn bytes(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
            Self::Int8 => 1,
        }
    }

    /// Name of the dtype.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Int8 => "int8",
        }
    }
}

/// Backend state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendState {
    /// Not initialized.
    #[default]
    Uninitialized,
    /// Loading model.
    Loading,
    /// Ready for inference.
    Ready,
    /// Error state.
    Error,
}

impl BackendState {
    /// Check if ready for inference.
    pub const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Check if in error state.
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error)
    }
}

/// Inference backend.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::inference::{InferenceBackend, BackendConfig, DataType};
///
/// let config = BackendConfig::new("meta-llama/Llama-2-7b")
///     .device("cuda:0")
///     .dtype(DataType::Float16);
///
/// let backend = InferenceBackend::new(config);
/// assert!(!backend.is_ready());
/// ```
#[derive(Debug)]
pub struct InferenceBackend {
    config: BackendConfig,
    state: BackendState,
}

impl InferenceBackend {
    /// Create a new backend.
    pub fn new(config: BackendConfig) -> Self {
        Self {
            config,
            state: BackendState::Uninitialized,
        }
    }

    /// Create with default config.
    pub fn with_model(model_id: impl Into<String>) -> Self {
        Self::new(BackendConfig::new(model_id))
    }

    /// Get configuration.
    pub const fn config(&self) -> &BackendConfig {
        &self.config
    }

    /// Get current state.
    pub const fn state(&self) -> BackendState {
        self.state
    }

    /// Check if ready.
    pub const fn is_ready(&self) -> bool {
        self.state.is_ready()
    }

    /// Initialize the backend.
    pub fn initialize(&mut self) -> Result<()> {
        self.config.validate()?;
        self.state = BackendState::Loading;
        // Actual loading would happen here
        self.state = BackendState::Ready;
        Ok(())
    }

    /// Set error state.
    pub fn set_error(&mut self) {
        self.state = BackendState::Error;
    }

    /// Reset to uninitialized.
    pub fn reset(&mut self) {
        self.state = BackendState::Uninitialized;
    }
}

impl Default for InferenceBackend {
    fn default() -> Self {
        Self::new(BackendConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert!(config.model_id.is_empty());
        assert_eq!(config.device, "cpu");
        assert_eq!(config.dtype, DataType::Float16);
    }

    #[test]
    fn test_backend_config_builder() {
        let config = BackendConfig::new("test-model")
            .device("cuda:0")
            .dtype(DataType::Float32)
            .max_sequence_length(8192)
            .flash_attention(true);

        assert_eq!(config.model_id, "test-model");
        assert_eq!(config.device, "cuda:0");
        assert_eq!(config.dtype, DataType::Float32);
        assert_eq!(config.max_sequence_length, 8192);
        assert!(config.use_flash_attention);
    }

    #[test]
    fn test_backend_config_validate() {
        let valid = BackendConfig::new("model");
        assert!(valid.validate().is_ok());

        let empty_model = BackendConfig::default();
        assert!(empty_model.validate().is_err());

        let zero_len = BackendConfig::new("model").max_sequence_length(0);
        assert!(zero_len.validate().is_err());
    }

    #[test]
    fn test_dtype_bytes() {
        assert_eq!(DataType::Float32.bytes(), 4);
        assert_eq!(DataType::Float16.bytes(), 2);
        assert_eq!(DataType::BFloat16.bytes(), 2);
        assert_eq!(DataType::Int8.bytes(), 1);
    }

    #[test]
    fn test_dtype_name() {
        assert_eq!(DataType::Float32.name(), "float32");
        assert_eq!(DataType::Float16.name(), "float16");
        assert_eq!(DataType::BFloat16.name(), "bfloat16");
        assert_eq!(DataType::Int8.name(), "int8");
    }

    #[test]
    fn test_backend_state() {
        assert!(!BackendState::Uninitialized.is_ready());
        assert!(!BackendState::Loading.is_ready());
        assert!(BackendState::Ready.is_ready());
        assert!(!BackendState::Error.is_ready());

        assert!(!BackendState::Ready.is_error());
        assert!(BackendState::Error.is_error());
    }

    #[test]
    fn test_backend_creation() {
        let backend = InferenceBackend::with_model("test");
        assert!(!backend.is_ready());
        assert_eq!(backend.state(), BackendState::Uninitialized);
    }

    #[test]
    fn test_backend_initialize() {
        let mut backend = InferenceBackend::with_model("test");
        assert!(backend.initialize().is_ok());
        assert!(backend.is_ready());
    }

    #[test]
    fn test_backend_initialize_invalid() {
        let mut backend = InferenceBackend::default();
        assert!(backend.initialize().is_err());
    }

    #[test]
    fn test_backend_error_and_reset() {
        let mut backend = InferenceBackend::with_model("test");
        backend.initialize().unwrap();
        assert!(backend.is_ready());

        backend.set_error();
        assert!(backend.state().is_error());

        backend.reset();
        assert_eq!(backend.state(), BackendState::Uninitialized);
    }

    #[test]
    fn test_backend_config_access() {
        let config = BackendConfig::new("my-model").device("cuda:1");
        let backend = InferenceBackend::new(config);

        assert_eq!(backend.config().model_id, "my-model");
        assert_eq!(backend.config().device, "cuda:1");
    }
}
