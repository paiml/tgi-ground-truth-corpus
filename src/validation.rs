//! Request validation patterns.
//!
//! Input validation and token counting patterns extracted from TGI's
//! validation layer, ensuring requests meet model constraints before inference.
//!
//! # TGI Source
//!
//! Patterns derived from `router/src/validation.rs`:
//! - Token counting and limits
//! - Input length validation
//! - Parameter bounds checking
//! - Stop sequence validation
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to custom validation with `aprender::tokenizer` for token counting.
//!
//! # Examples
//!
//! ```rust
//! use tgi_gtc::validation::{RequestValidator, ValidationConfig, GenerateRequest};
//!
//! let config = ValidationConfig::builder()
//!     .max_input_tokens(4096)
//!     .max_total_tokens(8192)
//!     .max_stop_sequences(4)
//!     .build();
//!
//! let validator = RequestValidator::new(config);
//!
//! let request = GenerateRequest {
//!     inputs: "Hello, world!".to_string(),
//!     max_new_tokens: Some(100),
//!     temperature: Some(0.7),
//!     top_p: Some(0.9),
//!     stop: vec!["###".to_string()],
//!     ..Default::default()
//! };
//!
//! validator.validate(&request).expect("valid request");
//! ```

use crate::{Error, Result};

/// Configuration for request validation.
///
/// # TGI Source
///
/// Maps to `ValidatorConfig` in `router/src/validation.rs`.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::validation::ValidationConfig;
///
/// let config = ValidationConfig::builder()
///     .max_input_tokens(4096)
///     .max_total_tokens(8192)
///     .build();
///
/// assert_eq!(config.max_input_tokens, 4096);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationConfig {
    /// Maximum input tokens allowed.
    pub max_input_tokens: usize,

    /// Maximum total tokens (input + output).
    pub max_total_tokens: usize,

    /// Maximum number of stop sequences.
    pub max_stop_sequences: usize,

    /// Maximum length of each stop sequence.
    pub max_stop_sequence_length: usize,

    /// Maximum batch size for batch requests.
    pub max_batch_size: usize,

    /// Whether to allow empty inputs.
    pub allow_empty_input: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_input_tokens: 4096,
            max_total_tokens: 8192,
            max_stop_sequences: 4,
            max_stop_sequence_length: 256,
            max_batch_size: 32,
            allow_empty_input: false,
        }
    }
}

impl ValidationConfig {
    /// Create a new builder for `ValidationConfig`.
    pub fn builder() -> ValidationConfigBuilder {
        ValidationConfigBuilder::default()
    }

    /// Maximum new tokens that can be generated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::validation::ValidationConfig;
    ///
    /// let config = ValidationConfig::default();
    /// assert_eq!(config.max_new_tokens(), 4096); // 8192 - 4096
    /// ```
    pub const fn max_new_tokens(&self) -> usize {
        self.max_total_tokens.saturating_sub(self.max_input_tokens)
    }
}

/// Builder for `ValidationConfig`.
#[derive(Debug, Default)]
pub struct ValidationConfigBuilder {
    max_input_tokens: Option<usize>,
    max_total_tokens: Option<usize>,
    max_stop_sequences: Option<usize>,
    max_stop_sequence_length: Option<usize>,
    max_batch_size: Option<usize>,
    allow_empty_input: Option<bool>,
}

impl ValidationConfigBuilder {
    /// Set maximum input tokens.
    pub const fn max_input_tokens(mut self, value: usize) -> Self {
        self.max_input_tokens = Some(value);
        self
    }

    /// Set maximum total tokens.
    pub const fn max_total_tokens(mut self, value: usize) -> Self {
        self.max_total_tokens = Some(value);
        self
    }

    /// Set maximum stop sequences.
    pub const fn max_stop_sequences(mut self, value: usize) -> Self {
        self.max_stop_sequences = Some(value);
        self
    }

    /// Set maximum stop sequence length.
    pub const fn max_stop_sequence_length(mut self, value: usize) -> Self {
        self.max_stop_sequence_length = Some(value);
        self
    }

    /// Set maximum batch size.
    pub const fn max_batch_size(mut self, value: usize) -> Self {
        self.max_batch_size = Some(value);
        self
    }

    /// Set whether empty input is allowed.
    pub const fn allow_empty_input(mut self, value: bool) -> Self {
        self.allow_empty_input = Some(value);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ValidationConfig {
        let default = ValidationConfig::default();
        ValidationConfig {
            max_input_tokens: self.max_input_tokens.unwrap_or(default.max_input_tokens),
            max_total_tokens: self.max_total_tokens.unwrap_or(default.max_total_tokens),
            max_stop_sequences: self
                .max_stop_sequences
                .unwrap_or(default.max_stop_sequences),
            max_stop_sequence_length: self
                .max_stop_sequence_length
                .unwrap_or(default.max_stop_sequence_length),
            max_batch_size: self.max_batch_size.unwrap_or(default.max_batch_size),
            allow_empty_input: self.allow_empty_input.unwrap_or(default.allow_empty_input),
        }
    }
}

/// Generate request structure.
///
/// # TGI Source
///
/// Maps to `GenerateRequest` in `router/src/lib.rs`.
#[derive(Debug, Clone, Default)]
pub struct GenerateRequest {
    /// Input text to generate from.
    pub inputs: String,

    /// Maximum new tokens to generate.
    pub max_new_tokens: Option<usize>,

    /// Sampling temperature (0.0 - 2.0).
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling.
    pub top_p: Option<f32>,

    /// Top-k sampling.
    pub top_k: Option<usize>,

    /// Repetition penalty.
    pub repetition_penalty: Option<f32>,

    /// Stop sequences.
    pub stop: Vec<String>,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Whether to return log probabilities.
    pub return_logprobs: bool,

    /// Number of log probs to return per token.
    pub top_logprobs: Option<usize>,
}

impl GenerateRequest {
    /// Create a simple request with just input text.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::validation::GenerateRequest;
    ///
    /// let request = GenerateRequest::simple("Hello, world!");
    /// assert_eq!(request.inputs, "Hello, world!");
    /// ```
    pub fn simple(inputs: impl Into<String>) -> Self {
        Self {
            inputs: inputs.into(),
            ..Default::default()
        }
    }

    /// Estimated input token count (approximate).
    ///
    /// Uses simple whitespace tokenization as approximation.
    /// For accurate counts, use a proper tokenizer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::validation::GenerateRequest;
    ///
    /// let request = GenerateRequest::simple("Hello world");
    /// // ~11 chars / 4 = ~3 tokens (approximate)
    /// assert!(request.estimated_input_tokens() >= 2);
    /// assert!(request.estimated_input_tokens() <= 4);
    /// ```
    pub fn estimated_input_tokens(&self) -> usize {
        // Simple approximation: ~4 chars per token on average
        // TGI uses actual tokenizer, but this is good for validation
        (self.inputs.len() + 3) / 4
    }
}

/// Request validator.
///
/// Validates incoming requests against configuration constraints.
///
/// # TGI Source
///
/// Maps to `Validation` struct in `router/src/validation.rs`.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::validation::{RequestValidator, ValidationConfig, GenerateRequest};
///
/// let validator = RequestValidator::new(ValidationConfig::default());
///
/// let request = GenerateRequest::simple("Test input");
/// assert!(validator.validate(&request).is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct RequestValidator {
    config: ValidationConfig,
}

impl RequestValidator {
    /// Create a new validator with the given configuration.
    pub const fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Create a validator with default configuration.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::validation::RequestValidator;
    ///
    /// let validator = RequestValidator::default_validator();
    /// ```
    pub fn default_validator() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Get the configuration.
    pub const fn config(&self) -> &ValidationConfig {
        &self.config
    }

    /// Validate a generate request.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` if:
    /// - Input is empty and `allow_empty_input` is false
    /// - Input exceeds `max_input_tokens`
    /// - Total tokens would exceed `max_total_tokens`
    /// - Too many stop sequences
    /// - Stop sequence too long
    /// - Temperature out of range
    /// - Top-p out of range
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::validation::{RequestValidator, ValidationConfig, GenerateRequest};
    ///
    /// let validator = RequestValidator::new(ValidationConfig::default());
    ///
    /// // Valid request
    /// let request = GenerateRequest::simple("Hello");
    /// assert!(validator.validate(&request).is_ok());
    ///
    /// // Empty input (invalid by default)
    /// let empty = GenerateRequest::simple("");
    /// assert!(validator.validate(&empty).is_err());
    /// ```
    pub fn validate(&self, request: &GenerateRequest) -> Result<ValidatedRequest> {
        // Check empty input
        if request.inputs.is_empty() && !self.config.allow_empty_input {
            return Err(Error::validation("input cannot be empty"));
        }

        // Estimate token count
        let input_tokens = request.estimated_input_tokens();
        if input_tokens > self.config.max_input_tokens {
            return Err(Error::validation(format!(
                "input tokens ({input_tokens}) exceeds maximum ({})",
                self.config.max_input_tokens
            )));
        }

        // Check total tokens
        let max_new = request
            .max_new_tokens
            .unwrap_or(self.config.max_new_tokens());
        let total_tokens = input_tokens + max_new;
        if total_tokens > self.config.max_total_tokens {
            return Err(Error::validation(format!(
                "total tokens ({total_tokens}) exceeds maximum ({})",
                self.config.max_total_tokens
            )));
        }

        // Validate stop sequences
        if request.stop.len() > self.config.max_stop_sequences {
            return Err(Error::validation(format!(
                "too many stop sequences ({} > {})",
                request.stop.len(),
                self.config.max_stop_sequences
            )));
        }

        for stop in &request.stop {
            if stop.len() > self.config.max_stop_sequence_length {
                return Err(Error::validation(format!(
                    "stop sequence too long ({} > {})",
                    stop.len(),
                    self.config.max_stop_sequence_length
                )));
            }
        }

        // Validate temperature
        if let Some(temp) = request.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(Error::validation(format!(
                    "temperature ({temp}) must be between 0.0 and 2.0"
                )));
            }
        }

        // Validate top_p
        if let Some(top_p) = request.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err(Error::validation(format!(
                    "top_p ({top_p}) must be between 0.0 and 1.0"
                )));
            }
        }

        // Validate repetition penalty
        if let Some(penalty) = request.repetition_penalty {
            if penalty <= 0.0 {
                return Err(Error::validation(format!(
                    "repetition_penalty ({penalty}) must be positive"
                )));
            }
        }

        // Validate logprobs
        if let Some(top_logprobs) = request.top_logprobs {
            if top_logprobs > 20 {
                return Err(Error::validation(format!(
                    "top_logprobs ({top_logprobs}) must be <= 20"
                )));
            }
        }

        Ok(ValidatedRequest {
            inputs: request.inputs.clone(),
            input_tokens,
            max_new_tokens: max_new,
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k,
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            stop: request.stop.clone(),
            seed: request.seed,
            return_logprobs: request.return_logprobs,
            top_logprobs: request.top_logprobs,
        })
    }

    /// Validate a batch of requests.
    ///
    /// # Errors
    ///
    /// Returns error if batch size exceeds maximum or any request is invalid.
    pub fn validate_batch(&self, requests: &[GenerateRequest]) -> Result<Vec<ValidatedRequest>> {
        if requests.len() > self.config.max_batch_size {
            return Err(Error::validation(format!(
                "batch size ({}) exceeds maximum ({})",
                requests.len(),
                self.config.max_batch_size
            )));
        }

        requests.iter().map(|r| self.validate(r)).collect()
    }
}

/// A validated request ready for inference.
///
/// All fields have been validated and defaults applied.
#[derive(Debug, Clone)]
pub struct ValidatedRequest {
    /// Input text.
    pub inputs: String,

    /// Estimated input token count.
    pub input_tokens: usize,

    /// Maximum new tokens to generate.
    pub max_new_tokens: usize,

    /// Sampling temperature.
    pub temperature: f32,

    /// Top-p sampling value.
    pub top_p: f32,

    /// Top-k sampling value (if set).
    pub top_k: Option<usize>,

    /// Repetition penalty.
    pub repetition_penalty: f32,

    /// Stop sequences.
    pub stop: Vec<String>,

    /// Random seed.
    pub seed: Option<u64>,

    /// Whether to return log probabilities.
    pub return_logprobs: bool,

    /// Number of top log probs to return.
    pub top_logprobs: Option<usize>,
}

impl ValidatedRequest {
    /// Total tokens (input + max new).
    pub const fn total_tokens(&self) -> usize {
        self.input_tokens + self.max_new_tokens
    }

    /// Whether this request uses sampling (temperature > 0).
    pub fn uses_sampling(&self) -> bool {
        self.temperature > 0.0
    }

    /// Whether this request is deterministic.
    pub fn is_deterministic(&self) -> bool {
        self.seed.is_some() || self.temperature == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config.max_input_tokens, 4096);
        assert_eq!(config.max_total_tokens, 8192);
        assert_eq!(config.max_new_tokens(), 4096);
    }

    #[test]
    fn test_validation_config_builder() {
        let config = ValidationConfig::builder()
            .max_input_tokens(2048)
            .max_total_tokens(4096)
            .max_stop_sequences(8)
            .build();

        assert_eq!(config.max_input_tokens, 2048);
        assert_eq!(config.max_total_tokens, 4096);
        assert_eq!(config.max_stop_sequences, 8);
    }

    #[test]
    fn test_generate_request_simple() {
        let request = GenerateRequest::simple("Hello, world!");
        assert_eq!(request.inputs, "Hello, world!");
        assert!(request.max_new_tokens.is_none());
    }

    #[test]
    fn test_estimated_tokens() {
        let request = GenerateRequest::simple("Hello world test");
        // ~16 chars / 4 = 4 tokens
        assert!(request.estimated_input_tokens() >= 3);
        assert!(request.estimated_input_tokens() <= 5);
    }

    #[test]
    fn test_validator_valid_request() {
        let validator = RequestValidator::default_validator();
        let request = GenerateRequest::simple("Hello");
        assert!(validator.validate(&request).is_ok());
    }

    #[test]
    fn test_validator_empty_input() {
        let validator = RequestValidator::default_validator();
        let request = GenerateRequest::simple("");
        let result = validator.validate(&request);
        assert!(matches!(result, Err(Error::Validation(_))));
    }

    #[test]
    fn test_validator_empty_input_allowed() {
        let config = ValidationConfig::builder().allow_empty_input(true).build();
        let validator = RequestValidator::new(config);
        let request = GenerateRequest::simple("");
        assert!(validator.validate(&request).is_ok());
    }

    #[test]
    fn test_validator_temperature_range() {
        let validator = RequestValidator::default_validator();

        // Valid temperature
        let mut request = GenerateRequest::simple("Hello");
        request.temperature = Some(0.7);
        assert!(validator.validate(&request).is_ok());

        // Invalid temperature (too high)
        request.temperature = Some(3.0);
        assert!(validator.validate(&request).is_err());

        // Invalid temperature (negative)
        request.temperature = Some(-0.5);
        assert!(validator.validate(&request).is_err());
    }

    #[test]
    fn test_validator_top_p_range() {
        let validator = RequestValidator::default_validator();

        let mut request = GenerateRequest::simple("Hello");
        request.top_p = Some(0.9);
        assert!(validator.validate(&request).is_ok());

        request.top_p = Some(1.5);
        assert!(validator.validate(&request).is_err());
    }

    #[test]
    fn test_validator_stop_sequences() {
        let config = ValidationConfig::builder().max_stop_sequences(2).build();
        let validator = RequestValidator::new(config);

        let mut request = GenerateRequest::simple("Hello");
        request.stop = vec!["###".to_string(), "END".to_string()];
        assert!(validator.validate(&request).is_ok());

        request.stop.push("STOP".to_string());
        assert!(validator.validate(&request).is_err());
    }

    #[test]
    fn test_validator_stop_sequence_length() {
        let config = ValidationConfig::builder()
            .max_stop_sequence_length(5)
            .build();
        let validator = RequestValidator::new(config);

        let mut request = GenerateRequest::simple("Hello");
        request.stop = vec!["###".to_string()];
        assert!(validator.validate(&request).is_ok());

        request.stop = vec!["This is way too long".to_string()];
        assert!(validator.validate(&request).is_err());
    }

    #[test]
    fn test_validated_request_properties() {
        let validator = RequestValidator::default_validator();
        let mut request = GenerateRequest::simple("Hello");
        request.max_new_tokens = Some(100);
        request.temperature = Some(0.0);
        request.seed = Some(42);

        let validated = validator.validate(&request).unwrap();
        assert_eq!(validated.max_new_tokens, 100);
        assert!(!validated.uses_sampling());
        assert!(validated.is_deterministic());
    }

    #[test]
    fn test_validate_batch() {
        let config = ValidationConfig::builder().max_batch_size(2).build();
        let validator = RequestValidator::new(config);

        let requests = vec![
            GenerateRequest::simple("Hello"),
            GenerateRequest::simple("World"),
        ];
        assert!(validator.validate_batch(&requests).is_ok());

        let too_many = vec![
            GenerateRequest::simple("A"),
            GenerateRequest::simple("B"),
            GenerateRequest::simple("C"),
        ];
        assert!(validator.validate_batch(&too_many).is_err());
    }

    #[test]
    fn test_repetition_penalty_validation() {
        let validator = RequestValidator::default_validator();

        let mut request = GenerateRequest::simple("Hello");
        request.repetition_penalty = Some(1.2);
        assert!(validator.validate(&request).is_ok());

        request.repetition_penalty = Some(0.0);
        assert!(validator.validate(&request).is_err());

        request.repetition_penalty = Some(-1.0);
        assert!(validator.validate(&request).is_err());
    }

    #[test]
    fn test_top_logprobs_validation() {
        let validator = RequestValidator::default_validator();

        let mut request = GenerateRequest::simple("Hello");
        request.top_logprobs = Some(5);
        assert!(validator.validate(&request).is_ok());

        request.top_logprobs = Some(25);
        assert!(validator.validate(&request).is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_valid_temperature_accepted(temp in 0.0f32..=2.0) {
            let validator = RequestValidator::default_validator();
            let mut request = GenerateRequest::simple("Hello");
            request.temperature = Some(temp);
            prop_assert!(validator.validate(&request).is_ok());
        }

        #[test]
        fn prop_invalid_temperature_rejected(temp in 2.01f32..100.0) {
            let validator = RequestValidator::default_validator();
            let mut request = GenerateRequest::simple("Hello");
            request.temperature = Some(temp);
            prop_assert!(validator.validate(&request).is_err());
        }

        #[test]
        fn prop_valid_top_p_accepted(top_p in 0.0f32..=1.0) {
            let validator = RequestValidator::default_validator();
            let mut request = GenerateRequest::simple("Hello");
            request.top_p = Some(top_p);
            prop_assert!(validator.validate(&request).is_ok());
        }

        #[test]
        fn prop_any_nonempty_input_validates(input in ".+") {
            let validator = RequestValidator::default_validator();
            let request = GenerateRequest::simple(input);
            // Should at least not panic, may fail on length
            let _ = validator.validate(&request);
        }

        #[test]
        fn prop_estimated_tokens_proportional(len in 1usize..10000) {
            let input: String = "x".repeat(len);
            let request = GenerateRequest::simple(input);
            let tokens = request.estimated_input_tokens();
            // Should be roughly len/4
            prop_assert!(tokens >= len / 5);
            prop_assert!(tokens <= len / 3 + 1);
        }
    }
}
