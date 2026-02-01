//! Error types for TGI Ground Truth Corpus.
//!
//! Provides a unified error type hierarchy following TGI's error handling
//! patterns, mapped to Sovereign AI Stack conventions.
//!
//! # TGI Source
//!
//! Error patterns derived from:
//! - `router/src/lib.rs` - Router errors
//! - `backends/v3/src/lib.rs` - Backend errors
//!
//! # Examples
//!
//! ```rust
//! use tgi_gtc::{Error, Result};
//!
//! fn validate_tokens(count: usize) -> Result<()> {
//!     if count > 4096 {
//!         return Err(Error::Validation(
//!             "token count exceeds maximum".into()
//!         ));
//!     }
//!     Ok(())
//! }
//! ```

use std::fmt;

/// Main error type for the corpus.
///
/// Covers all error categories from TGI's router and backend layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Request validation failed.
    ///
    /// Maps to TGI's `ValidationError`.
    Validation(String),

    /// Batch processing error.
    ///
    /// Maps to TGI's queue/batch errors.
    Batching(String),

    /// Inference backend error.
    ///
    /// Maps to TGI's backend errors.
    Inference(String),

    /// Streaming/SSE error.
    Streaming(String),

    /// Scheduling/queue error.
    Scheduling(String),

    /// Quantization error.
    Quantization(String),

    /// Configuration error.
    Config(String),

    /// Resource exhaustion (memory, queue full, etc.).
    ResourceExhausted(String),

    /// Request timeout.
    Timeout(String),

    /// Request cancelled by client.
    Cancelled(String),

    /// Internal error.
    Internal(String),
}

impl Error {
    /// Create a validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::Error;
    ///
    /// let err = Error::validation("input too long");
    /// assert!(matches!(err, Error::Validation(_)));
    /// ```
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    /// Create a batching error.
    pub fn batching(msg: impl Into<String>) -> Self {
        Self::Batching(msg.into())
    }

    /// Create an inference error.
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }

    /// Create a streaming error.
    pub fn streaming(msg: impl Into<String>) -> Self {
        Self::Streaming(msg.into())
    }

    /// Create a scheduling error.
    pub fn scheduling(msg: impl Into<String>) -> Self {
        Self::Scheduling(msg.into())
    }

    /// Create a quantization error.
    pub fn quantization(msg: impl Into<String>) -> Self {
        Self::Quantization(msg.into())
    }

    /// Create a config error.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a resource exhausted error.
    pub fn resource_exhausted(msg: impl Into<String>) -> Self {
        Self::ResourceExhausted(msg.into())
    }

    /// Create a timeout error.
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }

    /// Create a cancelled error.
    pub fn cancelled(msg: impl Into<String>) -> Self {
        Self::Cancelled(msg.into())
    }

    /// Create an internal error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Check if error is retryable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::Error;
    ///
    /// let timeout = Error::timeout("request timed out");
    /// assert!(timeout.is_retryable());
    ///
    /// let validation = Error::validation("bad input");
    /// assert!(!validation.is_retryable());
    /// ```
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_) | Self::ResourceExhausted(_) | Self::Scheduling(_)
        )
    }

    /// Check if error is a client error (4xx equivalent).
    pub const fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::Validation(_) | Self::Cancelled(_) | Self::Config(_)
        )
    }

    /// Check if error is a server error (5xx equivalent).
    pub const fn is_server_error(&self) -> bool {
        matches!(
            self,
            Self::Inference(_)
                | Self::Internal(_)
                | Self::Batching(_)
                | Self::Streaming(_)
                | Self::Quantization(_)
        )
    }

    /// Get error code for metrics/logging.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::Error;
    ///
    /// let err = Error::validation("bad input");
    /// assert_eq!(err.code(), "VALIDATION");
    /// ```
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Validation(_) => "VALIDATION",
            Self::Batching(_) => "BATCHING",
            Self::Inference(_) => "INFERENCE",
            Self::Streaming(_) => "STREAMING",
            Self::Scheduling(_) => "SCHEDULING",
            Self::Quantization(_) => "QUANTIZATION",
            Self::Config(_) => "CONFIG",
            Self::ResourceExhausted(_) => "RESOURCE_EXHAUSTED",
            Self::Timeout(_) => "TIMEOUT",
            Self::Cancelled(_) => "CANCELLED",
            Self::Internal(_) => "INTERNAL",
        }
    }

    /// Get HTTP status code equivalent.
    pub const fn http_status(&self) -> u16 {
        match self {
            Self::Validation(_) => 400,
            Self::Config(_) => 400,
            Self::Cancelled(_) => 499,
            Self::Timeout(_) => 408,
            Self::ResourceExhausted(_) => 429,
            Self::Inference(_)
            | Self::Batching(_)
            | Self::Streaming(_)
            | Self::Scheduling(_)
            | Self::Quantization(_)
            | Self::Internal(_) => 500,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::Batching(msg) => write!(f, "batching error: {msg}"),
            Self::Inference(msg) => write!(f, "inference error: {msg}"),
            Self::Streaming(msg) => write!(f, "streaming error: {msg}"),
            Self::Scheduling(msg) => write!(f, "scheduling error: {msg}"),
            Self::Quantization(msg) => write!(f, "quantization error: {msg}"),
            Self::Config(msg) => write!(f, "config error: {msg}"),
            Self::ResourceExhausted(msg) => write!(f, "resource exhausted: {msg}"),
            Self::Timeout(msg) => write!(f, "timeout: {msg}"),
            Self::Cancelled(msg) => write!(f, "cancelled: {msg}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type alias for corpus operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::validation("test");
        assert_eq!(err.code(), "VALIDATION");
        assert!(err.is_client_error());
        assert!(!err.is_server_error());
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_error_retryable() {
        assert!(Error::timeout("test").is_retryable());
        assert!(Error::resource_exhausted("test").is_retryable());
        assert!(Error::scheduling("test").is_retryable());
        assert!(!Error::validation("test").is_retryable());
        assert!(!Error::inference("test").is_retryable());
    }

    #[test]
    fn test_error_http_status() {
        assert_eq!(Error::validation("test").http_status(), 400);
        assert_eq!(Error::timeout("test").http_status(), 408);
        assert_eq!(Error::resource_exhausted("test").http_status(), 429);
        assert_eq!(Error::inference("test").http_status(), 500);
    }

    #[test]
    fn test_error_display() {
        let err = Error::validation("bad input");
        assert_eq!(format!("{err}"), "validation error: bad input");
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(Error::validation("").code(), "VALIDATION");
        assert_eq!(Error::batching("").code(), "BATCHING");
        assert_eq!(Error::inference("").code(), "INFERENCE");
        assert_eq!(Error::streaming("").code(), "STREAMING");
        assert_eq!(Error::scheduling("").code(), "SCHEDULING");
        assert_eq!(Error::quantization("").code(), "QUANTIZATION");
        assert_eq!(Error::config("").code(), "CONFIG");
        assert_eq!(Error::resource_exhausted("").code(), "RESOURCE_EXHAUSTED");
        assert_eq!(Error::timeout("").code(), "TIMEOUT");
        assert_eq!(Error::cancelled("").code(), "CANCELLED");
        assert_eq!(Error::internal("").code(), "INTERNAL");
    }

    #[test]
    fn test_error_client_vs_server() {
        // Client errors
        assert!(Error::validation("").is_client_error());
        assert!(Error::config("").is_client_error());
        assert!(Error::cancelled("").is_client_error());

        // Server errors
        assert!(Error::inference("").is_server_error());
        assert!(Error::batching("").is_server_error());
        assert!(Error::streaming("").is_server_error());
        assert!(Error::quantization("").is_server_error());
        assert!(Error::internal("").is_server_error());

        // Neither (resource/timeout)
        assert!(!Error::timeout("").is_client_error());
        assert!(!Error::timeout("").is_server_error());
    }

    #[test]
    fn test_error_display_all_variants() {
        // Test Display for all error variants
        assert_eq!(
            format!("{}", Error::validation("bad")),
            "validation error: bad"
        );
        assert_eq!(
            format!("{}", Error::batching("fail")),
            "batching error: fail"
        );
        assert_eq!(
            format!("{}", Error::inference("model")),
            "inference error: model"
        );
        assert_eq!(
            format!("{}", Error::streaming("sse")),
            "streaming error: sse"
        );
        assert_eq!(
            format!("{}", Error::scheduling("queue")),
            "scheduling error: queue"
        );
        assert_eq!(
            format!("{}", Error::quantization("bits")),
            "quantization error: bits"
        );
        assert_eq!(format!("{}", Error::config("cfg")), "config error: cfg");
        assert_eq!(
            format!("{}", Error::resource_exhausted("mem")),
            "resource exhausted: mem"
        );
        assert_eq!(format!("{}", Error::timeout("slow")), "timeout: slow");
        assert_eq!(format!("{}", Error::cancelled("user")), "cancelled: user");
        assert_eq!(
            format!("{}", Error::internal("panic")),
            "internal error: panic"
        );
    }

    #[test]
    fn test_error_http_status_all_variants() {
        // Client errors (4xx)
        assert_eq!(Error::validation("").http_status(), 400);
        assert_eq!(Error::config("").http_status(), 400);
        assert_eq!(Error::cancelled("").http_status(), 499);
        assert_eq!(Error::timeout("").http_status(), 408);
        assert_eq!(Error::resource_exhausted("").http_status(), 429);

        // Server errors (5xx)
        assert_eq!(Error::inference("").http_status(), 500);
        assert_eq!(Error::batching("").http_status(), 500);
        assert_eq!(Error::streaming("").http_status(), 500);
        assert_eq!(Error::scheduling("").http_status(), 500);
        assert_eq!(Error::quantization("").http_status(), 500);
        assert_eq!(Error::internal("").http_status(), 500);
    }

    #[test]
    fn test_error_scheduling_is_server_error() {
        // Scheduling is retryable but also a server-side error
        let err = Error::scheduling("queue full");
        assert!(err.is_retryable());
        assert!(!err.is_server_error()); // It's neither client nor server
        assert!(!err.is_client_error());
    }

    #[test]
    fn test_error_resource_exhausted_not_server_error() {
        let err = Error::resource_exhausted("memory");
        assert!(err.is_retryable());
        assert!(!err.is_server_error());
        assert!(!err.is_client_error());
    }

    #[test]
    fn test_error_std_error_impl() {
        let err = Error::validation("test");
        let std_err: &dyn std::error::Error = &err;
        // Just verify we can use it as a std::error::Error
        assert!(std_err.to_string().contains("validation"));
    }

    #[test]
    fn test_error_equality() {
        assert_eq!(Error::validation("a"), Error::validation("a"));
        assert_ne!(Error::validation("a"), Error::validation("b"));
        assert_ne!(Error::validation("a"), Error::inference("a"));
    }
}
