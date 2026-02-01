//! Streaming response patterns.
//!
//! Server-Sent Events (SSE) streaming patterns for token-by-token delivery.
//!
//! # TGI Source
//!
//! Patterns derived from `router/src/server.rs`:
//! - SSE event formatting
//! - Chunked token delivery
//! - Backpressure handling
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `realizar::stream` for streaming inference.

/// Streaming event types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// Token generated.
    Token(TokenEvent),

    /// Generation complete.
    Complete(CompleteEvent),

    /// Error occurred.
    Error(String),
}

/// Token generation event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenEvent {
    /// Generated token text.
    pub token: String,

    /// Token ID.
    pub token_id: u32,

    /// Log probability (if requested).
    pub logprob: Option<i32>, // Fixed-point representation

    /// Whether this is a special token.
    pub special: bool,
}

/// Generation complete event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompleteEvent {
    /// Total tokens generated.
    pub generated_tokens: usize,

    /// Finish reason.
    pub finish_reason: FinishReason,

    /// Generation time in milliseconds.
    pub generation_time_ms: u64,
}

/// Reason for generation completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Reached max tokens.
    Length,

    /// Hit stop sequence.
    Stop,

    /// Hit EOS token.
    EndOfSequence,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Length => write!(f, "length"),
            Self::Stop => write!(f, "stop"),
            Self::EndOfSequence => write!(f, "eos"),
        }
    }
}

/// SSE formatter for streaming responses.
///
/// # TGI Source
///
/// Maps to SSE formatting in `router/src/server.rs`.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::streaming::{SseFormatter, TokenEvent};
///
/// let formatter = SseFormatter::new();
/// let event = TokenEvent {
///     token: "Hello".to_string(),
///     token_id: 1234,
///     logprob: None,
///     special: false,
/// };
///
/// let sse = formatter.format_token(&event);
/// assert!(sse.starts_with("data: "));
/// ```
#[derive(Debug, Default)]
pub struct SseFormatter {
    /// Include token IDs in output.
    pub include_token_ids: bool,

    /// Include logprobs in output.
    pub include_logprobs: bool,
}

impl SseFormatter {
    /// Create a new formatter.
    pub const fn new() -> Self {
        Self {
            include_token_ids: false,
            include_logprobs: false,
        }
    }

    /// Format a token event as SSE.
    pub fn format_token(&self, event: &TokenEvent) -> String {
        let mut data = format!(r#"{{"token":"{}""#, escape_json(&event.token));

        if self.include_token_ids {
            data.push_str(&format!(r#","token_id":{}"#, event.token_id));
        }

        if self.include_logprobs {
            if let Some(logprob) = event.logprob {
                data.push_str(&format!(r#","logprob":{}"#, logprob));
            }
        }

        if event.special {
            data.push_str(r#","special":true"#);
        }

        data.push('}');

        format!("data: {data}\n\n")
    }

    /// Format a complete event as SSE.
    pub fn format_complete(&self, event: &CompleteEvent) -> String {
        let data = format!(
            r#"{{"generated_tokens":{},"finish_reason":"{}","generation_time_ms":{}}}"#,
            event.generated_tokens, event.finish_reason, event.generation_time_ms
        );

        format!("data: {data}\n\ndata: [DONE]\n\n")
    }

    /// Format an error as SSE.
    pub fn format_error(&self, message: &str) -> String {
        let data = format!(r#"{{"error":"{}"}}"#, escape_json(message));
        format!("data: {data}\n\n")
    }
}

/// Escape string for JSON.
fn escape_json(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_event() {
        let event = TokenEvent {
            token: "Hello".to_string(),
            token_id: 1234,
            logprob: Some(-100),
            special: false,
        };

        assert_eq!(event.token, "Hello");
        assert_eq!(event.token_id, 1234);
    }

    #[test]
    fn test_finish_reason_display() {
        assert_eq!(FinishReason::Length.to_string(), "length");
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::EndOfSequence.to_string(), "eos");
    }

    #[test]
    fn test_sse_formatter_token() {
        let formatter = SseFormatter::new();
        let event = TokenEvent {
            token: "Hello".to_string(),
            token_id: 1234,
            logprob: None,
            special: false,
        };

        let sse = formatter.format_token(&event);
        assert!(sse.starts_with("data: "));
        assert!(sse.contains(r#""token":"Hello""#));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_sse_formatter_with_token_id() {
        let mut formatter = SseFormatter::new();
        formatter.include_token_ids = true;

        let event = TokenEvent {
            token: "Hi".to_string(),
            token_id: 42,
            logprob: None,
            special: false,
        };

        let sse = formatter.format_token(&event);
        assert!(sse.contains(r#""token_id":42"#));
    }

    #[test]
    fn test_sse_formatter_complete() {
        let formatter = SseFormatter::new();
        let event = CompleteEvent {
            generated_tokens: 100,
            finish_reason: FinishReason::Stop,
            generation_time_ms: 1500,
        };

        let sse = formatter.format_complete(&event);
        assert!(sse.contains(r#""generated_tokens":100"#));
        assert!(sse.contains(r#""finish_reason":"stop""#));
        assert!(sse.contains("[DONE]"));
    }

    #[test]
    fn test_sse_formatter_error() {
        let formatter = SseFormatter::new();
        let sse = formatter.format_error("Something went wrong");
        assert!(sse.contains(r#""error":"Something went wrong""#));
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("hello\"world"), "hello\\\"world");
        assert_eq!(escape_json("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_json("tab\there"), "tab\\there");
    }

    #[test]
    fn test_special_token() {
        let formatter = SseFormatter::new();
        let event = TokenEvent {
            token: "<|endoftext|>".to_string(),
            token_id: 0,
            logprob: None,
            special: true,
        };

        let sse = formatter.format_token(&event);
        assert!(sse.contains(r#""special":true"#));
    }
}
