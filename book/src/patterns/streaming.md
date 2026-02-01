# SSE Streaming Pattern

Server-Sent Events for token-by-token delivery.

## TGI Source

Derived from `router/src/server.rs`:
- SSE event formatting
- Chunked token delivery
- Backpressure handling

## Sovereign AI Stack Equivalent

Maps to `realizar::stream` for streaming inference.

## Why Streaming?

Without streaming:
- Wait for full generation (seconds to minutes)
- Poor user experience
- Memory for full response

With streaming:
- Tokens appear as generated
- Interactive feel
- Lower memory footprint

## Key Components

### StreamEvent

Event types in the stream:

```rust
use tgi_gtc::streaming::{StreamEvent, TokenEvent, CompleteEvent};

enum StreamEvent {
    Token(TokenEvent),      // Generated token
    Complete(CompleteEvent), // Generation done
    Error(String),          // Error occurred
}
```

### TokenEvent

Individual token data:

```rust
pub struct TokenEvent {
    pub token: String,     // Token text
    pub token_id: u32,     // Vocabulary ID
    pub logprob: Option<i32>, // Log probability
    pub special: bool,     // Is special token
}
```

### CompleteEvent

Generation completion:

```rust
pub struct CompleteEvent {
    pub generated_tokens: usize,
    pub finish_reason: FinishReason,
    pub generation_time_ms: u64,
}
```

### FinishReason

Why generation stopped:

```rust
enum FinishReason {
    Length,        // Hit max_new_tokens
    Stop,          // Hit stop sequence
    EndOfSequence, // Hit EOS token
}
```

### SseFormatter

Format events as SSE:

```rust
use tgi_gtc::streaming::SseFormatter;

let mut formatter = SseFormatter::new();
formatter.include_token_ids = true;
formatter.include_logprobs = true;

// Format token
let sse = formatter.format_token(&token_event);
// "data: {\"token\":\"Hello\",\"token_id\":1234}\n\n"

// Format completion
let sse = formatter.format_complete(&complete_event);
// "data: {...}\n\ndata: [DONE]\n\n"

// Format error
let sse = formatter.format_error("Out of memory");
// "data: {\"error\":\"Out of memory\"}\n\n"
```

## SSE Format

Server-Sent Events format:

```
data: {"token":"Hello"}\n
\n
data: {"token":" world"}\n
\n
data: {"generated_tokens":2,"finish_reason":"stop"}\n
\n
data: [DONE]\n
\n
```

Key rules:
- Each event starts with `data: `
- Events end with `\n\n` (double newline)
- JSON must be escaped (no raw newlines)
- `[DONE]` signals end of stream

## Example

```bash
cargo run --example streaming_sse
```
