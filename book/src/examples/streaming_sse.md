# Streaming SSE Example

Demonstrates TGI's Server-Sent Events streaming for token-by-token delivery.

## Run

```bash
cargo run --example streaming_sse
```

## Output

```
=== SSE Streaming Demo ===

Formatter Configuration:
  Include token IDs: true
  Include logprobs: true

Simulated Token Stream:
============================================================
data: {"token":"Hello","token_id":1234,"logprob":-50}

data: {"token":",","token_id":44,"logprob":-10}

data: {"token":" world","token_id":5678,"logprob":-75}

data: {"token":"!","token_id":999,"logprob":-25}

data: {"token":"<|endoftext|>","token_id":0,"special":true}

data: {"generated_tokens":5,"finish_reason":"eos","generation_time_ms":125}

data: [DONE]

============================================================

Error Event Example:
----------------------------------------
data: {"error":"Model inference failed: out of memory"}

----------------------------------------

Finish Reasons:
  Length: length
  Stop: stop
  EOS: eos

=== Demo Complete ===
```

## Code Walkthrough

### 1. Create Formatter

```rust
let mut formatter = SseFormatter::new();
formatter.include_token_ids = true;
formatter.include_logprobs = true;
```

### 2. Format Token Events

```rust
let event = TokenEvent {
    token: "Hello".to_string(),
    token_id: 1234,
    logprob: Some(-50),
    special: false,
};

let sse = formatter.format_token(&event);
print!("{}", sse);
```

### 3. Format Completion

```rust
let complete = CompleteEvent {
    generated_tokens: 5,
    finish_reason: FinishReason::EndOfSequence,
    generation_time_ms: 125,
};

let sse = formatter.format_complete(&complete);
print!("{}", sse);
```
