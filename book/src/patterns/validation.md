# Request Validation Pattern

Input validation and token counting for safe inference.

## TGI Source

Derived from `router/src/validation.rs`:
- Input length validation
- Parameter range checking
- Token estimation
- Stop sequence validation

## Sovereign AI Stack Equivalent

Maps to `realizar::validate` for request validation.

## Why Validation?

Invalid requests can:
- Exhaust GPU memory (too many tokens)
- Cause numerical issues (invalid temperature)
- Block other requests (endless generation)
- Expose security vulnerabilities

## Key Components

### ValidationConfig

```rust
use tgi_gtc::validation::ValidationConfig;

let config = ValidationConfig::builder()
    .max_input_tokens(4096)
    .max_new_tokens(2048)
    .max_total_tokens(8192)
    .max_stop_sequences(4)
    .max_stop_sequence_length(64)
    .allow_empty_input(false)
    .build();
```

### GenerateRequest

Request structure:

```rust
use tgi_gtc::validation::GenerateRequest;

// Simple request
let request = GenerateRequest::simple("What is AI?");

// With parameters
let request = GenerateRequest::new("Explain quantum computing")
    .max_new_tokens(500)
    .temperature(0.7)
    .top_p(0.9)
    .top_k(50)
    .repetition_penalty(1.1)
    .stop_sequences(vec!["###".to_string()]);
```

### RequestValidator

```rust
use tgi_gtc::validation::RequestValidator;

let validator = RequestValidator::new(config);

// Validate single request
match validator.validate(&request) {
    Ok(validated) => {
        println!("Input tokens: {}", validated.estimated_input_tokens());
        println!("Max new: {}", validated.max_new_tokens());
    }
    Err(e) => {
        println!("Invalid: {}", e);
    }
}

// Validate batch
let validated = validator.validate_batch(&requests)?;
```

### ValidatedRequest

Post-validation request:

```rust
let validated = validator.validate(&request)?;
validated.inputs()                // Original input
validated.estimated_input_tokens() // Token estimate
validated.max_new_tokens()        // Clamped max
validated.temperature()           // Validated temp
validated.top_p()                 // Validated top_p
```

## Validation Rules

### Temperature
- Range: `0.0 <= temp <= 2.0`
- Default: `1.0`
- `0.0` = deterministic
- `> 1.0` = more random

### Top-P (Nucleus Sampling)
- Range: `0.0 < top_p <= 1.0`
- Default: `1.0` (disabled)
- `0.9` = consider tokens until 90% probability mass

### Top-K
- Range: `0 < top_k` (0 = disabled)
- Default: `0` (disabled)
- `50` = consider only top 50 tokens

### Repetition Penalty
- Range: `0.0 < penalty`
- Default: `1.0` (disabled)
- `1.2` = reduce repeated token probability by 20%

### Token Limits
- `input_tokens <= max_input_tokens`
- `max_new_tokens <= config.max_new_tokens`
- `input_tokens + max_new_tokens <= max_total_tokens`

## Token Estimation

Simple character-based estimation:

```rust
// ~4 characters per token (rough estimate)
fn estimated_input_tokens(&self) -> usize {
    (self.inputs.len() + 3) / 4
}
```

For production, use a proper tokenizer.

## Example

```bash
cargo run --example request_validation
```
