# Request Validation Example

Demonstrates TGI's request validation patterns.

## Run

```bash
cargo run --example request_validation
```

## Output

```
=== Request Validation Demo ===

Validation Configuration:
  Max input tokens: 4096
  Max new tokens: 2048
  Max total tokens: 8192
  Max stop sequences: 4

Validation Results:
------------------------------------------------------------
Valid simple request: ✓ VALID
    Input tokens: ~7, Max new: 2048

Valid with parameters: ✓ VALID
    Input tokens: ~6, Max new: 500

Invalid temperature (too high): ✗ INVALID
    Error: validation error: temperature must be between 0.0 and 2.0

Invalid temperature (negative): ✗ INVALID
    Error: validation error: temperature must be between 0.0 and 2.0

Invalid top_p (too high): ✗ INVALID
    Error: validation error: top_p must be between 0.0 and 1.0

Empty input (not allowed): ✗ INVALID
    Error: validation error: empty input not allowed

Too many stop sequences: ✗ INVALID
    Error: validation error: too many stop sequences (max 4)


Batch Validation:
------------------------------------------------------------
✓ Batch valid: 3 requests
    Request 1: ~2 input tokens
    Request 2: ~2 input tokens
    Request 3: ~2 input tokens

=== Demo Complete ===
```

## Code Walkthrough

### 1. Configure Validator

```rust
let config = ValidationConfig::builder()
    .max_input_tokens(4096)
    .max_new_tokens(2048)
    .max_total_tokens(8192)
    .max_stop_sequences(4)
    .allow_empty_input(false)
    .build();

let validator = RequestValidator::new(config);
```

### 2. Create Requests

```rust
// Simple request
let request = GenerateRequest::simple("What is AI?");

// With parameters
let request = GenerateRequest::new("Explain quantum")
    .max_new_tokens(500)
    .temperature(0.7)
    .top_p(0.9);
```

### 3. Validate

```rust
match validator.validate(&request) {
    Ok(validated) => {
        println!("Input tokens: {}", validated.estimated_input_tokens());
    }
    Err(e) => {
        println!("Invalid: {}", e);
    }
}
```
