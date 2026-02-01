# Inference Backend Example

Demonstrates TGI's backend initialization patterns.

## Run

```bash
cargo run --example inference_backend
```

## Output

```
=== Inference Backend Demo ===

Backend Configuration:
  Model: meta-llama/Llama-2-7b-hf
  Device: cuda:0
  DType: float16 (2 bytes/element)
  Max sequence length: 4096
  Flash attention: true

Backend State Transitions:
  Initial: Uninitialized
  Initializing...
  After init: Ready
  Simulating error...
  After error: Error
  Resetting...
  After reset: Uninitialized

Supported Data Types:
----------------------------------------
  float32    - 4 bytes/element
  float16    - 2 bytes/element
  bfloat16   - 2 bytes/element
  int8       - 1 bytes/element

Configuration Validation:
----------------------------------------
  Valid config: Ok(())
  Empty model: Err(config error: model_id is required)
  Zero seq len: Err(config error: max_sequence_length must be > 0)

=== Demo Complete ===
```

## Code Walkthrough

### 1. Configure Backend

```rust
let config = BackendConfig::new("meta-llama/Llama-2-7b-hf")
    .device("cuda:0")
    .dtype(DataType::Float16)
    .max_sequence_length(4096)
    .flash_attention(true);
```

### 2. Create and Initialize

```rust
let mut backend = InferenceBackend::new(config);
backend.initialize()?;
assert!(backend.is_ready());
```

### 3. Handle State Changes

```rust
// Error state
backend.set_error();
assert!(backend.state().is_error());

// Reset
backend.reset();
assert_eq!(backend.state(), BackendState::Uninitialized);
```
