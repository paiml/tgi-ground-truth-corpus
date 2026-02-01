# Inference Backend Pattern

Backend initialization and model loading.

## TGI Source

Derived from `backends/v3/src/backend.rs`:
- Backend initialization
- Model loading
- Inference execution

## Sovereign AI Stack Equivalent

Maps to `realizar::inference` for model inference.

## Key Components

### DataType

Supported inference data types:

```rust
use tgi_gtc::inference::DataType;

enum DataType {
    Float32,  // 4 bytes, full precision
    Float16,  // 2 bytes, half precision
    BFloat16, // 2 bytes, brain float
    Int8,     // 1 byte, quantized
}

// Get bytes per element
DataType::Float16.bytes(); // 2

// Get name
DataType::Float16.name(); // "float16"
```

### BackendConfig

```rust
use tgi_gtc::inference::BackendConfig;

let config = BackendConfig::new("meta-llama/Llama-2-7b-hf")
    .device("cuda:0")
    .dtype(DataType::Float16)
    .max_sequence_length(4096)
    .flash_attention(true);

// Validate before use
config.validate()?;
```

### BackendState

State machine for backend lifecycle:

```rust
use tgi_gtc::inference::BackendState;

enum BackendState {
    Uninitialized, // Not started
    Loading,       // Model loading
    Ready,         // Ready for inference
    Error,         // Error state
}

state.is_ready();  // true if Ready
state.is_error();  // true if Error
```

### InferenceBackend

```rust
use tgi_gtc::inference::InferenceBackend;

// Create backend
let mut backend = InferenceBackend::new(config);
// or
let mut backend = InferenceBackend::with_model("gpt2");

// Check state
backend.state();    // BackendState
backend.is_ready(); // bool

// Initialize (load model)
backend.initialize()?;

// Handle errors
backend.set_error();

// Reset to uninitialized
backend.reset();

// Access config
backend.config().model_id;
```

## Lifecycle

```
┌──────────────┐
│ Uninitialized│
└──────┬───────┘
       │ initialize()
       ▼
┌──────────────┐
│   Loading    │
└──────┬───────┘
       │ success
       ▼
┌──────────────┐     set_error()     ┌──────────────┐
│    Ready     │────────────────────▶│    Error     │
└──────────────┘                     └──────┬───────┘
       ▲                                    │
       │              reset()               │
       └────────────────────────────────────┘
```

## Configuration Validation

The `validate()` method checks:

1. **model_id**: Must not be empty
2. **max_sequence_length**: Must be > 0

```rust
// Valid
BackendConfig::new("gpt2").validate()?; // Ok

// Invalid - empty model
BackendConfig::default().validate()?; // Err

// Invalid - zero sequence length
BackendConfig::new("gpt2")
    .max_sequence_length(0)
    .validate()?; // Err
```

## Device Selection

Common device strings:

| Device | Description |
|--------|-------------|
| `cpu` | CPU inference |
| `cuda:0` | First NVIDIA GPU |
| `cuda:1` | Second NVIDIA GPU |
| `mps` | Apple Metal (M1/M2) |

## Data Type Selection

| Type | Bytes | Use Case |
|------|-------|----------|
| `Float32` | 4 | Full precision, debugging |
| `Float16` | 2 | Production, most GPUs |
| `BFloat16` | 2 | A100/H100, better range |
| `Int8` | 1 | Quantized, memory limited |

## Example

```bash
cargo run --example inference_backend
```
