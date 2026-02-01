//! Inference Backend Example
//!
//! Demonstrates TGI's backend initialization patterns for
//! model loading and inference state management.
//!
//! # Run
//!
//! ```bash
//! cargo run --example inference_backend
//! ```

use tgi_gtc::inference::{BackendConfig, BackendState, DataType, InferenceBackend};

fn main() {
    println!("=== Inference Backend Demo ===\n");

    // Configure backend with TGI-style settings
    let config = BackendConfig::new("meta-llama/Llama-2-7b-hf")
        .device("cuda:0")
        .dtype(DataType::Float16)
        .max_sequence_length(4096)
        .flash_attention(true);

    println!("Backend Configuration:");
    println!("  Model: {}", config.model_id);
    println!("  Device: {}", config.device);
    println!(
        "  DType: {} ({} bytes/element)",
        config.dtype.name(),
        config.dtype.bytes()
    );
    println!("  Max sequence length: {}", config.max_sequence_length);
    println!("  Flash attention: {}", config.use_flash_attention);
    println!();

    // Create backend
    let mut backend = InferenceBackend::new(config);

    println!("Backend State Transitions:");
    println!("  Initial: {:?}", backend.state());
    assert_eq!(backend.state(), BackendState::Uninitialized);

    // Initialize (load model)
    println!("  Initializing...");
    match backend.initialize() {
        Ok(()) => {
            println!("  After init: {:?}", backend.state());
            assert!(backend.is_ready());
        }
        Err(e) => {
            println!("  Init failed: {}", e);
        }
    }

    // Simulate error
    println!("  Simulating error...");
    backend.set_error();
    println!("  After error: {:?}", backend.state());
    assert!(backend.state().is_error());

    // Reset
    println!("  Resetting...");
    backend.reset();
    println!("  After reset: {:?}", backend.state());
    assert_eq!(backend.state(), BackendState::Uninitialized);

    // Show data types
    println!("\nSupported Data Types:");
    println!("{}", "-".repeat(40));
    let dtypes = [
        DataType::Float32,
        DataType::Float16,
        DataType::BFloat16,
        DataType::Int8,
    ];

    for dtype in dtypes {
        println!("  {:10} - {} bytes/element", dtype.name(), dtype.bytes());
    }

    // Validation example
    println!("\nConfiguration Validation:");
    println!("{}", "-".repeat(40));

    let valid_config = BackendConfig::new("gpt2");
    println!("  Valid config: {:?}", valid_config.validate());

    let invalid_config = BackendConfig::default(); // Empty model_id
    println!("  Empty model: {:?}", invalid_config.validate());

    let zero_seq = BackendConfig::new("gpt2").max_sequence_length(0);
    println!("  Zero seq len: {:?}", zero_seq.validate());

    println!("\n=== Demo Complete ===");
}
