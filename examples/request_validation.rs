//! Request Validation Example
//!
//! Demonstrates TGI's request validation patterns for ensuring
//! safe and bounded inference requests.
//!
//! # Run
//!
//! ```bash
//! cargo run --example request_validation
//! ```

use tgi_gtc::validation::{GenerateRequest, RequestValidator, ValidationConfig};

fn main() {
    println!("=== Request Validation Demo ===\n");

    // Configure validator with TGI-style limits
    let config = ValidationConfig::builder()
        .max_input_tokens(4096)
        .max_total_tokens(8192)
        .max_stop_sequences(4)
        .max_stop_sequence_length(64)
        .allow_empty_input(false)
        .build();

    println!("Validation Configuration:");
    println!("  Max input tokens: {}", config.max_input_tokens);
    println!("  Max new tokens: {}", config.max_new_tokens());
    println!("  Max total tokens: {}", config.max_total_tokens);
    println!("  Max stop sequences: {}", config.max_stop_sequences);
    println!();

    let validator = RequestValidator::new(config);

    // Test various requests
    println!("Validation Results:");
    println!("{}", "-".repeat(60));

    // Valid simple request
    let request = GenerateRequest::simple("What is the capital of France?");
    print!("Valid simple request: ");
    match validator.validate(&request) {
        Ok(validated) => {
            println!("✓ VALID");
            println!(
                "    Input tokens: ~{}, Max new: {}",
                validated.input_tokens, validated.max_new_tokens
            );
        }
        Err(e) => println!("✗ INVALID: {}", e),
    }
    println!();

    // Valid with parameters
    let request = GenerateRequest {
        inputs: "Explain quantum computing".to_string(),
        max_new_tokens: Some(500),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: vec!["###".to_string()],
        ..Default::default()
    };
    print!("Valid with parameters: ");
    match validator.validate(&request) {
        Ok(validated) => {
            println!("✓ VALID");
            println!(
                "    Input tokens: ~{}, Max new: {}",
                validated.input_tokens, validated.max_new_tokens
            );
        }
        Err(e) => println!("✗ INVALID: {}", e),
    }
    println!();

    // Invalid temperature (too high)
    let request = GenerateRequest {
        inputs: "Test".to_string(),
        temperature: Some(2.5),
        ..Default::default()
    };
    print!("Invalid temperature (too high): ");
    match validator.validate(&request) {
        Ok(_) => println!("✓ VALID"),
        Err(e) => {
            println!("✗ INVALID");
            println!("    Error: {}", e);
        }
    }
    println!();

    // Invalid temperature (negative)
    let request = GenerateRequest {
        inputs: "Test".to_string(),
        temperature: Some(-0.1),
        ..Default::default()
    };
    print!("Invalid temperature (negative): ");
    match validator.validate(&request) {
        Ok(_) => println!("✓ VALID"),
        Err(e) => {
            println!("✗ INVALID");
            println!("    Error: {}", e);
        }
    }
    println!();

    // Invalid top_p (too high)
    let request = GenerateRequest {
        inputs: "Test".to_string(),
        top_p: Some(1.5),
        ..Default::default()
    };
    print!("Invalid top_p (too high): ");
    match validator.validate(&request) {
        Ok(_) => println!("✓ VALID"),
        Err(e) => {
            println!("✗ INVALID");
            println!("    Error: {}", e);
        }
    }
    println!();

    // Empty input (not allowed)
    let request = GenerateRequest::simple("");
    print!("Empty input (not allowed): ");
    match validator.validate(&request) {
        Ok(_) => println!("✓ VALID"),
        Err(e) => {
            println!("✗ INVALID");
            println!("    Error: {}", e);
        }
    }
    println!();

    // Too many stop sequences
    let request = GenerateRequest {
        inputs: "Test".to_string(),
        stop: vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ],
        ..Default::default()
    };
    print!("Too many stop sequences: ");
    match validator.validate(&request) {
        Ok(_) => println!("✓ VALID"),
        Err(e) => {
            println!("✗ INVALID");
            println!("    Error: {}", e);
        }
    }
    println!();

    // Batch validation
    println!("\nBatch Validation:");
    println!("{}", "-".repeat(60));

    let batch = vec![
        GenerateRequest::simple("Request 1"),
        GenerateRequest::simple("Request 2"),
        GenerateRequest {
            inputs: "Request 3".to_string(),
            temperature: Some(0.8),
            ..Default::default()
        },
    ];

    match validator.validate_batch(&batch) {
        Ok(validated) => {
            println!("✓ Batch valid: {} requests", validated.len());
            for (i, v) in validated.iter().enumerate() {
                println!("    Request {}: ~{} input tokens", i + 1, v.input_tokens);
            }
        }
        Err(e) => {
            println!("✗ Batch invalid: {}", e);
        }
    }

    println!("\n=== Demo Complete ===");
}
