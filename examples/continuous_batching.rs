//! Continuous Batching Example
//!
//! Demonstrates TGI's continuous batching algorithm for optimal
//! GPU utilization during inference.
//!
//! # Run
//!
//! ```bash
//! cargo run --example continuous_batching
//! ```

use tgi_gtc::batching::{BatchConfig, BatchRequest, ContinuousBatcher};

fn main() {
    println!("=== Continuous Batching Demo ===\n");

    // Configure batcher with TGI-style settings
    let config = BatchConfig::builder()
        .max_batch_size(8) // Max requests per batch
        .max_batch_tokens(2048) // Token budget per batch
        .min_batch_size(2) // Wait for at least 2 requests
        .max_wait_ms(50) // Or 50ms timeout
        .build();

    println!("Batcher Configuration:");
    println!("  Max batch size: {}", config.max_batch_size);
    println!("  Max batch tokens: {}", config.max_batch_tokens);
    println!("  Min batch size: {}", config.min_batch_size);
    println!("  Max wait: {}ms", config.max_wait_ms);
    println!();

    let batcher = ContinuousBatcher::new(config);

    // Simulate incoming requests
    println!("Simulating incoming requests...\n");

    let requests = [
        ("Short prompt", 50, 100),
        ("Medium prompt", 150, 200),
        ("Long prompt", 300, 150),
        ("Another short", 75, 50),
        ("Complex query", 200, 300),
        ("Simple ask", 25, 50),
    ];

    for (desc, input_tokens, max_new) in requests {
        let id = batcher.next_id();
        let request = BatchRequest::new(id, input_tokens, max_new);
        batcher.add(request);
        println!(
            "  Added request {}: {} ({} input, {} max new)",
            id, desc, input_tokens, max_new
        );
    }

    println!("\nQueue length: {}", batcher.queue_len());

    // Form batches respecting limits
    println!("\nForming batches...\n");

    let mut batch_num = 1;
    while let Some(batch) = batcher.try_form_batch() {
        println!("Batch {}:", batch_num);
        println!("  Requests: {}", batch.size());
        println!("  Total input tokens: {}", batch.total_input_tokens);
        println!("  Request IDs: {:?}", batch.requests.iter().map(|r| r.id).collect::<Vec<_>>());
        println!();
        batch_num += 1;
    }

    if batcher.queue_len() > 0 {
        println!("Remaining in queue: {} (waiting for min batch size)", batcher.queue_len());

        // Force remaining batch
        if let Some(batch) = batcher.force_batch() {
            println!("\nForced final batch:");
            println!("  Requests: {}", batch.size());
            println!("  Total input tokens: {}", batch.total_input_tokens);
        }
    }

    println!("\n=== Demo Complete ===");
}
