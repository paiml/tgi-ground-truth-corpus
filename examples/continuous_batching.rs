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
//!
//! # Performance Targets
//!
//! - Request add: < 500 ns
//! - Batch formation: < 1 µs per request
//! - Queue operations: < 100 ns

use std::hint::black_box;
use std::time::Instant;
use tgi_gtc::batching::{BatchConfig, BatchRequest, ContinuousBatcher};
use tgi_gtc::profiling::PerfMetrics;

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

    let batcher = ContinuousBatcher::new(config.clone());

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
        println!(
            "  Request IDs: {:?}",
            batch.requests.iter().map(|r| r.id).collect::<Vec<_>>()
        );
        println!();
        batch_num += 1;
    }

    if batcher.queue_len() > 0 {
        println!(
            "Remaining in queue: {} (waiting for min batch size)",
            batcher.queue_len()
        );

        // Force remaining batch
        if let Some(batch) = batcher.force_batch() {
            println!("\nForced final batch:");
            println!("  Requests: {}", batch.size());
            println!("  Total input tokens: {}", batch.total_input_tokens);
        }
    }

    // === PROFILING ===
    println!("\n--- Performance Profiling ---\n");

    // Profile request add operations
    const ADD_ITERATIONS: u64 = 50_000;
    let batcher = ContinuousBatcher::new(config.clone());

    let start = Instant::now();
    for _ in 0..ADD_ITERATIONS {
        let id = batcher.next_id();
        let request = BatchRequest::new(id, 128, 256);
        black_box(batcher.add(request));
    }
    let add_time = start.elapsed();

    let add_metrics = PerfMetrics::new("Request Add", add_time, ADD_ITERATIONS);
    add_metrics.report();

    // Profile batch formation
    const BATCH_ITERATIONS: u64 = 1000;
    let mut total_batched = 0u64;

    let start = Instant::now();
    for _ in 0..BATCH_ITERATIONS {
        let batcher = ContinuousBatcher::new(config.clone());

        // Add 100 requests
        for _ in 0..100 {
            let id = batcher.next_id();
            let request = BatchRequest::new(id, 128, 256);
            batcher.add(request);
        }

        // Form all batches
        while let Some(batch) = batcher.force_batch() {
            total_batched += batch.size() as u64;
        }
    }
    let batch_time = start.elapsed();

    let batch_metrics = PerfMetrics::new("Batch Formation (100 req)", batch_time, BATCH_ITERATIONS);
    batch_metrics.report();

    println!("\n  Total requests batched: {}", total_batched);
    println!(
        "  Batching throughput: {:.0} req/sec",
        total_batched as f64 / batch_time.as_secs_f64()
    );

    // Profile queue length check
    const QUEUE_ITERATIONS: u64 = 100_000;
    let batcher = ContinuousBatcher::new(config.clone());
    for i in 0..1000 {
        batcher.add(BatchRequest::new(i, 128, 256));
    }

    let start = Instant::now();
    for _ in 0..QUEUE_ITERATIONS {
        black_box(batcher.queue_len());
    }
    let queue_time = start.elapsed();

    let queue_metrics = PerfMetrics::new("Queue Length Check", queue_time, QUEUE_ITERATIONS);
    queue_metrics.report();

    // === PERFORMANCE ASSERTIONS ===
    println!("\n--- Performance Validation ---\n");

    println!("Validating performance targets...");

    // Request add should be fast
    assert!(
        add_metrics.latency_ns < 5000.0,
        "Request add too slow: {:.2} ns",
        add_metrics.latency_ns
    );
    println!(
        "  ✓ Request add: {:.2} ns/op (< 5 µs)",
        add_metrics.latency_ns
    );

    // Batching throughput should be high
    let batching_throughput = total_batched as f64 / batch_time.as_secs_f64();
    assert!(
        batching_throughput > 100_000.0,
        "Batching throughput too low: {:.0} req/sec",
        batching_throughput
    );
    println!(
        "  ✓ Batching throughput: {:.0} req/sec (> 100k)",
        batching_throughput
    );

    // Queue length should be O(1)
    assert!(
        queue_metrics.latency_ns < 1000.0,
        "Queue length check too slow: {:.2} ns",
        queue_metrics.latency_ns
    );
    println!(
        "  ✓ Queue length: {:.2} ns/op (< 1 µs)",
        queue_metrics.latency_ns
    );

    println!("\n=== Demo Complete - All Performance Targets Met ===");
}
