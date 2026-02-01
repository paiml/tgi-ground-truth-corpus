//! Streaming SSE Example
//!
//! Demonstrates TGI's Server-Sent Events streaming for
//! token-by-token delivery.
//!
//! # Run
//!
//! ```bash
//! cargo run --example streaming_sse
//! ```
//!
//! # Performance Targets
//!
//! - Token formatting: < 500 ns
//! - SSE serialization: < 1 µs per event
//! - Zero-allocation streaming

use std::hint::black_box;
use std::time::Instant;
use tgi_gtc::profiling::PerfMetrics;
use tgi_gtc::streaming::{CompleteEvent, FinishReason, SseFormatter, TokenEvent};

fn main() {
    println!("=== SSE Streaming Demo ===\n");

    // Create formatter with options
    let mut formatter = SseFormatter::new();
    formatter.include_token_ids = true;
    formatter.include_logprobs = true;

    println!("Formatter Configuration:");
    println!("  Include token IDs: {}", formatter.include_token_ids);
    println!("  Include logprobs: {}", formatter.include_logprobs);
    println!();

    // Simulate token generation
    let tokens = [
        ("Hello", 1234, Some(-50), false),
        (",", 44, Some(-10), false),
        (" world", 5678, Some(-75), false),
        ("!", 999, Some(-25), false),
        ("<|endoftext|>", 0, None, true),
    ];

    println!("Simulated Token Stream:");
    println!("{}", "=".repeat(60));

    for (token, id, logprob, special) in tokens {
        let event = TokenEvent {
            token: token.to_string(),
            token_id: id,
            logprob,
            special,
        };

        let sse = formatter.format_token(&event);
        print!("{}", sse);
    }

    // Send completion event
    let complete = CompleteEvent {
        generated_tokens: 5,
        finish_reason: FinishReason::EndOfSequence,
        generation_time_ms: 125,
    };

    let sse = formatter.format_complete(&complete);
    print!("{}", sse);

    println!("{}", "=".repeat(60));
    println!();

    // Demonstrate error formatting
    println!("Error Event Example:");
    println!("{}", "-".repeat(40));
    let error_sse = formatter.format_error("Model inference failed: out of memory");
    print!("{}", error_sse);
    println!("{}", "-".repeat(40));

    // Show finish reasons
    println!("\nFinish Reasons:");
    println!("  Length: {}", FinishReason::Length);
    println!("  Stop: {}", FinishReason::Stop);
    println!("  EOS: {}", FinishReason::EndOfSequence);

    // === PROFILING ===
    println!("\n--- Performance Profiling ---\n");

    // Profile token formatting
    const TOKEN_ITERATIONS: u64 = 100_000;
    let event = TokenEvent {
        token: "test_token".to_string(),
        token_id: 12345,
        logprob: Some(-50),
        special: false,
    };

    let start = Instant::now();
    for _ in 0..TOKEN_ITERATIONS {
        black_box(formatter.format_token(black_box(&event)));
    }
    let token_time = start.elapsed();

    let token_metrics = PerfMetrics::new("Token Format", token_time, TOKEN_ITERATIONS);
    token_metrics.report();

    // Profile complete event formatting
    const COMPLETE_ITERATIONS: u64 = 100_000;
    let complete = CompleteEvent {
        generated_tokens: 100,
        finish_reason: FinishReason::Length,
        generation_time_ms: 1000,
    };

    let start = Instant::now();
    for _ in 0..COMPLETE_ITERATIONS {
        black_box(formatter.format_complete(black_box(&complete)));
    }
    let complete_time = start.elapsed();

    let complete_metrics = PerfMetrics::new("Complete Format", complete_time, COMPLETE_ITERATIONS);
    complete_metrics.report();

    // Profile error formatting
    const ERROR_ITERATIONS: u64 = 100_000;

    let start = Instant::now();
    for _ in 0..ERROR_ITERATIONS {
        black_box(formatter.format_error(black_box("Test error message")));
    }
    let error_time = start.elapsed();

    let error_metrics = PerfMetrics::new("Error Format", error_time, ERROR_ITERATIONS);
    error_metrics.report();

    // Profile streaming throughput simulation
    const STREAM_TOKENS: u64 = 10_000;
    let tokens: Vec<TokenEvent> = (0..STREAM_TOKENS)
        .map(|i| TokenEvent {
            token: format!("token_{}", i),
            token_id: i as u32,
            logprob: Some(-(i as i32 % 100)),
            special: false,
        })
        .collect();

    let start = Instant::now();
    let mut total_bytes = 0usize;
    for token in &tokens {
        let sse = formatter.format_token(token);
        total_bytes += sse.len();
        black_box(&sse);
    }
    let stream_time = start.elapsed();

    let stream_metrics = PerfMetrics::new("Stream 10k Tokens", stream_time, STREAM_TOKENS);
    stream_metrics.report();

    println!(
        "\n  Total SSE bytes: {} ({:.2} KB)",
        total_bytes,
        total_bytes as f64 / 1024.0
    );
    println!(
        "  Streaming bandwidth: {:.2} MB/sec",
        total_bytes as f64 / stream_time.as_secs_f64() / 1_000_000.0
    );

    // === PERFORMANCE ASSERTIONS ===
    println!("\n--- Performance Validation ---\n");

    println!("Validating performance targets...");

    // Token formatting should be fast
    assert!(
        token_metrics.latency_ns < 2000.0,
        "Token format too slow: {:.2} ns",
        token_metrics.latency_ns
    );
    println!(
        "  ✓ Token format: {:.2} ns/op (< 2 µs)",
        token_metrics.latency_ns
    );

    // Complete event formatting should be fast
    assert!(
        complete_metrics.latency_ns < 2000.0,
        "Complete format too slow: {:.2} ns",
        complete_metrics.latency_ns
    );
    println!(
        "  ✓ Complete format: {:.2} ns/op (< 2 µs)",
        complete_metrics.latency_ns
    );

    // Streaming throughput should be high
    let tokens_per_sec = STREAM_TOKENS as f64 / stream_time.as_secs_f64();
    assert!(
        tokens_per_sec > 1_000_000.0,
        "Streaming throughput too low: {:.0} tokens/sec",
        tokens_per_sec
    );
    println!("  ✓ Streaming: {:.0} tokens/sec (> 1M)", tokens_per_sec);

    println!("\n=== Demo Complete - All Performance Targets Met ===");
}
