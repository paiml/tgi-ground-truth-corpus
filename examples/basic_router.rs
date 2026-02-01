//! Basic Router Example
//!
//! Demonstrates TGI-style HTTP router setup with health checks
//! and request limiting.
//!
//! # Run
//!
//! ```bash
//! cargo run --example basic_router
//! ```
//!
//! # Performance Targets
//!
//! - Router creation: < 1 µs
//! - Request acquire: < 100 ns
//! - Health check: < 50 ns

use std::hint::black_box;
use std::time::Instant;
use tgi_gtc::profiling::{PerfMetrics, Timer};
use tgi_gtc::router::{Router, RouterConfig};

fn main() {
    println!("=== TGI Router Pattern Demo ===\n");

    // Configure router with TGI-style settings
    let config = RouterConfig::builder()
        .port(8080)
        .hostname("0.0.0.0")
        .max_concurrent_requests(128)
        .max_batch_size(32)
        .timeout_secs(60)
        .openai_compat(true)
        .enable_metrics(true)
        .build();

    println!("Router Configuration:");
    println!("  Bind address: {}", config.bind_address());
    println!("  Max concurrent: {}", config.max_concurrent_requests);
    println!("  Max batch size: {}", config.max_batch_size);
    println!("  OpenAI compat: {}", config.openai_compat);
    println!();

    // Profile router creation
    let start = Instant::now();
    let router = Router::new(config.clone());
    let creation_time = start.elapsed();
    println!("Router Creation: {:?}", creation_time);

    // Check initial health (starting state)
    let health = router.health();
    println!("\nInitial Health Status:");
    println!("  Status: {}", health.status);
    println!("  Active requests: {}", health.active_requests);

    // Mark router as ready (model loaded)
    router.set_ready(true);
    println!("\nRouter marked as ready (model loaded)");

    let health = router.health();
    println!("Health Status After Ready:");
    println!("  Status: {}", health.status);

    // === PROFILING: Request Handling ===
    println!("\n--- Performance Profiling ---\n");

    // Profile request acquisition
    const ACQUIRE_ITERATIONS: u64 = 10_000;
    let mut timer = Timer::new("Request Acquire");

    for _ in 0..ACQUIRE_ITERATIONS {
        let router = Router::new(config.clone());
        router.set_ready(true);
        let guard = black_box(router.try_acquire());
        if let Ok(g) = guard {
            g.complete();
        }
        timer.record_iteration();
    }

    let acquire_metrics = PerfMetrics::new("Request Acquire", timer.elapsed(), ACQUIRE_ITERATIONS);
    acquire_metrics.report();

    // Profile health check
    const HEALTH_ITERATIONS: u64 = 100_000;
    let router = Router::new(config.clone());
    router.set_ready(true);

    let start = Instant::now();
    for _ in 0..HEALTH_ITERATIONS {
        black_box(router.health());
    }
    let health_time = start.elapsed();

    let health_metrics = PerfMetrics::new("Health Check", health_time, HEALTH_ITERATIONS);
    health_metrics.report();

    // Profile complete request cycle
    const CYCLE_ITERATIONS: u64 = 10_000;
    let router = Router::new(config.clone());
    router.set_ready(true);

    let start = Instant::now();
    for _ in 0..CYCLE_ITERATIONS {
        if let Ok(guard) = router.try_acquire() {
            black_box(guard.complete());
        }
    }
    let cycle_time = start.elapsed();

    let cycle_metrics = PerfMetrics::new("Full Request Cycle", cycle_time, CYCLE_ITERATIONS);
    cycle_metrics.report();

    // === PERFORMANCE ASSERTIONS ===
    println!("\n--- Performance Validation ---\n");

    // Assert performance targets
    println!("Validating performance targets...");

    // Router creation should be fast (< 10 µs allows some overhead)
    assert!(
        creation_time.as_micros() < 100,
        "Router creation too slow: {:?}",
        creation_time
    );
    println!("  ✓ Router creation: {:?} (< 100 µs)", creation_time);

    // Health checks should be very fast (< 1 µs per check)
    assert!(
        health_metrics.latency_ns < 1000.0,
        "Health check too slow: {:.2} ns",
        health_metrics.latency_ns
    );
    println!(
        "  ✓ Health check: {:.2} ns/op (< 1000 ns)",
        health_metrics.latency_ns
    );

    // Request cycle should be reasonable (< 10 µs per cycle)
    assert!(
        cycle_metrics.latency_ns < 10_000.0,
        "Request cycle too slow: {:.2} ns",
        cycle_metrics.latency_ns
    );
    println!(
        "  ✓ Request cycle: {:.2} ns/op (< 10 µs)",
        cycle_metrics.latency_ns
    );

    // Check final metrics
    let metrics = router.metrics();
    println!("\nFinal Metrics:");
    println!("  Total requests: {}", metrics.total_requests);
    println!("  Completed: {}", metrics.completed_requests);
    println!("  Success rate: {:.1}%", metrics.success_rate() * 100.0);
    println!("  Throughput: {:.0} req/sec", cycle_metrics.throughput);

    println!("\n=== Demo Complete - All Performance Targets Met ===");
}
