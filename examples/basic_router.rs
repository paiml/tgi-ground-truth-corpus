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

    // Create router instance
    let router = Router::new(config);

    // Check initial health (starting state)
    let health = router.health();
    println!("Initial Health Status:");
    println!("  Status: {}", health.status);
    println!("  Active requests: {}", health.active_requests);
    println!();

    // Mark router as ready (model loaded)
    router.set_ready(true);
    println!("Router marked as ready (model loaded)");

    let health = router.health();
    println!("Health Status After Ready:");
    println!("  Status: {}", health.status);
    println!();

    // Simulate request handling with guards
    println!("Simulating request handling...");

    // Acquire request slots
    let guard1 = router.try_acquire().expect("Should acquire slot 1");
    let guard2 = router.try_acquire().expect("Should acquire slot 2");

    println!("  Active requests: {}", router.state().active_count());

    // Complete first request successfully
    guard1.complete();
    println!("  Request 1 completed successfully");

    // Second request fails
    guard2.fail();
    println!("  Request 2 failed");

    // Check metrics
    let metrics = router.metrics();
    println!("\nFinal Metrics:");
    println!("  Total requests: {}", metrics.total_requests);
    println!("  Completed: {}", metrics.completed_requests);
    println!("  Failed: {}", metrics.failed_requests);
    println!("  Success rate: {:.1}%", metrics.success_rate() * 100.0);
    println!("  Utilization: {:.1}%", metrics.utilization() * 100.0);

    println!("\n=== Demo Complete ===");
}
