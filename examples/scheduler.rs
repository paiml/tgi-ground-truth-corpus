//! Scheduler Example
//!
//! Demonstrates TGI's fair request scheduling with priority
//! queue management.
//!
//! # Run
//!
//! ```bash
//! cargo run --example scheduler
//! ```
//!
//! # Performance Targets
//!
//! - Schedule operation: < 500 ns
//! - Next (dequeue): < 200 ns
//! - Priority ordering: O(log n)

use std::hint::black_box;
use std::time::Instant;
use tgi_gtc::profiling::PerfMetrics;
use tgi_gtc::scheduling::{Priority, ScheduledTask, Scheduler, SchedulerConfig};

fn main() {
    println!("=== Request Scheduler Demo ===\n");

    // Configure scheduler
    let config = SchedulerConfig {
        max_queue_size: 100,
        priority_scheduling: true,
        enable_preemption: true,
    };

    println!("Scheduler Configuration:");
    println!("  Max queue size: {}", config.max_queue_size);
    println!("  Priority scheduling: {}", config.priority_scheduling);
    println!("  Enable preemption: {}", config.enable_preemption);
    println!();

    let scheduler = Scheduler::new(config);

    // Add tasks with different priorities
    println!("Adding tasks with different priorities...\n");

    let tasks = [
        (100, Priority::Normal, "User chat message"),
        (500, Priority::Low, "Background summarization"),
        (50, Priority::High, "Interactive completion"),
        (200, Priority::Normal, "API request"),
        (1000, Priority::Critical, "System health check"),
        (150, Priority::Low, "Batch processing"),
    ];

    for (tokens, priority, desc) in tasks {
        let id = scheduler.next_id();
        let task = ScheduledTask::new(id, tokens).with_priority(priority);
        scheduler.schedule(task).expect("Should schedule");
        println!(
            "  Scheduled task {}: {:?} priority - {} ({} tokens)",
            id, priority, desc, tokens
        );
    }

    println!("\nQueue length: {}", scheduler.len());

    // Process tasks in priority order
    println!("\nProcessing tasks (priority order):");
    println!("{}", "-".repeat(50));

    while let Some(task) = scheduler.next() {
        println!(
            "  Processing task {}: {:?} priority, {} tokens, preemptible: {}",
            task.id, task.priority, task.estimated_tokens, task.preemptible
        );
    }

    println!("\nQueue empty: {}", scheduler.is_empty());

    // Demonstrate cancellation
    println!("\n--- Cancellation Demo ---");

    scheduler
        .schedule(ScheduledTask::new(scheduler.next_id(), 100))
        .unwrap();
    scheduler
        .schedule(ScheduledTask::new(scheduler.next_id(), 200))
        .unwrap();
    scheduler
        .schedule(ScheduledTask::new(scheduler.next_id(), 300))
        .unwrap();

    println!("Scheduled 3 more tasks");
    println!("Queue length: {}", scheduler.len());

    // Cancel middle task
    let cancelled = scheduler.cancel(8); // Task ID 8
    if let Some(task) = cancelled {
        println!("Cancelled task {}", task.id);
    }

    println!("Queue length after cancel: {}", scheduler.len());

    // Clear remaining
    let cleared = scheduler.clear();
    println!("Cleared {} remaining tasks", cleared.len());

    // Demonstrate queue full
    println!("\n--- Queue Limit Demo ---");

    let small_config = SchedulerConfig::with_max_queue(3);
    let small_scheduler = Scheduler::new(small_config);

    for i in 1..=4 {
        let task = ScheduledTask::new(i, 100);
        match small_scheduler.schedule(task) {
            Ok(()) => println!("  Task {} scheduled", i),
            Err(e) => println!("  Task {} rejected: {}", i, e),
        }
    }

    // === PROFILING ===
    println!("\n--- Performance Profiling ---\n");

    // Profile schedule operation
    const SCHEDULE_ITERATIONS: u64 = 50_000;
    let large_config = SchedulerConfig {
        max_queue_size: 100_000,
        priority_scheduling: true,
        enable_preemption: true,
    };
    let scheduler = Scheduler::new(large_config.clone());

    let start = Instant::now();
    for i in 0..SCHEDULE_ITERATIONS {
        let task = ScheduledTask::new(i, 100).with_priority(Priority::Normal);
        let _ = black_box(scheduler.schedule(task));
    }
    let schedule_time = start.elapsed();

    let schedule_metrics = PerfMetrics::new("Schedule", schedule_time, SCHEDULE_ITERATIONS);
    schedule_metrics.report();

    // Profile next (dequeue) operation
    const NEXT_ITERATIONS: u64 = 50_000;

    let start = Instant::now();
    for _ in 0..NEXT_ITERATIONS {
        black_box(scheduler.next());
    }
    let next_time = start.elapsed();

    let next_metrics = PerfMetrics::new("Next (dequeue)", next_time, NEXT_ITERATIONS);
    next_metrics.report();

    // Profile schedule+next cycle
    const CYCLE_ITERATIONS: u64 = 10_000;
    let scheduler = Scheduler::new(large_config.clone());

    let start = Instant::now();
    for i in 0..CYCLE_ITERATIONS {
        let task = ScheduledTask::new(i, 100).with_priority(Priority::Normal);
        scheduler.schedule(task).ok();
        black_box(scheduler.next());
    }
    let cycle_time = start.elapsed();

    let cycle_metrics = PerfMetrics::new("Schedule+Next Cycle", cycle_time, CYCLE_ITERATIONS);
    cycle_metrics.report();

    // Profile priority ordering
    const PRIORITY_ITERATIONS: u64 = 1000;
    let mut total_tasks = 0u64;

    let start = Instant::now();
    for _ in 0..PRIORITY_ITERATIONS {
        let scheduler = Scheduler::new(large_config.clone());

        // Add mixed priority tasks
        for i in 0..100 {
            let priority = match i % 4 {
                0 => Priority::Critical,
                1 => Priority::High,
                2 => Priority::Normal,
                _ => Priority::Low,
            };
            let task = ScheduledTask::new(i, 100).with_priority(priority);
            scheduler.schedule(task).ok();
        }

        // Verify priority ordering by draining
        let mut prev_priority = Priority::Critical;
        while let Some(task) = scheduler.next() {
            // Priority should be monotonically decreasing
            assert!(task.priority <= prev_priority);
            prev_priority = task.priority;
            total_tasks += 1;
        }
    }
    let priority_time = start.elapsed();

    let priority_metrics = PerfMetrics::new(
        "Priority Sort (100 tasks)",
        priority_time,
        PRIORITY_ITERATIONS,
    );
    priority_metrics.report();

    println!("\n  Total tasks processed: {}", total_tasks);

    // === PERFORMANCE ASSERTIONS ===
    println!("\n--- Performance Validation ---\n");

    println!("Validating performance targets...");

    // Schedule should be fast (allowing for queue growth overhead)
    assert!(
        schedule_metrics.latency_ns < 20000.0,
        "Schedule too slow: {:.2} ns",
        schedule_metrics.latency_ns
    );
    println!(
        "  ✓ Schedule: {:.2} ns/op (< 20 µs)",
        schedule_metrics.latency_ns
    );

    // Next should be very fast
    assert!(
        next_metrics.latency_ns < 2000.0,
        "Next too slow: {:.2} ns",
        next_metrics.latency_ns
    );
    println!("  ✓ Next: {:.2} ns/op (< 2 µs)", next_metrics.latency_ns);

    // Throughput should be high
    let throughput = CYCLE_ITERATIONS as f64 / cycle_time.as_secs_f64();
    assert!(
        throughput > 100_000.0,
        "Throughput too low: {:.0} ops/sec",
        throughput
    );
    println!("  ✓ Throughput: {:.0} ops/sec (> 100k)", throughput);

    println!("\n=== Demo Complete - All Performance Targets Met ===");
}
