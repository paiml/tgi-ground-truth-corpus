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

    println!("\n=== Demo Complete ===");
}
