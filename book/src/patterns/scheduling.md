# Scheduling Pattern

Fair request scheduling with priority queue management.

## TGI Source

Derived from `backends/v3/src/block_allocator.rs`:
- Block-based KV cache allocation
- Memory pool management
- Fair request scheduling

## Sovereign AI Stack Equivalent

Maps to `realizar::schedule` for inference scheduling.

## Key Components

### Priority

Request priority levels:

```rust
use tgi_gtc::scheduling::Priority;

enum Priority {
    Low,      // Background tasks
    Normal,   // Default
    High,     // Interactive
    Critical, // System
}
```

Priority ordering: `Critical > High > Normal > Low`

### ScheduledTask

Task in the scheduler:

```rust
use tgi_gtc::scheduling::ScheduledTask;

let task = ScheduledTask::new(
    id,              // Unique ID
    estimated_tokens // Token estimate
)
.with_priority(Priority::High)
.preemptible(false);  // Can't be preempted
```

### SchedulerConfig

```rust
use tgi_gtc::scheduling::SchedulerConfig;

let config = SchedulerConfig {
    max_queue_size: 1000,
    priority_scheduling: true,
    enable_preemption: true,
};

// Or with helper
let config = SchedulerConfig::with_max_queue(500);
```

### Scheduler

```rust
use tgi_gtc::scheduling::Scheduler;

let scheduler = Scheduler::new(config);

// Generate IDs
let id = scheduler.next_id();

// Schedule task
scheduler.schedule(task)?;

// Get next task (priority order)
if let Some(task) = scheduler.next() {
    // Process task
}

// Peek without removing
if let Some(task) = scheduler.peek() {
    println!("Next: {:?}", task.priority);
}

// Cancel specific task
scheduler.cancel(task_id);

// Clear all
let cancelled = scheduler.clear();
```

## Scheduling Algorithm

### Priority Scheduling (enabled)

Tasks sorted by priority, FIFO within same priority:

```
Queue: [Critical-1, Critical-2, High-1, Normal-1, Low-1]
       ^-- next()
```

### FIFO Scheduling (disabled)

Pure first-in-first-out:

```
Queue: [Normal-1, High-1, Critical-1, Low-1]
       ^-- next()
```

## Preemption

When `enable_preemption` is true:

1. Higher priority task arrives
2. If running task is `preemptible`:
   - Pause current task
   - Save KV cache state
   - Run higher priority task
   - Resume original task

Use `preemptible(false)` for tasks that can't be interrupted.

## Queue Management

### Backpressure

When queue is full (`len >= max_queue_size`):

```rust
match scheduler.schedule(task) {
    Ok(()) => println!("Scheduled"),
    Err(e) => println!("Queue full: {}", e),
}
```

### Cancellation

Cancel tasks that are no longer needed:

```rust
// Cancel by ID
if let Some(cancelled) = scheduler.cancel(task_id) {
    println!("Cancelled task {}", cancelled.id);
}

// Clear all
let cancelled = scheduler.clear();
println!("Cancelled {} tasks", cancelled.len());
```

## Thread Safety

The scheduler uses `Mutex` for thread-safe access:

```rust
// Safe to use from multiple threads
let scheduler = Arc::new(Scheduler::default());

// Thread 1
scheduler.schedule(task1)?;

// Thread 2
if let Some(task) = scheduler.next() {
    // Process
}
```

## Example

```bash
cargo run --example scheduler
```
