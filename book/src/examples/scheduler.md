# Scheduler Example

Demonstrates TGI's fair request scheduling with priority queues.

## Run

```bash
cargo run --example scheduler
```

## Output

```
=== Request Scheduler Demo ===

Scheduler Configuration:
  Max queue size: 100
  Priority scheduling: true
  Enable preemption: true

Adding tasks with different priorities...

  Scheduled task 1: Normal priority - User chat message (100 tokens)
  Scheduled task 2: Low priority - Background summarization (500 tokens)
  Scheduled task 3: High priority - Interactive completion (50 tokens)
  Scheduled task 4: Normal priority - API request (200 tokens)
  Scheduled task 5: Critical priority - System health check (1000 tokens)
  Scheduled task 6: Low priority - Batch processing (150 tokens)

Queue length: 6

Processing tasks (priority order):
--------------------------------------------------
  Processing task 5: Critical priority, 1000 tokens, preemptible: true
  Processing task 3: High priority, 50 tokens, preemptible: true
  Processing task 1: Normal priority, 100 tokens, preemptible: true
  Processing task 4: Normal priority, 200 tokens, preemptible: true
  Processing task 2: Low priority, 500 tokens, preemptible: true
  Processing task 6: Low priority, 150 tokens, preemptible: true

Queue empty: true
```

## Code Walkthrough

### 1. Configure Scheduler

```rust
let config = SchedulerConfig {
    max_queue_size: 100,
    priority_scheduling: true,
    enable_preemption: true,
};

let scheduler = Scheduler::new(config);
```

### 2. Schedule Tasks

```rust
let task = ScheduledTask::new(id, tokens)
    .with_priority(Priority::High);

scheduler.schedule(task)?;
```

### 3. Process in Priority Order

```rust
while let Some(task) = scheduler.next() {
    println!("Processing: {:?}", task.priority);
}
```
