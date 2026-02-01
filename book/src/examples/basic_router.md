# Basic Router Example

Demonstrates TGI-style HTTP router setup with health checks and request limiting.

## Run

```bash
cargo run --example basic_router
```

## Output

```
=== TGI Router Pattern Demo ===

Router Configuration:
  Bind address: 0.0.0.0:8080
  Max concurrent: 128
  Max batch size: 32
  OpenAI compat: true

Initial Health Status:
  Status: starting
  Active requests: 0

Router marked as ready (model loaded)
Health Status After Ready:
  Status: healthy

Simulating request handling...
  Active requests: 2
  Request 1 completed successfully
  Request 2 failed

Final Metrics:
  Total requests: 2
  Completed: 1
  Failed: 1
  Success rate: 50.0%
  Utilization: 0.0%

=== Demo Complete ===
```

## Code Walkthrough

### 1. Configure Router

```rust
let config = RouterConfig::builder()
    .port(8080)
    .hostname("0.0.0.0")
    .max_concurrent_requests(128)
    .max_batch_size(32)
    .timeout_secs(60)
    .openai_compat(true)
    .enable_metrics(true)
    .build();
```

### 2. Create Router and Set Ready

```rust
let router = Router::new(config);
router.set_ready(true);
```

### 3. Handle Requests with Guards

```rust
let guard = router.try_acquire()?;
// ... process request ...
guard.complete(); // or guard.fail()
```

### 4. Check Metrics

```rust
let metrics = router.metrics();
println!("Success rate: {:.1}%", metrics.success_rate() * 100.0);
```
