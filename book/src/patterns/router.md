# Router Pattern

HTTP routing with concurrency control, health checks, and metrics.

## TGI Source

Derived from `router/src/server.rs`:
- Axum-based HTTP server
- OpenAI-compatible `/v1/chat/completions` endpoint
- Health and readiness probes
- Prometheus metrics

## Sovereign AI Stack Equivalent

Maps to `realizar::serve` for model serving infrastructure.

## Key Components

### RouterConfig

Configuration for the HTTP router:

```rust
use tgi_gtc::router::RouterConfig;

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

### Router

The main router with state management:

```rust
use tgi_gtc::router::Router;

let router = Router::new(config);

// Mark ready after model loads
router.set_ready(true);

// Acquire request slots
let guard = router.try_acquire()?;
// ... handle request ...
guard.complete(); // or guard.fail()
```

### RequestGuard

RAII guard for request slot management:

- Automatically releases slot on drop
- Tracks completed vs failed requests
- Prevents resource leaks on panic

### Health Status

```rust
let health = router.health();
// health.status: "healthy" | "starting" | "unhealthy"
// health.active_requests: current count
// health.max_concurrent_requests: limit
```

### Metrics

```rust
let metrics = router.metrics();
// metrics.total_requests
// metrics.completed_requests
// metrics.failed_requests
// metrics.success_rate() -> f64
// metrics.utilization() -> f64
```

## Design Patterns

### Backpressure

The router implements backpressure through `max_concurrent_requests`:

1. Each request acquires a slot via `try_acquire()`
2. If at capacity, returns `ResourceExhausted` error
3. Client receives 429 Too Many Requests
4. Slot released when guard drops

### Health Probes

TGI-style health endpoints:

- `/health` - Always returns 200 (liveness)
- `/ready` - Returns 200 only when ready (readiness)

Use `is_ready()` to check readiness before accepting requests.

## Example

```bash
cargo run --example basic_router
```
