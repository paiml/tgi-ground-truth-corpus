# API Documentation

Full API documentation is available on docs.rs:

**[docs.rs/tgi-gtc](https://docs.rs/tgi-gtc)**

## Module Overview

### `tgi_gtc::router`

HTTP routing with concurrency control.

- `RouterConfig` - Configuration builder
- `Router` - Main router instance
- `RouterState` - Request tracking
- `RequestGuard` - RAII slot guard
- `HealthStatus` - Health probe response
- `RouterMetrics` - Prometheus-style metrics

### `tgi_gtc::batching`

Continuous batching for GPU utilization.

- `BatchConfig` - Batching configuration
- `BatchRequest` - Individual request
- `Batch` - Formed batch
- `ContinuousBatcher` - Main batcher

### `tgi_gtc::streaming`

SSE streaming for token delivery.

- `StreamEvent` - Event enum
- `TokenEvent` - Token data
- `CompleteEvent` - Completion data
- `FinishReason` - Why generation stopped
- `SseFormatter` - Event formatter

### `tgi_gtc::validation`

Request validation and limits.

- `ValidationConfig` - Validation rules
- `GenerateRequest` - Request builder
- `RequestValidator` - Validator instance
- `ValidatedRequest` - Post-validation

### `tgi_gtc::scheduling`

Priority-based request scheduling.

- `Priority` - Priority levels
- `ScheduledTask` - Task data
- `SchedulerConfig` - Scheduler config
- `Scheduler` - Priority queue

### `tgi_gtc::inference`

Backend initialization patterns.

- `DataType` - Float32/Float16/BFloat16/Int8
- `BackendConfig` - Model configuration
- `BackendState` - Lifecycle states
- `InferenceBackend` - Backend instance

### `tgi_gtc::quantization`

Quantization types and compression.

- `QuantType` - Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/F16/F32

### `tgi_gtc::error`

Error types and handling.

- `Error` - Main error enum
- `Result<T>` - Result type alias

## Generate Local Docs

```bash
cargo doc --no-deps --open
```
