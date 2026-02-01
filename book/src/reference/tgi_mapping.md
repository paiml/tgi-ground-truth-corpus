# TGI Source Mapping

This document maps each pattern to its original TGI source location.

## Router

| Pattern | TGI Source |
|---------|------------|
| `RouterConfig` | `router/src/main.rs` CLI args |
| `Router` | `router/src/server.rs` Axum router |
| `RequestGuard` | `router/src/server.rs` semaphore |
| `HealthStatus` | `router/src/server.rs` `/health` |
| `RouterMetrics` | `router/src/server.rs` `/metrics` |

## Batching

| Pattern | TGI Source |
|---------|------------|
| `BatchConfig` | `backends/v3/src/queue.rs` |
| `ContinuousBatcher` | `backends/v3/src/queue.rs` Queue |
| `BatchRequest` | `backends/v3/src/queue.rs` Entry |
| `Batch` | `backends/v3/src/queue.rs` Batch |

## Streaming

| Pattern | TGI Source |
|---------|------------|
| `SseFormatter` | `router/src/server.rs` |
| `TokenEvent` | `router/src/lib.rs` StreamResponse |
| `CompleteEvent` | `router/src/lib.rs` StreamResponse |
| `FinishReason` | `router/src/lib.rs` |

## Validation

| Pattern | TGI Source |
|---------|------------|
| `ValidationConfig` | `router/src/validation.rs` |
| `GenerateRequest` | `router/src/lib.rs` GenerateRequest |
| `RequestValidator` | `router/src/validation.rs` |
| `ValidatedRequest` | `router/src/validation.rs` |

## Scheduling

| Pattern | TGI Source |
|---------|------------|
| `Scheduler` | `backends/v3/src/block_allocator.rs` |
| `Priority` | `backends/v3/src/queue.rs` |
| `ScheduledTask` | `backends/v3/src/queue.rs` Entry |

## Inference

| Pattern | TGI Source |
|---------|------------|
| `InferenceBackend` | `backends/v3/src/backend.rs` |
| `BackendConfig` | `backends/v3/src/backend.rs` |
| `BackendState` | `backends/v3/src/backend.rs` |
| `DataType` | `backends/v3/src/lib.rs` |

## Quantization

| Pattern | TGI Source |
|---------|------------|
| `QuantType` | `backends/llamacpp/src/quantize.rs` |

## TGI Repository

- **Repository**: [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)
- **Version**: 3.1.0
- **License**: Apache 2.0
