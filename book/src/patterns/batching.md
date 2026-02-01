# Continuous Batching Pattern

Dynamic batch formation for optimal GPU utilization.

## TGI Source

Derived from `backends/v3/src/queue.rs`:
- Continuous batching algorithm
- Prefill vs decode phase handling
- Dynamic batch formation
- Request prioritization

## Sovereign AI Stack Equivalent

Maps to `realizar::batch` for inference batching.

## Why Continuous Batching?

Traditional static batching:
1. Wait for N requests
2. Process all together
3. Return all results

**Problem**: High latency for early arrivals, GPU idle while waiting.

Continuous batching:
1. Start inference immediately
2. Dynamically add new requests
3. Remove completed sequences
4. **2-4x higher throughput**

## Key Components

### BatchConfig

```rust
use tgi_gtc::batching::BatchConfig;

let config = BatchConfig::builder()
    .max_batch_size(32)       // Max requests per batch
    .max_batch_tokens(4096)   // Token budget (prefill)
    .min_batch_size(1)        // Don't wait if >= this
    .max_wait_ms(50)          // Max wait before forcing
    .build();
```

### BatchRequest

```rust
use tgi_gtc::batching::BatchRequest;

let request = BatchRequest::new(
    id,           // Unique ID
    input_tokens, // Prompt length
    max_new_tokens // Generation limit
).with_priority(10);
```

### ContinuousBatcher

```rust
use tgi_gtc::batching::ContinuousBatcher;

let batcher = ContinuousBatcher::new(config);

// Add requests
batcher.add(request1);
batcher.add(request2);

// Form batch when ready
if let Some(batch) = batcher.try_form_batch() {
    // Process batch
    for request in batch.requests {
        // Run inference
    }
}
```

### Batch

Formed batch with metadata:

```rust
let batch = batcher.try_form_batch().unwrap();
batch.size()              // Number of requests
batch.total_input_tokens  // Sum of input tokens
batch.avg_wait_time()     // Average queue time
batch.max_wait_time()     // Longest wait
```

## Batching Algorithm

1. **Check queue**: Skip if empty
2. **Check conditions**:
   - `queue.len() >= min_batch_size`, OR
   - `oldest_request.wait_time >= max_wait_ms`
3. **Collect requests** until:
   - `batch.size() >= max_batch_size`, OR
   - `batch.tokens >= max_batch_tokens`
4. **Return batch** for processing

## Prefill vs Decode

### Prefill Phase
- Process input tokens
- **Compute-bound** (parallelizable)
- Benefits from large batches

### Decode Phase
- Generate output tokens
- **Memory-bound** (sequential)
- KV cache access dominates

TGI separates these phases for optimal scheduling.

## Example

```bash
cargo run --example continuous_batching
```
