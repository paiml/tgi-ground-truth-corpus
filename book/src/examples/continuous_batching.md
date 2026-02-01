# Continuous Batching Example

Demonstrates TGI's continuous batching algorithm for optimal GPU utilization.

## Run

```bash
cargo run --example continuous_batching
```

## Output

```
=== Continuous Batching Demo ===

Batcher Configuration:
  Max batch size: 8
  Max batch tokens: 2048
  Min batch size: 2
  Max wait: 50ms

Simulating incoming requests...

  Added request 1: Short prompt (50 input, 100 max new)
  Added request 2: Medium prompt (150 input, 200 max new)
  Added request 3: Long prompt (300 input, 150 max new)
  Added request 4: Another short (75 input, 50 max new)
  Added request 5: Complex query (200 input, 300 max new)
  Added request 6: Simple ask (25 input, 50 max new)

Queue length: 6

Forming batches...

Batch 1:
  Requests: 6
  Total input tokens: 800
  Request IDs: [1, 2, 3, 4, 5, 6]

=== Demo Complete ===
```

## Code Walkthrough

### 1. Configure Batcher

```rust
let config = BatchConfig::builder()
    .max_batch_size(8)
    .max_batch_tokens(2048)
    .min_batch_size(2)
    .max_wait_ms(50)
    .build();
```

### 2. Add Requests

```rust
let batcher = ContinuousBatcher::new(config);

let request = BatchRequest::new(id, input_tokens, max_new_tokens);
batcher.add(request);
```

### 3. Form Batches

```rust
while let Some(batch) = batcher.try_form_batch() {
    println!("Batch size: {}", batch.size());
    println!("Total tokens: {}", batch.total_input_tokens);

    for request in batch.requests {
        // Process each request
    }
}
```
