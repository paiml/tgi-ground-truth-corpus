//! SafeTensors Inference Example
//!
//! Demonstrates real inference using:
//! - safetensors for model weight loading
//! - tgi-gtc patterns for batching and streaming
//!
//! # Run
//!
//! ```bash
//! cargo run --example safetensors_inference --features inference
//! ```
//!
//! # What This Demonstrates
//!
//! 1. Creating a simple MLP model
//! 2. Saving/loading weights via safetensors
//! 3. Batched inference using ContinuousBatcher
//! 4. Streaming results via SseFormatter
//! 5. CPU computation with proper memory layout

use safetensors::tensor::{SafeTensors, TensorView};
use safetensors::serialize;
use std::collections::HashMap;
use std::time::Instant;

use tgi_gtc::batching::{BatchConfig, BatchRequest, ContinuousBatcher};
use tgi_gtc::streaming::{CompleteEvent, FinishReason, SseFormatter, TokenEvent};

/// Simple MLP layer weights
struct MlpWeights {
    /// Weight matrix [hidden, input]
    w1: Vec<f32>,
    /// Bias vector [hidden]
    b1: Vec<f32>,
    /// Output weights [output, hidden]
    w2: Vec<f32>,
    /// Output bias [output]
    b2: Vec<f32>,
    /// Dimensions
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl MlpWeights {
    /// Create random weights for demo
    fn random(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        use std::f32::consts::PI;

        // Simple pseudo-random initialization (Xavier-like)
        let scale1 = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale2 = (2.0 / (hidden_dim + output_dim) as f32).sqrt();

        let w1: Vec<f32> = (0..hidden_dim * input_dim)
            .map(|i| (i as f32 * PI).sin() * scale1)
            .collect();
        let b1 = vec![0.0; hidden_dim];

        let w2: Vec<f32> = (0..output_dim * hidden_dim)
            .map(|i| (i as f32 * PI * 1.5).sin() * scale2)
            .collect();
        let b2 = vec![0.0; output_dim];

        Self {
            w1, b1, w2, b2,
            input_dim, hidden_dim, output_dim,
        }
    }

    /// Convert f32 slice to bytes
    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    /// Convert bytes to f32 vec
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Save to safetensors format
    fn to_safetensors(&self) -> Vec<u8> {
        let w1_bytes = Self::f32_to_bytes(&self.w1);
        let b1_bytes = Self::f32_to_bytes(&self.b1);
        let w2_bytes = Self::f32_to_bytes(&self.w2);
        let b2_bytes = Self::f32_to_bytes(&self.b2);

        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();

        tensors.insert(
            "w1".to_string(),
            TensorView::new(
                safetensors::Dtype::F32,
                vec![self.hidden_dim, self.input_dim],
                &w1_bytes,
            ).unwrap(),
        );
        tensors.insert(
            "b1".to_string(),
            TensorView::new(
                safetensors::Dtype::F32,
                vec![self.hidden_dim],
                &b1_bytes,
            ).unwrap(),
        );
        tensors.insert(
            "w2".to_string(),
            TensorView::new(
                safetensors::Dtype::F32,
                vec![self.output_dim, self.hidden_dim],
                &w2_bytes,
            ).unwrap(),
        );
        tensors.insert(
            "b2".to_string(),
            TensorView::new(
                safetensors::Dtype::F32,
                vec![self.output_dim],
                &b2_bytes,
            ).unwrap(),
        );

        serialize(tensors, None).unwrap()
    }

    /// Load from safetensors bytes
    fn from_safetensors(data: &[u8]) -> Self {
        let tensors = SafeTensors::deserialize(data).unwrap();

        let w1_tensor = tensors.tensor("w1").unwrap();
        let b1_tensor = tensors.tensor("b1").unwrap();
        let w2_tensor = tensors.tensor("w2").unwrap();
        let b2_tensor = tensors.tensor("b2").unwrap();

        let w1 = Self::bytes_to_f32(w1_tensor.data());
        let b1 = Self::bytes_to_f32(b1_tensor.data());
        let w2 = Self::bytes_to_f32(w2_tensor.data());
        let b2 = Self::bytes_to_f32(b2_tensor.data());

        let hidden_dim = w1_tensor.shape()[0];
        let input_dim = w1_tensor.shape()[1];
        let output_dim = w2_tensor.shape()[0];

        Self {
            w1, b1, w2, b2,
            input_dim, hidden_dim, output_dim,
        }
    }
}

/// Simple MLP forward pass (CPU, SIMD-friendly layout)
fn mlp_forward(weights: &MlpWeights, input: &[f32]) -> Vec<f32> {
    assert_eq!(input.len(), weights.input_dim);

    // Layer 1: hidden = ReLU(input @ W1.T + b1)
    let mut hidden = vec![0.0f32; weights.hidden_dim];
    for h in 0..weights.hidden_dim {
        let mut sum = weights.b1[h];
        for i in 0..weights.input_dim {
            sum += input[i] * weights.w1[h * weights.input_dim + i];
        }
        // ReLU activation
        hidden[h] = sum.max(0.0);
    }

    // Layer 2: output = hidden @ W2.T + b2
    let mut output = vec![0.0f32; weights.output_dim];
    for o in 0..weights.output_dim {
        let mut sum = weights.b2[o];
        for h in 0..weights.hidden_dim {
            sum += hidden[h] * weights.w2[o * weights.hidden_dim + h];
        }
        output[o] = sum;
    }

    // Softmax for probability distribution
    let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = output.iter().map(|x| (x - max_val).exp()).sum();
    for o in output.iter_mut() {
        *o = (*o - max_val).exp() / exp_sum;
    }

    output
}

/// Batched MLP forward pass
fn mlp_forward_batch(weights: &MlpWeights, batch: &[Vec<f32>]) -> Vec<Vec<f32>> {
    batch.iter().map(|input| mlp_forward(weights, input)).collect()
}

/// Simulated token generation using MLP as a tiny "language model"
fn generate_tokens(
    weights: &MlpWeights,
    input: &[f32],
    max_tokens: usize,
) -> Vec<(u32, f32)> {
    let mut tokens = Vec::new();
    let mut current_input = input.to_vec();

    for _ in 0..max_tokens {
        let probs = mlp_forward(weights, &current_input);

        // Argmax sampling (greedy)
        let (token_id, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        tokens.push((token_id as u32, prob));

        // Feed output back as input (autoregressive)
        // Pad/truncate to input_dim
        current_input = probs.clone();
        current_input.resize(weights.input_dim, 0.0);

        // Stop if we hit "EOS" (token 0 with high confidence)
        if token_id == 0 && prob > 0.5 {
            break;
        }
    }

    tokens
}

fn main() {
    println!("=== SafeTensors Inference Demo ===\n");

    // Model dimensions (tiny for demo)
    let input_dim = 64;
    let hidden_dim = 128;
    let output_dim = 32; // "vocabulary size"

    println!("Model Architecture:");
    println!("  Input:  {} dimensions", input_dim);
    println!("  Hidden: {} dimensions", hidden_dim);
    println!("  Output: {} dimensions (vocab)", output_dim);
    println!();

    // Create and save model
    println!("Creating random MLP weights...");
    let weights = MlpWeights::random(input_dim, hidden_dim, output_dim);

    let safetensors_data = weights.to_safetensors();
    println!("Serialized to safetensors: {} bytes", safetensors_data.len());

    // Load model back (simulates loading from disk)
    println!("Loading model from safetensors...");
    let loaded_weights = MlpWeights::from_safetensors(&safetensors_data);
    println!("  w1: [{}, {}]", loaded_weights.hidden_dim, loaded_weights.input_dim);
    println!("  w2: [{}, {}]", loaded_weights.output_dim, loaded_weights.hidden_dim);
    println!();

    // Setup batching using our patterns
    let batch_config = BatchConfig::builder()
        .max_batch_size(4)
        .max_batch_tokens(1024)
        .min_batch_size(1)
        .build();

    let batcher = ContinuousBatcher::new(batch_config);

    println!("Batching Configuration:");
    println!("  Max batch size: {}", batcher.config().max_batch_size);
    println!("  Max batch tokens: {}", batcher.config().max_batch_tokens);
    println!();

    // Create sample requests
    let requests: Vec<Vec<f32>> = (0..6)
        .map(|i| {
            // Random-ish input vectors
            (0..input_dim)
                .map(|j| ((i * 17 + j * 13) as f32 * 0.01).sin())
                .collect()
        })
        .collect();

    // Add to batcher
    println!("Adding {} inference requests to batcher...", requests.len());
    for _ in 0..requests.len() {
        let req = BatchRequest::new(batcher.next_id(), input_dim, 10);
        batcher.add(req);
    }

    // Process batches
    let formatter = SseFormatter::new();

    println!("\n--- Batched Inference ---\n");

    let start = Instant::now();
    let mut total_tokens = 0;
    let mut batch_num = 0;

    while let Some(batch) = batcher.force_batch() {
        batch_num += 1;
        println!("Processing batch {} ({} requests)...", batch_num, batch.size());

        // Get inputs for this batch
        let batch_inputs: Vec<Vec<f32>> = batch
            .requests
            .iter()
            .map(|r| requests[(r.id - 1) as usize].clone())
            .collect();

        // Run batched forward pass
        let batch_start = Instant::now();
        let _outputs = mlp_forward_batch(&loaded_weights, &batch_inputs);
        let batch_time = batch_start.elapsed();

        // Generate tokens for each request in batch
        for (i, req) in batch.requests.iter().enumerate() {
            let tokens = generate_tokens(&loaded_weights, &batch_inputs[i], 5);
            total_tokens += tokens.len();

            // Stream tokens
            print!("  Request {}: ", req.id);
            for (token_id, prob) in &tokens {
                print!("[{}:{:.2}] ", token_id, prob);
            }
            println!();
        }

        println!("  Batch time: {:?}\n", batch_time);
    }

    let total_time = start.elapsed();

    // Final metrics
    println!("--- Results ---\n");
    println!("Total batches: {}", batch_num);
    println!("Total tokens: {}", total_tokens);
    println!("Total time: {:?}", total_time);
    println!(
        "Throughput: {:.1} tokens/sec",
        total_tokens as f64 / total_time.as_secs_f64()
    );

    // Show SSE format example
    println!("\n--- SSE Output Format Example ---\n");

    let sample_tokens = generate_tokens(&loaded_weights, &requests[0], 3);
    for (token_id, prob) in &sample_tokens {
        let event = TokenEvent {
            token: format!("token_{}", token_id),
            token_id: *token_id,
            logprob: Some((prob.ln() * 100.0) as i32),
            special: false,
        };
        print!("{}", formatter.format_token(&event));
    }

    let complete = CompleteEvent {
        generated_tokens: sample_tokens.len(),
        finish_reason: FinishReason::Length,
        generation_time_ms: total_time.as_millis() as u64,
    };
    print!("{}", formatter.format_complete(&complete));

    println!("\n=== Demo Complete ===");
}
