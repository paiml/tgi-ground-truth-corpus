//! Chunked Prefill
//!
//! Implements chunked prefill for processing long prompts in smaller chunks
//! to avoid out-of-memory errors and enable better batching.
//!
//! # TGI Reference
//!
//! Based on TGI's chunked prefill implementation.
//! See: <https://github.com/huggingface/text-generation-inference>
//!
//! # Algorithm
//!
//! 1. Split long prompt into fixed-size chunks
//! 2. Process chunks sequentially or interleaved with decode steps
//! 3. Accumulate KV cache across chunks
//! 4. Start generation after all chunks are processed
//!
//! # Example
//!
//! ```rust
//! use tgi_gtc::chunked_prefill::{ChunkedPrefill, ChunkConfig};
//!
//! let config = ChunkConfig::default();
//! let prefill = ChunkedPrefill::new(config);
//!
//! let long_prompt = vec![1u32; 10000];
//! let chunks = prefill.split_into_chunks(&long_prompt);
//!
//! println!("Split into {} chunks", chunks.len());
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for chunked prefill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Maximum tokens per chunk.
    pub chunk_size: usize,

    /// Overlap between chunks (for context continuity).
    pub overlap: usize,

    /// Maximum total prompt length.
    pub max_prompt_length: usize,

    /// Whether to pad chunks to uniform size.
    pub pad_to_chunk_size: bool,

    /// Interleave prefill with decode (disaggregated serving).
    pub interleave_with_decode: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 0,
            max_prompt_length: 32768,
            pad_to_chunk_size: false,
            interleave_with_decode: false,
        }
    }
}

impl ChunkConfig {
    /// Create config for short context models.
    pub fn short_context() -> Self {
        Self {
            chunk_size: 256,
            max_prompt_length: 4096,
            ..Default::default()
        }
    }

    /// Create config for long context models.
    pub fn long_context() -> Self {
        Self {
            chunk_size: 2048,
            max_prompt_length: 128000,
            ..Default::default()
        }
    }

    /// Create config for disaggregated serving.
    pub fn disaggregated() -> Self {
        Self {
            chunk_size: 256,
            interleave_with_decode: true,
            ..Default::default()
        }
    }
}

/// A chunk of the prompt to be processed.
#[derive(Debug, Clone)]
pub struct PrefillChunk {
    /// Token IDs for this chunk.
    pub tokens: Vec<u32>,

    /// Starting position in the original prompt.
    pub start_pos: usize,

    /// Ending position in the original prompt.
    pub end_pos: usize,

    /// Chunk index (0-based).
    pub chunk_index: usize,

    /// Total number of chunks.
    pub total_chunks: usize,

    /// Whether this is the last chunk.
    pub is_last: bool,
}

impl PrefillChunk {
    /// Length of this chunk.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        if self.total_chunks == 0 {
            1.0
        } else {
            (self.chunk_index + 1) as f32 / self.total_chunks as f32
        }
    }
}

/// State of chunked prefill processing.
#[derive(Debug, Clone)]
pub struct PrefillState {
    /// Request ID.
    pub request_id: u64,

    /// Total prompt length.
    pub total_length: usize,

    /// Tokens processed so far.
    pub processed_tokens: usize,

    /// Current chunk index.
    pub current_chunk: usize,

    /// Total number of chunks.
    pub total_chunks: usize,

    /// Whether prefill is complete.
    pub is_complete: bool,

    /// Accumulated KV cache block IDs.
    pub block_ids: Vec<u32>,
}

impl PrefillState {
    /// Create a new prefill state.
    pub fn new(request_id: u64, total_length: usize, total_chunks: usize) -> Self {
        Self {
            request_id,
            total_length,
            processed_tokens: 0,
            current_chunk: 0,
            total_chunks,
            is_complete: false,
            block_ids: Vec::new(),
        }
    }

    /// Update state after processing a chunk.
    pub fn update(&mut self, chunk: &PrefillChunk, block_ids: Vec<u32>) {
        self.processed_tokens += chunk.len();
        self.current_chunk = chunk.chunk_index + 1;
        self.block_ids.extend(block_ids);
        self.is_complete = chunk.is_last;
    }

    /// Progress as a fraction.
    pub fn progress(&self) -> f32 {
        if self.total_length == 0 {
            1.0
        } else {
            self.processed_tokens as f32 / self.total_length as f32
        }
    }

    /// Remaining tokens to process.
    pub fn remaining_tokens(&self) -> usize {
        self.total_length.saturating_sub(self.processed_tokens)
    }
}

/// Chunked prefill processor.
#[derive(Debug)]
pub struct ChunkedPrefill {
    config: ChunkConfig,
}

impl ChunkedPrefill {
    /// Create a new chunked prefill processor.
    pub fn new(config: ChunkConfig) -> Self {
        Self { config }
    }

    /// Get configuration.
    pub fn config(&self) -> &ChunkConfig {
        &self.config
    }

    /// Split a prompt into chunks.
    pub fn split_into_chunks(&self, tokens: &[u32]) -> Vec<PrefillChunk> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let effective_chunk_size = self.config.chunk_size.saturating_sub(self.config.overlap);
        if effective_chunk_size == 0 {
            return vec![PrefillChunk {
                tokens: tokens.to_vec(),
                start_pos: 0,
                end_pos: tokens.len(),
                chunk_index: 0,
                total_chunks: 1,
                is_last: true,
            }];
        }

        let mut chunks = Vec::new();
        let mut pos = 0;

        while pos < tokens.len() {
            let end = (pos + self.config.chunk_size).min(tokens.len());
            let chunk_tokens = tokens[pos..end].to_vec();
            let is_last = end >= tokens.len();

            chunks.push(PrefillChunk {
                tokens: chunk_tokens,
                start_pos: pos,
                end_pos: end,
                chunk_index: chunks.len(),
                total_chunks: 0, // Will be updated
                is_last,
            });

            // Break if this was the last chunk
            if is_last {
                break;
            }

            // Advance position, accounting for overlap
            pos = end.saturating_sub(self.config.overlap);
        }

        // Update total_chunks
        let total = chunks.len();
        for chunk in &mut chunks {
            chunk.total_chunks = total;
        }

        chunks
    }

    /// Calculate number of chunks needed for a prompt.
    pub fn num_chunks(&self, prompt_length: usize) -> usize {
        if prompt_length == 0 {
            return 0;
        }

        let effective_chunk_size = self.config.chunk_size.saturating_sub(self.config.overlap);
        if effective_chunk_size == 0 {
            return 1;
        }

        (prompt_length + effective_chunk_size - 1) / effective_chunk_size
    }

    /// Check if a prompt needs chunking.
    pub fn needs_chunking(&self, prompt_length: usize) -> bool {
        prompt_length > self.config.chunk_size
    }

    /// Validate prompt length.
    pub fn validate_length(&self, prompt_length: usize) -> Result<(), ChunkError> {
        if prompt_length > self.config.max_prompt_length {
            Err(ChunkError::PromptTooLong {
                length: prompt_length,
                max: self.config.max_prompt_length,
            })
        } else {
            Ok(())
        }
    }
}

/// Error type for chunked prefill.
#[allow(missing_docs)]
#[derive(Debug, Clone, thiserror::Error)]
pub enum ChunkError {
    /// Prompt exceeds maximum length.
    #[error("Prompt length {length} exceeds maximum {max}")]
    PromptTooLong { length: usize, max: usize },

    /// Chunk processing failed.
    #[error("Chunk {index} processing failed: {reason}")]
    ChunkFailed { index: usize, reason: String },
}

/// Scheduler for chunked prefill with decode interleaving.
#[derive(Debug)]
pub struct ChunkScheduler {
    config: ChunkConfig,
    pending_prefills: Vec<PrefillState>,
    prefill_budget: usize,
    decode_budget: usize,
}

impl ChunkScheduler {
    /// Create a new chunk scheduler.
    pub fn new(config: ChunkConfig, prefill_budget: usize, decode_budget: usize) -> Self {
        Self {
            config,
            pending_prefills: Vec::new(),
            prefill_budget,
            decode_budget,
        }
    }

    /// Add a new prefill request.
    pub fn add_prefill(&mut self, request_id: u64, tokens: &[u32]) {
        let prefill = ChunkedPrefill::new(self.config.clone());
        let num_chunks = prefill.num_chunks(tokens.len());
        let state = PrefillState::new(request_id, tokens.len(), num_chunks);
        self.pending_prefills.push(state);
    }

    /// Get next batch of work (prefill chunks + decode tokens).
    pub fn schedule(&mut self) -> ScheduledBatch {
        let mut batch = ScheduledBatch::default();
        let mut remaining_budget = self.prefill_budget;

        // Schedule prefill chunks
        for state in &mut self.pending_prefills {
            if state.is_complete {
                continue;
            }

            let chunk_size = self.config.chunk_size.min(state.remaining_tokens());
            if chunk_size <= remaining_budget {
                batch.prefill_requests.push(state.request_id);
                batch.prefill_tokens += chunk_size;
                remaining_budget -= chunk_size;
            }
        }

        // Remove completed prefills
        self.pending_prefills.retain(|s| !s.is_complete);

        // Decode budget is separate
        batch.decode_budget = self.decode_budget;

        batch
    }

    /// Number of pending prefill requests.
    pub fn pending_count(&self) -> usize {
        self.pending_prefills.len()
    }
}

/// A scheduled batch of work.
#[derive(Debug, Clone, Default)]
pub struct ScheduledBatch {
    /// Request IDs for prefill.
    pub prefill_requests: Vec<u64>,

    /// Total prefill tokens in this batch.
    pub prefill_tokens: usize,

    /// Available budget for decode tokens.
    pub decode_budget: usize,
}

impl ScheduledBatch {
    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.prefill_requests.is_empty() && self.decode_budget == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.overlap, 0);
    }

    #[test]
    fn test_split_into_chunks_small() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 0,
            ..Default::default()
        };
        let prefill = ChunkedPrefill::new(config);

        let tokens: Vec<u32> = (0..50).collect();
        let chunks = prefill.split_into_chunks(&tokens);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 50);
        assert!(chunks[0].is_last);
    }

    #[test]
    fn test_split_into_chunks_exact() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 0,
            ..Default::default()
        };
        let prefill = ChunkedPrefill::new(config);

        let tokens: Vec<u32> = (0..300).collect();
        let chunks = prefill.split_into_chunks(&tokens);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 100);
        assert_eq!(chunks[1].len(), 100);
        assert_eq!(chunks[2].len(), 100);
        assert!(!chunks[0].is_last);
        assert!(!chunks[1].is_last);
        assert!(chunks[2].is_last);
    }

    #[test]
    fn test_split_into_chunks_with_overlap() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 20,
            ..Default::default()
        };
        let prefill = ChunkedPrefill::new(config);

        let tokens: Vec<u32> = (0..200).collect();
        let chunks = prefill.split_into_chunks(&tokens);

        // With overlap, chunks should overlap
        assert!(chunks.len() >= 2);

        // First chunk ends at 100
        assert_eq!(chunks[0].end_pos, 100);

        // Second chunk starts at 80 (100 - 20 overlap)
        assert_eq!(chunks[1].start_pos, 80);
    }

    #[test]
    fn test_num_chunks() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 0,
            ..Default::default()
        };
        let prefill = ChunkedPrefill::new(config);

        assert_eq!(prefill.num_chunks(0), 0);
        assert_eq!(prefill.num_chunks(50), 1);
        assert_eq!(prefill.num_chunks(100), 1);
        assert_eq!(prefill.num_chunks(101), 2);
        assert_eq!(prefill.num_chunks(250), 3);
    }

    #[test]
    fn test_needs_chunking() {
        let config = ChunkConfig {
            chunk_size: 100,
            ..Default::default()
        };
        let prefill = ChunkedPrefill::new(config);

        assert!(!prefill.needs_chunking(50));
        assert!(!prefill.needs_chunking(100));
        assert!(prefill.needs_chunking(101));
    }

    #[test]
    fn test_validate_length() {
        let config = ChunkConfig {
            max_prompt_length: 1000,
            ..Default::default()
        };
        let prefill = ChunkedPrefill::new(config);

        assert!(prefill.validate_length(500).is_ok());
        assert!(prefill.validate_length(1000).is_ok());
        assert!(prefill.validate_length(1001).is_err());
    }

    #[test]
    fn test_prefill_state() {
        let mut state = PrefillState::new(1, 1000, 10);

        assert_eq!(state.progress(), 0.0);
        assert_eq!(state.remaining_tokens(), 1000);
        assert!(!state.is_complete);

        let chunk = PrefillChunk {
            tokens: vec![0; 100],
            start_pos: 0,
            end_pos: 100,
            chunk_index: 0,
            total_chunks: 10,
            is_last: false,
        };

        state.update(&chunk, vec![1, 2]);

        assert_eq!(state.processed_tokens, 100);
        assert_eq!(state.current_chunk, 1);
        assert_eq!(state.block_ids, vec![1, 2]);
    }

    #[test]
    fn test_chunk_scheduler() {
        let config = ChunkConfig {
            chunk_size: 100,
            ..Default::default()
        };
        let mut scheduler = ChunkScheduler::new(config, 500, 100);

        scheduler.add_prefill(1, &vec![0u32; 250]);
        scheduler.add_prefill(2, &vec![0u32; 150]);

        assert_eq!(scheduler.pending_count(), 2);

        let batch = scheduler.schedule();
        assert!(!batch.prefill_requests.is_empty());
        assert!(batch.prefill_tokens <= 500);
    }

    #[test]
    fn test_chunk_progress() {
        let chunk = PrefillChunk {
            tokens: vec![],
            start_pos: 0,
            end_pos: 100,
            chunk_index: 4,
            total_chunks: 10,
            is_last: false,
        };

        assert_eq!(chunk.progress(), 0.5);
    }
}
