//! KV Cache and Block Allocation patterns.
//!
//! Memory-efficient KV cache management using block-based allocation,
//! inspired by PagedAttention for optimal GPU memory utilization.
//!
//! # TGI Source
//!
//! Patterns derived from `backends/v3/src/block_allocator.rs`:
//! - Block-based KV cache allocation
//! - Memory pool management
//! - Copy-on-write for shared prefixes
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `realizar::kv_cache` for KV cache management.
//!
//! # Key Concepts
//!
//! ## Why Block-Based Allocation?
//!
//! Traditional KV cache allocation wastes memory:
//! - Allocate max_seq_len for every request
//! - Most requests use much less
//! - Memory fragmentation
//!
//! Block-based allocation (PagedAttention):
//! - Divide memory into fixed-size blocks
//! - Allocate blocks on-demand as sequence grows
//! - Reclaim blocks when sequence completes
//! - **2-4x more concurrent requests**

use crate::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque};

/// Block size for KV cache allocation.
/// Typically 16 tokens per block for good balance.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// A single block of KV cache memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Create a new block ID.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    pub const fn value(&self) -> u32 {
        self.0
    }
}

/// Block allocation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockState {
    /// Block is free and available.
    Free,
    /// Block is allocated to a sequence.
    Allocated,
    /// Block is shared (copy-on-write).
    Shared(u32), // Reference count
}

impl BlockState {
    /// Check if block is free.
    pub const fn is_free(&self) -> bool {
        matches!(self, Self::Free)
    }

    /// Check if block is allocated.
    pub const fn is_allocated(&self) -> bool {
        matches!(self, Self::Allocated | Self::Shared(_))
    }

    /// Get reference count for shared blocks.
    pub const fn ref_count(&self) -> u32 {
        match self {
            Self::Free => 0,
            Self::Allocated => 1,
            Self::Shared(count) => *count,
        }
    }
}

/// Configuration for the block allocator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockAllocatorConfig {
    /// Total number of blocks available.
    pub num_blocks: usize,

    /// Tokens per block.
    pub block_size: usize,

    /// Number of layers in the model.
    pub num_layers: usize,

    /// Number of KV heads.
    pub num_kv_heads: usize,

    /// Head dimension.
    pub head_dim: usize,
}

impl Default for BlockAllocatorConfig {
    fn default() -> Self {
        Self {
            num_blocks: 1024,
            block_size: DEFAULT_BLOCK_SIZE,
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
        }
    }
}

impl BlockAllocatorConfig {
    /// Create a new config with specified blocks.
    pub const fn with_blocks(num_blocks: usize) -> Self {
        Self {
            num_blocks,
            block_size: DEFAULT_BLOCK_SIZE,
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
        }
    }

    /// Set block size.
    pub const fn block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Set number of layers.
    pub const fn num_layers(mut self, layers: usize) -> Self {
        self.num_layers = layers;
        self
    }

    /// Set number of KV heads.
    pub const fn num_kv_heads(mut self, heads: usize) -> Self {
        self.num_kv_heads = heads;
        self
    }

    /// Set head dimension.
    pub const fn head_dim(mut self, dim: usize) -> Self {
        self.head_dim = dim;
        self
    }

    /// Calculate bytes per block.
    pub const fn bytes_per_block(&self) -> usize {
        // K and V for each layer, each head
        // 2 (K+V) * num_layers * num_kv_heads * head_dim * block_size * sizeof(f16)
        2 * self.num_layers * self.num_kv_heads * self.head_dim * self.block_size * 2
    }

    /// Calculate total memory in bytes.
    pub const fn total_memory_bytes(&self) -> usize {
        self.num_blocks * self.bytes_per_block()
    }
}

/// A sequence's block table mapping logical to physical blocks.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    /// Physical block IDs for this sequence.
    blocks: Vec<BlockId>,

    /// Number of tokens currently stored.
    num_tokens: usize,

    /// Block size for calculating positions.
    block_size: usize,
}

impl BlockTable {
    /// Create a new empty block table.
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    /// Get the number of blocks allocated.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the number of tokens stored.
    pub const fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get block IDs.
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Add a block to the table.
    pub fn add_block(&mut self, block_id: BlockId) {
        self.blocks.push(block_id);
    }

    /// Remove and return the last block.
    pub fn pop_block(&mut self) -> Option<BlockId> {
        self.blocks.pop()
    }

    /// Set number of tokens.
    pub fn set_num_tokens(&mut self, num_tokens: usize) {
        self.num_tokens = num_tokens;
    }

    /// Calculate blocks needed for a given number of tokens.
    pub fn blocks_needed(&self, num_tokens: usize) -> usize {
        (num_tokens + self.block_size - 1) / self.block_size
    }

    /// Check if we need more blocks for additional tokens.
    pub fn needs_more_blocks(&self, additional_tokens: usize) -> bool {
        let total_tokens = self.num_tokens + additional_tokens;
        self.blocks_needed(total_tokens) > self.blocks.len()
    }

    /// Get the physical block and offset for a token position.
    pub fn get_block_and_offset(&self, token_pos: usize) -> Option<(BlockId, usize)> {
        let block_idx = token_pos / self.block_size;
        let offset = token_pos % self.block_size;

        self.blocks
            .get(block_idx)
            .map(|&block_id| (block_id, offset))
    }
}

/// Block allocator for KV cache management.
///
/// # TGI Source
///
/// Maps to `BlockAllocator` in `backends/v3/src/block_allocator.rs`.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::kv_cache::{BlockAllocator, BlockAllocatorConfig};
///
/// let config = BlockAllocatorConfig::with_blocks(100);
/// let mut allocator = BlockAllocator::new(config);
///
/// // Allocate blocks for a sequence
/// let seq_id = 1;
/// allocator.allocate(seq_id, 5).unwrap(); // 5 blocks
///
/// // Check allocation
/// assert_eq!(allocator.get_block_table(seq_id).unwrap().num_blocks(), 5);
///
/// // Free when done
/// allocator.free(seq_id);
/// ```
#[derive(Debug)]
pub struct BlockAllocator {
    config: BlockAllocatorConfig,

    /// State of each block.
    block_states: Vec<BlockState>,

    /// Free block list for O(1) allocation.
    free_blocks: VecDeque<BlockId>,

    /// Block tables for each sequence.
    sequence_tables: HashMap<u64, BlockTable>,

    /// Sequences sharing each block (for CoW).
    block_sequences: HashMap<BlockId, HashSet<u64>>,
}

impl BlockAllocator {
    /// Create a new block allocator.
    pub fn new(config: BlockAllocatorConfig) -> Self {
        let num_blocks = config.num_blocks;

        let block_states = vec![BlockState::Free; num_blocks];
        let free_blocks: VecDeque<BlockId> = (0..num_blocks as u32).map(BlockId::new).collect();

        Self {
            config,
            block_states,
            free_blocks,
            sequence_tables: HashMap::new(),
            block_sequences: HashMap::new(),
        }
    }

    /// Get configuration.
    pub const fn config(&self) -> &BlockAllocatorConfig {
        &self.config
    }

    /// Get number of free blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get number of allocated blocks.
    pub fn num_allocated_blocks(&self) -> usize {
        self.config.num_blocks - self.free_blocks.len()
    }

    /// Check if we can allocate n blocks.
    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.free_blocks.len() >= num_blocks
    }

    /// Allocate blocks for a new sequence.
    pub fn allocate(&mut self, seq_id: u64, num_blocks: usize) -> Result<()> {
        if !self.can_allocate(num_blocks) {
            return Err(Error::resource_exhausted(format!(
                "cannot allocate {} blocks, only {} free",
                num_blocks,
                self.free_blocks.len()
            )));
        }

        let mut table = BlockTable::new(self.config.block_size);

        for _ in 0..num_blocks {
            let block_id = self.free_blocks.pop_front().unwrap();
            self.block_states[block_id.0 as usize] = BlockState::Allocated;
            table.add_block(block_id);

            self.block_sequences
                .entry(block_id)
                .or_default()
                .insert(seq_id);
        }

        self.sequence_tables.insert(seq_id, table);
        Ok(())
    }

    /// Allocate additional blocks for an existing sequence.
    pub fn allocate_more(&mut self, seq_id: u64, additional_blocks: usize) -> Result<()> {
        if !self.can_allocate(additional_blocks) {
            return Err(Error::resource_exhausted(format!(
                "cannot allocate {} more blocks, only {} free",
                additional_blocks,
                self.free_blocks.len()
            )));
        }

        let table = self
            .sequence_tables
            .get_mut(&seq_id)
            .ok_or_else(|| Error::validation(format!("sequence {} not found", seq_id)))?;

        for _ in 0..additional_blocks {
            let block_id = self.free_blocks.pop_front().unwrap();
            self.block_states[block_id.0 as usize] = BlockState::Allocated;
            table.add_block(block_id);

            self.block_sequences
                .entry(block_id)
                .or_default()
                .insert(seq_id);
        }

        Ok(())
    }

    /// Free all blocks for a sequence.
    pub fn free(&mut self, seq_id: u64) -> Vec<BlockId> {
        let mut freed = Vec::new();

        if let Some(table) = self.sequence_tables.remove(&seq_id) {
            for block_id in table.blocks() {
                // Remove sequence from block's set
                if let Some(seqs) = self.block_sequences.get_mut(block_id) {
                    seqs.remove(&seq_id);

                    // If no more sequences use this block, free it
                    if seqs.is_empty() {
                        self.block_sequences.remove(block_id);
                        self.block_states[block_id.0 as usize] = BlockState::Free;
                        self.free_blocks.push_back(*block_id);
                        freed.push(*block_id);
                    } else {
                        // Update reference count for shared blocks
                        let count = seqs.len() as u32;
                        self.block_states[block_id.0 as usize] = if count == 1 {
                            BlockState::Allocated
                        } else {
                            BlockState::Shared(count)
                        };
                    }
                }
            }
        }

        freed
    }

    /// Get block table for a sequence.
    pub fn get_block_table(&self, seq_id: u64) -> Option<&BlockTable> {
        self.sequence_tables.get(&seq_id)
    }

    /// Get mutable block table for a sequence.
    pub fn get_block_table_mut(&mut self, seq_id: u64) -> Option<&mut BlockTable> {
        self.sequence_tables.get_mut(&seq_id)
    }

    /// Fork a sequence (copy-on-write for shared prefix).
    ///
    /// Creates a new sequence that shares blocks with the parent.
    pub fn fork(&mut self, parent_seq_id: u64, child_seq_id: u64) -> Result<()> {
        let parent_table = self
            .sequence_tables
            .get(&parent_seq_id)
            .ok_or_else(|| {
                Error::validation(format!("parent sequence {} not found", parent_seq_id))
            })?
            .clone();

        // Share all blocks with the child
        for block_id in parent_table.blocks() {
            // Update block state to shared
            let state = &mut self.block_states[block_id.0 as usize];
            *state = match *state {
                BlockState::Allocated => BlockState::Shared(2),
                BlockState::Shared(n) => BlockState::Shared(n + 1),
                BlockState::Free => {
                    return Err(Error::internal("cannot fork with free block"));
                }
            };

            // Add child to block's sequence set
            self.block_sequences
                .entry(*block_id)
                .or_default()
                .insert(child_seq_id);
        }

        self.sequence_tables.insert(child_seq_id, parent_table);
        Ok(())
    }

    /// Get block state.
    pub fn block_state(&self, block_id: BlockId) -> BlockState {
        self.block_states
            .get(block_id.0 as usize)
            .copied()
            .unwrap_or(BlockState::Free)
    }

    /// Get memory usage statistics.
    pub fn memory_stats(&self) -> MemoryStats {
        let allocated = self.num_allocated_blocks();
        let total = self.config.num_blocks;
        let bytes_per_block = self.config.bytes_per_block();

        MemoryStats {
            total_blocks: total,
            allocated_blocks: allocated,
            free_blocks: total - allocated,
            total_bytes: total * bytes_per_block,
            allocated_bytes: allocated * bytes_per_block,
            free_bytes: (total - allocated) * bytes_per_block,
            num_sequences: self.sequence_tables.len(),
        }
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryStats {
    /// Total blocks in pool.
    pub total_blocks: usize,
    /// Currently allocated blocks.
    pub allocated_blocks: usize,
    /// Free blocks available.
    pub free_blocks: usize,
    /// Total memory in bytes.
    pub total_bytes: usize,
    /// Allocated memory in bytes.
    pub allocated_bytes: usize,
    /// Free memory in bytes.
    pub free_bytes: usize,
    /// Number of active sequences.
    pub num_sequences: usize,
}

impl MemoryStats {
    /// Calculate utilization as a fraction.
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.allocated_blocks as f64 / self.total_blocks as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_id() {
        let id = BlockId::new(42);
        assert_eq!(id.value(), 42);
        assert_eq!(id, BlockId(42));
    }

    #[test]
    fn test_block_state() {
        assert!(BlockState::Free.is_free());
        assert!(!BlockState::Free.is_allocated());
        assert_eq!(BlockState::Free.ref_count(), 0);

        assert!(!BlockState::Allocated.is_free());
        assert!(BlockState::Allocated.is_allocated());
        assert_eq!(BlockState::Allocated.ref_count(), 1);

        assert!(!BlockState::Shared(3).is_free());
        assert!(BlockState::Shared(3).is_allocated());
        assert_eq!(BlockState::Shared(3).ref_count(), 3);
    }

    #[test]
    fn test_config_default() {
        let config = BlockAllocatorConfig::default();
        assert_eq!(config.num_blocks, 1024);
        assert_eq!(config.block_size, 16);
    }

    #[test]
    fn test_config_builder() {
        let config = BlockAllocatorConfig::with_blocks(512)
            .block_size(32)
            .num_layers(24)
            .num_kv_heads(4)
            .head_dim(64);

        assert_eq!(config.num_blocks, 512);
        assert_eq!(config.block_size, 32);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn test_config_memory_calculation() {
        let config = BlockAllocatorConfig::with_blocks(100)
            .block_size(16)
            .num_layers(2)
            .num_kv_heads(2)
            .head_dim(64);

        // 2 (K+V) * 2 layers * 2 heads * 64 dim * 16 tokens * 2 bytes
        let expected = 2 * 2 * 2 * 64 * 16 * 2;
        assert_eq!(config.bytes_per_block(), expected);
        assert_eq!(config.total_memory_bytes(), 100 * expected);
    }

    #[test]
    fn test_block_table_new() {
        let table = BlockTable::new(16);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.num_tokens(), 0);
    }

    #[test]
    fn test_block_table_add_blocks() {
        let mut table = BlockTable::new(16);
        table.add_block(BlockId::new(0));
        table.add_block(BlockId::new(1));

        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.blocks(), &[BlockId::new(0), BlockId::new(1)]);
    }

    #[test]
    fn test_block_table_pop() {
        let mut table = BlockTable::new(16);
        table.add_block(BlockId::new(0));
        table.add_block(BlockId::new(1));

        assert_eq!(table.pop_block(), Some(BlockId::new(1)));
        assert_eq!(table.num_blocks(), 1);
    }

    #[test]
    fn test_block_table_blocks_needed() {
        let table = BlockTable::new(16);

        assert_eq!(table.blocks_needed(0), 0);
        assert_eq!(table.blocks_needed(1), 1);
        assert_eq!(table.blocks_needed(16), 1);
        assert_eq!(table.blocks_needed(17), 2);
        assert_eq!(table.blocks_needed(32), 2);
        assert_eq!(table.blocks_needed(33), 3);
    }

    #[test]
    fn test_block_table_needs_more_blocks() {
        let mut table = BlockTable::new(16);
        table.add_block(BlockId::new(0));
        table.set_num_tokens(10);

        // Have 1 block (16 tokens), at 10 tokens
        assert!(!table.needs_more_blocks(5)); // 15 total, fits in 1
        assert!(!table.needs_more_blocks(6)); // 16 total, fits in 1
        assert!(table.needs_more_blocks(7)); // 17 total, needs 2
    }

    #[test]
    fn test_block_table_get_block_and_offset() {
        let mut table = BlockTable::new(16);
        table.add_block(BlockId::new(10));
        table.add_block(BlockId::new(20));

        assert_eq!(table.get_block_and_offset(0), Some((BlockId::new(10), 0)));
        assert_eq!(table.get_block_and_offset(15), Some((BlockId::new(10), 15)));
        assert_eq!(table.get_block_and_offset(16), Some((BlockId::new(20), 0)));
        assert_eq!(table.get_block_and_offset(31), Some((BlockId::new(20), 15)));
        assert_eq!(table.get_block_and_offset(32), None);
    }

    #[test]
    fn test_allocator_new() {
        let config = BlockAllocatorConfig::with_blocks(100);
        let allocator = BlockAllocator::new(config);

        assert_eq!(allocator.num_free_blocks(), 100);
        assert_eq!(allocator.num_allocated_blocks(), 0);
    }

    #[test]
    fn test_allocator_allocate() {
        let config = BlockAllocatorConfig::with_blocks(100);
        let mut allocator = BlockAllocator::new(config);

        allocator.allocate(1, 5).unwrap();

        assert_eq!(allocator.num_free_blocks(), 95);
        assert_eq!(allocator.num_allocated_blocks(), 5);

        let table = allocator.get_block_table(1).unwrap();
        assert_eq!(table.num_blocks(), 5);
    }

    #[test]
    fn test_allocator_allocate_more() {
        let config = BlockAllocatorConfig::with_blocks(100);
        let mut allocator = BlockAllocator::new(config);

        allocator.allocate(1, 5).unwrap();
        allocator.allocate_more(1, 3).unwrap();

        assert_eq!(allocator.num_allocated_blocks(), 8);

        let table = allocator.get_block_table(1).unwrap();
        assert_eq!(table.num_blocks(), 8);
    }

    #[test]
    fn test_allocator_free() {
        let config = BlockAllocatorConfig::with_blocks(100);
        let mut allocator = BlockAllocator::new(config);

        allocator.allocate(1, 5).unwrap();
        let freed = allocator.free(1);

        assert_eq!(freed.len(), 5);
        assert_eq!(allocator.num_free_blocks(), 100);
        assert!(allocator.get_block_table(1).is_none());
    }

    #[test]
    fn test_allocator_insufficient_blocks() {
        let config = BlockAllocatorConfig::with_blocks(10);
        let mut allocator = BlockAllocator::new(config);

        assert!(allocator.allocate(1, 5).is_ok());
        assert!(allocator.allocate(2, 10).is_err()); // Only 5 left
    }

    #[test]
    fn test_allocator_can_allocate() {
        let config = BlockAllocatorConfig::with_blocks(10);
        let mut allocator = BlockAllocator::new(config);

        assert!(allocator.can_allocate(10));
        assert!(!allocator.can_allocate(11));

        allocator.allocate(1, 5).unwrap();

        assert!(allocator.can_allocate(5));
        assert!(!allocator.can_allocate(6));
    }

    #[test]
    fn test_allocator_fork() {
        let config = BlockAllocatorConfig::with_blocks(100);
        let mut allocator = BlockAllocator::new(config);

        allocator.allocate(1, 5).unwrap();
        allocator.fork(1, 2).unwrap();

        // Child should have same blocks
        let parent_table = allocator.get_block_table(1).unwrap();
        let child_table = allocator.get_block_table(2).unwrap();
        assert_eq!(parent_table.blocks(), child_table.blocks());

        // Blocks should be shared
        for block_id in parent_table.blocks() {
            assert_eq!(allocator.block_state(*block_id), BlockState::Shared(2));
        }

        // No new blocks allocated
        assert_eq!(allocator.num_allocated_blocks(), 5);
    }

    #[test]
    fn test_allocator_fork_then_free_child() {
        let config = BlockAllocatorConfig::with_blocks(100);
        let mut allocator = BlockAllocator::new(config);

        allocator.allocate(1, 5).unwrap();
        allocator.fork(1, 2).unwrap();

        // Free child - blocks should still be allocated (to parent)
        allocator.free(2);
        assert_eq!(allocator.num_allocated_blocks(), 5);

        // Blocks should no longer be shared
        let parent_table = allocator.get_block_table(1).unwrap();
        for block_id in parent_table.blocks() {
            assert_eq!(allocator.block_state(*block_id), BlockState::Allocated);
        }
    }

    #[test]
    fn test_allocator_memory_stats() {
        let config = BlockAllocatorConfig::with_blocks(100).block_size(16);
        let mut allocator = BlockAllocator::new(config);

        allocator.allocate(1, 10).unwrap();
        allocator.allocate(2, 20).unwrap();

        let stats = allocator.memory_stats();
        assert_eq!(stats.total_blocks, 100);
        assert_eq!(stats.allocated_blocks, 30);
        assert_eq!(stats.free_blocks, 70);
        assert_eq!(stats.num_sequences, 2);
        assert!((stats.utilization() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_memory_stats_utilization_empty() {
        let stats = MemoryStats {
            total_blocks: 0,
            allocated_blocks: 0,
            free_blocks: 0,
            total_bytes: 0,
            allocated_bytes: 0,
            free_bytes: 0,
            num_sequences: 0,
        };
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn test_allocator_block_state() {
        let config = BlockAllocatorConfig::with_blocks(10);
        let mut allocator = BlockAllocator::new(config);

        assert_eq!(allocator.block_state(BlockId::new(0)), BlockState::Free);

        allocator.allocate(1, 1).unwrap();
        assert_eq!(
            allocator.block_state(BlockId::new(0)),
            BlockState::Allocated
        );
    }

    #[test]
    fn test_allocator_multiple_sequences() {
        let config = BlockAllocatorConfig::with_blocks(100);
        let mut allocator = BlockAllocator::new(config);

        allocator.allocate(1, 10).unwrap();
        allocator.allocate(2, 20).unwrap();
        allocator.allocate(3, 30).unwrap();

        assert_eq!(allocator.num_allocated_blocks(), 60);

        allocator.free(2);
        assert_eq!(allocator.num_allocated_blocks(), 40);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_blocks_needed_monotonic(block_size in 1usize..100, tokens in 0usize..10000) {
            let table = BlockTable::new(block_size);
            let needed = table.blocks_needed(tokens);

            // Should be enough to hold all tokens
            prop_assert!(needed * block_size >= tokens);

            // Should not be more than necessary
            if tokens > 0 {
                prop_assert!((needed - 1) * block_size < tokens);
            }
        }

        #[test]
        fn prop_allocate_free_invariant(num_blocks in 10usize..100, alloc_sizes in prop::collection::vec(1usize..10, 1..5)) {
            let config = BlockAllocatorConfig::with_blocks(num_blocks);
            let mut allocator = BlockAllocator::new(config);

            let total_alloc: usize = alloc_sizes.iter().sum();
            if total_alloc <= num_blocks {
                // Allocate all
                for (i, &size) in alloc_sizes.iter().enumerate() {
                    allocator.allocate(i as u64, size).unwrap();
                }

                prop_assert_eq!(allocator.num_allocated_blocks(), total_alloc);

                // Free all
                for i in 0..alloc_sizes.len() {
                    allocator.free(i as u64);
                }

                prop_assert_eq!(allocator.num_free_blocks(), num_blocks);
            }
        }
    }
}
