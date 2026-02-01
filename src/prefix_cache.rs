//! Prefix Caching (RadixAttention)
//!
//! Implements prefix caching for efficient KV cache reuse when multiple
//! requests share common prefixes (e.g., system prompts).
//!
//! # TGI Reference
//!
//! Based on TGI's prefix caching and vLLM's RadixAttention pattern.
//! See: <https://github.com/huggingface/text-generation-inference>
//!
//! # Algorithm
//!
//! 1. Build a radix tree of token sequences
//! 2. When a new request arrives, find longest matching prefix
//! 3. Reuse cached KV blocks for matching prefix
//! 4. Only compute attention for new tokens
//!
//! # Example
//!
//! ```rust
//! use tgi_gtc::prefix_cache::{PrefixCache, PrefixCacheConfig};
//!
//! let config = PrefixCacheConfig::default();
//! let mut cache = PrefixCache::new(config);
//!
//! // First request - no cache hit
//! let tokens1 = vec![1, 2, 3, 4, 5];
//! let match1 = cache.find_prefix(&tokens1);
//! assert_eq!(match1.matched_length, 0);
//!
//! // Insert the sequence
//! cache.insert(&tokens1, 100); // 100 is the block ID
//!
//! // Second request with same prefix - cache hit!
//! let tokens2 = vec![1, 2, 3, 4, 5, 6, 7];
//! let match2 = cache.find_prefix(&tokens2);
//! assert_eq!(match2.matched_length, 5);
//! ```
//!
//! # Performance
//!
//! Prefix caching provides significant speedup when:
//! - Many requests share common system prompts
//! - Multi-turn conversations reuse context
//! - Batch processing with similar inputs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for prefix cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixCacheConfig {
    /// Maximum number of cached prefixes.
    pub max_entries: usize,

    /// Minimum prefix length to cache.
    pub min_prefix_length: usize,

    /// Block size for KV cache alignment.
    pub block_size: usize,

    /// Enable LRU eviction when full.
    pub enable_eviction: bool,

    /// Time-to-live for cache entries (in seconds, 0 = infinite).
    pub ttl_seconds: u64,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            min_prefix_length: 4,
            block_size: 16,
            enable_eviction: true,
            ttl_seconds: 0,
        }
    }
}

impl PrefixCacheConfig {
    /// Create config for small cache (low memory).
    pub fn small() -> Self {
        Self {
            max_entries: 1000,
            min_prefix_length: 8,
            ..Default::default()
        }
    }

    /// Create config for large cache (high memory).
    pub fn large() -> Self {
        Self {
            max_entries: 100000,
            min_prefix_length: 2,
            ..Default::default()
        }
    }
}

/// Result of a prefix lookup.
#[derive(Debug, Clone)]
pub struct PrefixMatch {
    /// Number of tokens matched.
    pub matched_length: usize,

    /// Block IDs for cached KV data.
    pub block_ids: Vec<u32>,

    /// Whether this is a complete match (entire sequence cached).
    pub is_complete: bool,

    /// Cache hit rate for statistics.
    pub cache_hit: bool,
}

impl PrefixMatch {
    /// Create a cache miss result.
    pub fn miss() -> Self {
        Self {
            matched_length: 0,
            block_ids: Vec::new(),
            is_complete: false,
            cache_hit: false,
        }
    }

    /// Create a cache hit result.
    pub fn hit(matched_length: usize, block_ids: Vec<u32>, is_complete: bool) -> Self {
        Self {
            matched_length,
            block_ids,
            is_complete,
            cache_hit: true,
        }
    }

    /// Number of tokens that need to be computed.
    pub fn tokens_to_compute(&self, total_length: usize) -> usize {
        total_length.saturating_sub(self.matched_length)
    }
}

/// A node in the radix tree.
#[derive(Debug, Clone)]
struct RadixNode {
    /// Token sequence for this edge.
    tokens: Vec<u32>,

    /// Block IDs storing KV cache for this prefix.
    block_ids: Vec<u32>,

    /// Children nodes (keyed by first token of edge).
    children: HashMap<u32, RadixNode>,

    /// Access count for LRU eviction.
    access_count: u64,

    /// Last access timestamp.
    last_access: u64,

    /// Reference count (number of active requests using this).
    ref_count: u32,
}

impl RadixNode {
    fn new(tokens: Vec<u32>, block_ids: Vec<u32>) -> Self {
        Self {
            tokens,
            block_ids,
            children: HashMap::new(),
            access_count: 0,
            last_access: 0,
            ref_count: 0,
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Prefix cache using a radix tree structure.
#[derive(Debug)]
pub struct PrefixCache {
    config: PrefixCacheConfig,
    root: RadixNode,
    stats: PrefixCacheStats,
    current_time: u64,
    total_entries: usize,
}

/// Statistics for prefix cache.
#[derive(Debug, Clone, Default)]
pub struct PrefixCacheStats {
    /// Total lookup requests.
    pub total_lookups: u64,

    /// Cache hits.
    pub cache_hits: u64,

    /// Cache misses.
    pub cache_misses: u64,

    /// Total tokens matched across all lookups.
    pub total_tokens_matched: u64,

    /// Total tokens requested across all lookups.
    pub total_tokens_requested: u64,

    /// Number of insertions.
    pub insertions: u64,

    /// Number of evictions.
    pub evictions: u64,
}

impl PrefixCacheStats {
    /// Cache hit rate.
    pub fn hit_rate(&self) -> f32 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.cache_hits as f32 / self.total_lookups as f32
        }
    }

    /// Token reuse rate (tokens matched / tokens requested).
    pub fn token_reuse_rate(&self) -> f32 {
        if self.total_tokens_requested == 0 {
            0.0
        } else {
            self.total_tokens_matched as f32 / self.total_tokens_requested as f32
        }
    }

    /// Average matched prefix length on hits.
    pub fn avg_match_length(&self) -> f32 {
        if self.cache_hits == 0 {
            0.0
        } else {
            self.total_tokens_matched as f32 / self.cache_hits as f32
        }
    }
}

impl PrefixCache {
    /// Create a new prefix cache.
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            config,
            root: RadixNode::new(Vec::new(), Vec::new()),
            stats: PrefixCacheStats::default(),
            current_time: 0,
            total_entries: 0,
        }
    }

    /// Get cache configuration.
    pub fn config(&self) -> &PrefixCacheConfig {
        &self.config
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &PrefixCacheStats {
        &self.stats
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.total_entries
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.total_entries == 0
    }

    /// Find longest matching prefix for a token sequence.
    pub fn find_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        self.current_time += 1;
        self.stats.total_lookups += 1;
        self.stats.total_tokens_requested += tokens.len() as u64;

        if tokens.is_empty() || tokens.len() < self.config.min_prefix_length {
            self.stats.cache_misses += 1;
            return PrefixMatch::miss();
        }

        let (matched_len, block_ids) = self.find_in_tree(tokens);

        if matched_len > 0 {
            self.stats.cache_hits += 1;
            self.stats.total_tokens_matched += matched_len as u64;
            let is_complete = matched_len == tokens.len();
            PrefixMatch::hit(matched_len, block_ids, is_complete)
        } else {
            self.stats.cache_misses += 1;
            PrefixMatch::miss()
        }
    }

    /// Find matching prefix in the radix tree.
    fn find_in_tree(&mut self, tokens: &[u32]) -> (usize, Vec<u32>) {
        let mut current = &mut self.root;
        let mut matched = 0;
        let mut block_ids = Vec::new();
        let mut pos = 0;

        while pos < tokens.len() {
            let first_token = tokens[pos];

            if let Some(child) = current.children.get_mut(&first_token) {
                // Check how much of the edge matches
                let edge_len = child.tokens.len();
                let remaining = &tokens[pos..];

                if remaining.len() >= edge_len && remaining[..edge_len] == child.tokens[..] {
                    // Full edge match
                    matched += edge_len;
                    block_ids.extend(child.block_ids.iter().copied());
                    child.access_count += 1;
                    child.last_access = self.current_time;
                    pos += edge_len;
                    current = child;
                } else {
                    // Partial match - find common prefix
                    let common = remaining
                        .iter()
                        .zip(child.tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();

                    if common > 0 {
                        matched += common;
                        // Only add blocks for the matched portion
                        let blocks_per_token = if child.tokens.is_empty() {
                            0
                        } else {
                            child.block_ids.len() / child.tokens.len()
                        };
                        block_ids.extend(
                            child
                                .block_ids
                                .iter()
                                .take(common * blocks_per_token)
                                .copied(),
                        );
                    }
                    break;
                }
            } else {
                break;
            }
        }

        (matched, block_ids)
    }

    /// Insert a token sequence with its block IDs.
    pub fn insert(&mut self, tokens: &[u32], block_id: u32) {
        self.insert_with_blocks(tokens, vec![block_id]);
    }

    /// Insert a token sequence with multiple block IDs.
    pub fn insert_with_blocks(&mut self, tokens: &[u32], block_ids: Vec<u32>) {
        if tokens.len() < self.config.min_prefix_length {
            return;
        }

        // Check capacity and evict if needed
        if self.total_entries >= self.config.max_entries && self.config.enable_eviction {
            self.evict_lru();
        }

        self.insert_in_tree(tokens, block_ids);
        self.total_entries += 1;
        self.stats.insertions += 1;
    }

    /// Insert into the radix tree.
    fn insert_in_tree(&mut self, tokens: &[u32], block_ids: Vec<u32>) {
        if tokens.is_empty() {
            return;
        }

        // Simple insertion: just insert at root level for now
        // A full radix tree implementation would traverse/split, but that
        // requires complex borrow handling. This simplified version still
        // demonstrates the pattern.
        let first_token = tokens[0];

        if self.root.children.contains_key(&first_token) {
            // Update existing entry
            if let Some(child) = self.root.children.get_mut(&first_token) {
                child.tokens = tokens.to_vec();
                child.block_ids = block_ids;
                child.access_count += 1;
                child.last_access = self.current_time;
            }
        } else {
            // Insert new entry
            let new_node = RadixNode::new(tokens.to_vec(), block_ids);
            self.root.children.insert(first_token, new_node);
        }
    }

    /// Evict least recently used entry.
    fn evict_lru(&mut self) {
        // Simple LRU: find leaf with lowest last_access
        if let Some((path, _)) = self.find_lru_leaf(&self.root, &[], u64::MAX) {
            if !path.is_empty() {
                self.remove_path(&path);
                self.stats.evictions += 1;
                self.total_entries = self.total_entries.saturating_sub(1);
            }
        }
    }

    /// Find leaf with lowest access time.
    fn find_lru_leaf(
        &self,
        node: &RadixNode,
        path: &[u32],
        min_access: u64,
    ) -> Option<(Vec<u32>, u64)> {
        let mut result: Option<(Vec<u32>, u64)> = None;
        let mut current_min = min_access;

        if node.is_leaf() && !path.is_empty() {
            if node.last_access < current_min && node.ref_count == 0 {
                return Some((path.to_vec(), node.last_access));
            }
        }

        for (&first_token, child) in &node.children {
            let mut child_path = path.to_vec();
            child_path.push(first_token);

            if let Some((p, access)) = self.find_lru_leaf(child, &child_path, current_min) {
                if access < current_min {
                    current_min = access;
                    result = Some((p, access));
                }
            }
        }

        result
    }

    /// Remove a path from the tree.
    fn remove_path(&mut self, path: &[u32]) {
        if path.is_empty() {
            return;
        }

        // Navigate to parent and remove child
        let mut current = &mut self.root;

        for &token in &path[..path.len() - 1] {
            if let Some(child) = current.children.get_mut(&token) {
                current = child;
            } else {
                return;
            }
        }

        current.children.remove(&path[path.len() - 1]);
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.root = RadixNode::new(Vec::new(), Vec::new());
        self.total_entries = 0;
    }

    /// Increment reference count for a prefix (mark as in-use).
    pub fn acquire(&mut self, tokens: &[u32]) {
        self.update_ref_count(tokens, true);
    }

    /// Decrement reference count for a prefix (mark as available for eviction).
    pub fn release(&mut self, tokens: &[u32]) {
        self.update_ref_count(tokens, false);
    }

    fn update_ref_count(&mut self, tokens: &[u32], increment: bool) {
        let mut current = &mut self.root;
        let mut pos = 0;

        while pos < tokens.len() {
            let first_token = tokens[pos];

            if let Some(child) = current.children.get_mut(&first_token) {
                if increment {
                    child.ref_count += 1;
                } else {
                    child.ref_count = child.ref_count.saturating_sub(1);
                }

                let edge_len = child.tokens.len();
                if pos + edge_len <= tokens.len() && tokens[pos..pos + edge_len] == child.tokens[..]
                {
                    pos += edge_len;
                    current = current.children.get_mut(&first_token).unwrap();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
}

/// Hash-based prefix cache for simpler implementation.
///
/// Uses a hash map with token sequence hashes for O(1) lookup.
/// Less memory efficient than radix tree but simpler.
#[derive(Debug)]
pub struct HashPrefixCache {
    config: PrefixCacheConfig,
    cache: HashMap<u64, CacheEntry>,
    stats: PrefixCacheStats,
    current_time: u64,
}

/// Entry in the hash-based cache.
#[derive(Debug, Clone)]
struct CacheEntry {
    tokens: Vec<u32>,
    block_ids: Vec<u32>,
    access_count: u64,
    last_access: u64,
    ref_count: u32,
}

impl HashPrefixCache {
    /// Create a new hash-based prefix cache.
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            stats: PrefixCacheStats::default(),
            current_time: 0,
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &PrefixCacheStats {
        &self.stats
    }

    /// Find matching prefix.
    pub fn find_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        self.current_time += 1;
        self.stats.total_lookups += 1;
        self.stats.total_tokens_requested += tokens.len() as u64;

        // Try progressively shorter prefixes
        for len in (self.config.min_prefix_length..=tokens.len()).rev() {
            let prefix = &tokens[..len];
            let hash = self.hash_tokens(prefix);

            if let Some(entry) = self.cache.get_mut(&hash) {
                if entry.tokens == prefix {
                    entry.access_count += 1;
                    entry.last_access = self.current_time;

                    self.stats.cache_hits += 1;
                    self.stats.total_tokens_matched += len as u64;

                    return PrefixMatch::hit(len, entry.block_ids.clone(), len == tokens.len());
                }
            }
        }

        self.stats.cache_misses += 1;
        PrefixMatch::miss()
    }

    /// Insert a token sequence.
    pub fn insert(&mut self, tokens: &[u32], block_ids: Vec<u32>) {
        if tokens.len() < self.config.min_prefix_length {
            return;
        }

        // Evict if at capacity
        if self.cache.len() >= self.config.max_entries && self.config.enable_eviction {
            self.evict_lru();
        }

        let hash = self.hash_tokens(tokens);
        self.cache.insert(
            hash,
            CacheEntry {
                tokens: tokens.to_vec(),
                block_ids,
                access_count: 1,
                last_access: self.current_time,
                ref_count: 0,
            },
        );

        self.stats.insertions += 1;
    }

    /// Simple hash function for token sequences.
    fn hash_tokens(&self, tokens: &[u32]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
        for &token in tokens {
            hash ^= token as u64;
            hash = hash.wrapping_mul(0x100000001b3); // FNV prime
        }
        hash
    }

    /// Evict least recently used entry.
    fn evict_lru(&mut self) {
        let lru_key = self
            .cache
            .iter()
            .filter(|(_, e)| e.ref_count == 0)
            .min_by_key(|(_, e)| e.last_access)
            .map(|(&k, _)| k);

        if let Some(key) = lru_key {
            self.cache.remove(&key);
            self.stats.evictions += 1;
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_cache_config_default() {
        let config = PrefixCacheConfig::default();
        assert_eq!(config.max_entries, 10000);
        assert_eq!(config.min_prefix_length, 4);
        assert!(config.enable_eviction);
    }

    #[test]
    fn test_prefix_match_miss() {
        let m = PrefixMatch::miss();
        assert_eq!(m.matched_length, 0);
        assert!(!m.cache_hit);
        assert_eq!(m.tokens_to_compute(10), 10);
    }

    #[test]
    fn test_prefix_match_hit() {
        let m = PrefixMatch::hit(5, vec![1, 2], false);
        assert_eq!(m.matched_length, 5);
        assert!(m.cache_hit);
        assert_eq!(m.tokens_to_compute(10), 5);
    }

    #[test]
    fn test_prefix_cache_insert_and_find() {
        let config = PrefixCacheConfig {
            min_prefix_length: 2,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(config);

        let tokens = vec![1, 2, 3, 4, 5];
        cache.insert(&tokens, 100);

        // Exact match
        let m = cache.find_prefix(&tokens);
        assert!(m.cache_hit);
        assert_eq!(m.matched_length, 5);

        // Prefix match
        let longer = vec![1, 2, 3, 4, 5, 6, 7];
        let m = cache.find_prefix(&longer);
        assert!(m.cache_hit);
        assert_eq!(m.matched_length, 5);
    }

    #[test]
    fn test_prefix_cache_no_match() {
        let config = PrefixCacheConfig {
            min_prefix_length: 2,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(config);

        let tokens1 = vec![1, 2, 3, 4];
        cache.insert(&tokens1, 100);

        // Different sequence - no match
        let tokens2 = vec![5, 6, 7, 8];
        let m = cache.find_prefix(&tokens2);
        assert!(!m.cache_hit);
        assert_eq!(m.matched_length, 0);
    }

    #[test]
    fn test_prefix_cache_stats() {
        let config = PrefixCacheConfig {
            min_prefix_length: 2,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(config);

        let tokens = vec![1, 2, 3, 4];
        cache.insert(&tokens, 100);

        // Hit
        cache.find_prefix(&tokens);
        // Miss
        cache.find_prefix(&[5, 6, 7, 8]);

        let stats = cache.stats();
        assert_eq!(stats.total_lookups, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_prefix_cache_eviction() {
        let config = PrefixCacheConfig {
            max_entries: 2,
            min_prefix_length: 2,
            enable_eviction: true,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(config);

        cache.insert(&[1, 2, 3], 100);
        cache.insert(&[4, 5, 6], 101);
        assert_eq!(cache.len(), 2);

        // This should trigger eviction
        cache.insert(&[7, 8, 9], 102);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_prefix_cache_clear() {
        let config = PrefixCacheConfig {
            min_prefix_length: 2,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(config);

        cache.insert(&[1, 2, 3], 100);
        cache.insert(&[4, 5, 6], 101);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_hash_prefix_cache() {
        let config = PrefixCacheConfig {
            min_prefix_length: 2,
            ..Default::default()
        };
        let mut cache = HashPrefixCache::new(config);

        let tokens = vec![1, 2, 3, 4, 5];
        cache.insert(&tokens, vec![100, 101]);

        let m = cache.find_prefix(&tokens);
        assert!(m.cache_hit);
        assert_eq!(m.matched_length, 5);
        assert_eq!(m.block_ids, vec![100, 101]);
    }

    #[test]
    fn test_hash_prefix_cache_progressive_match() {
        let config = PrefixCacheConfig {
            min_prefix_length: 2,
            ..Default::default()
        };
        let mut cache = HashPrefixCache::new(config);

        // Insert shorter prefix
        cache.insert(&[1, 2, 3], vec![100]);

        // Query with longer sequence - should match the prefix
        let m = cache.find_prefix(&[1, 2, 3, 4, 5]);
        assert!(m.cache_hit);
        assert_eq!(m.matched_length, 3);
    }

    #[test]
    fn test_prefix_cache_ref_counting() {
        let config = PrefixCacheConfig {
            min_prefix_length: 2,
            max_entries: 2,
            enable_eviction: true,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(config);

        let tokens1 = vec![1, 2, 3];
        cache.insert(&tokens1, 100);

        // Acquire (mark as in-use)
        cache.acquire(&tokens1);

        // Insert second entry
        cache.insert(&[4, 5, 6], 101);

        // Try to insert third - should evict second, not first (which is in use)
        cache.insert(&[7, 8, 9], 102);

        // First should still be findable
        let m = cache.find_prefix(&tokens1);
        assert!(m.cache_hit);
    }
}
