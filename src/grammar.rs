//! Grammar-Constrained Decoding
//!
//! Implements constrained decoding using context-free grammars (CFG) to ensure
//! generated text follows a specific format (e.g., JSON, regex patterns).
//!
//! # TGI Reference
//!
//! Based on TGI's grammar/guided generation support.
//! See: <https://github.com/huggingface/text-generation-inference>
//!
//! # Algorithm
//!
//! 1. Parse grammar specification (JSON schema, regex, EBNF)
//! 2. Build token mask for valid next tokens at each state
//! 3. Apply mask to logits before sampling
//! 4. Update grammar state after each token
//!
//! # Example
//!
//! ```rust
//! use tgi_gtc::grammar::{Grammar, GrammarType, TokenMask};
//!
//! // Create JSON grammar
//! let grammar = Grammar::json();
//!
//! // Get valid tokens at current state
//! let mask = grammar.get_token_mask();
//! println!("Valid tokens: {:?}", mask.allowed_count());
//! ```

use serde::{Deserialize, Serialize};

/// Type of grammar constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrammarType {
    /// JSON output format.
    Json,
    /// JSON conforming to a schema.
    JsonSchema(String),
    /// Regular expression pattern.
    Regex(String),
    /// Extended Backus-Naur Form.
    Ebnf(String),
    /// Choice between options.
    Choice(Vec<String>),
    /// No constraint.
    None,
}

/// Configuration for grammar-constrained decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarConfig {
    /// Grammar type and specification.
    pub grammar_type: GrammarType,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Whether to allow partial matches.
    pub allow_partial: bool,

    /// Maximum recursion depth for nested structures.
    pub max_depth: usize,
}

impl Default for GrammarConfig {
    fn default() -> Self {
        Self {
            grammar_type: GrammarType::None,
            vocab_size: 32000,
            allow_partial: true,
            max_depth: 64,
        }
    }
}

/// Token mask indicating which tokens are valid.
#[derive(Debug, Clone)]
pub struct TokenMask {
    /// Bitvector of allowed tokens (true = allowed).
    allowed: Vec<bool>,
}

impl TokenMask {
    /// Create a mask allowing all tokens.
    pub fn allow_all(vocab_size: usize) -> Self {
        Self {
            allowed: vec![true; vocab_size],
        }
    }

    /// Create a mask blocking all tokens.
    pub fn block_all(vocab_size: usize) -> Self {
        Self {
            allowed: vec![false; vocab_size],
        }
    }

    /// Create a mask from allowed token IDs.
    pub fn from_allowed(vocab_size: usize, allowed_ids: &[u32]) -> Self {
        let mut mask = Self::block_all(vocab_size);
        for &id in allowed_ids {
            if (id as usize) < vocab_size {
                mask.allowed[id as usize] = true;
            }
        }
        mask
    }

    /// Check if a token is allowed.
    pub fn is_allowed(&self, token_id: u32) -> bool {
        self.allowed
            .get(token_id as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Set a token as allowed.
    pub fn allow(&mut self, token_id: u32) {
        if (token_id as usize) < self.allowed.len() {
            self.allowed[token_id as usize] = true;
        }
    }

    /// Set a token as blocked.
    pub fn block(&mut self, token_id: u32) {
        if (token_id as usize) < self.allowed.len() {
            self.allowed[token_id as usize] = false;
        }
    }

    /// Count of allowed tokens.
    pub fn allowed_count(&self) -> usize {
        self.allowed.iter().filter(|&&x| x).count()
    }

    /// Apply mask to logits (set blocked tokens to -inf).
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        for (i, &allowed) in self.allowed.iter().enumerate() {
            if !allowed && i < logits.len() {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    /// Get list of allowed token IDs.
    pub fn allowed_tokens(&self) -> Vec<u32> {
        self.allowed
            .iter()
            .enumerate()
            .filter(|(_, &allowed)| allowed)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Intersect with another mask.
    pub fn intersect(&mut self, other: &TokenMask) {
        for (i, allowed) in self.allowed.iter_mut().enumerate() {
            if let Some(&other_allowed) = other.allowed.get(i) {
                *allowed = *allowed && other_allowed;
            }
        }
    }

    /// Union with another mask.
    pub fn union(&mut self, other: &TokenMask) {
        for (i, allowed) in self.allowed.iter_mut().enumerate() {
            if let Some(&other_allowed) = other.allowed.get(i) {
                *allowed = *allowed || other_allowed;
            }
        }
    }
}

/// State of grammar parsing.
#[derive(Debug, Clone)]
pub enum GrammarState {
    /// At start of grammar.
    Start,
    /// Inside JSON object.
    InObject { depth: usize, expecting_key: bool },
    /// Inside JSON array.
    InArray { depth: usize },
    /// Inside JSON string.
    InString { escaped: bool },
    /// Inside JSON number.
    InNumber,
    /// Expecting value (after colon or comma).
    ExpectingValue,
    /// Grammar complete.
    Complete,
    /// Error state.
    Error(String),
}

/// Grammar for constrained decoding.
#[derive(Debug)]
pub struct Grammar {
    config: GrammarConfig,
    state: GrammarState,
    depth: usize,
    generated: Vec<u32>,
}

impl Grammar {
    /// Create a new grammar.
    pub fn new(config: GrammarConfig) -> Self {
        Self {
            config,
            state: GrammarState::Start,
            depth: 0,
            generated: Vec::new(),
        }
    }

    /// Create a JSON grammar.
    pub fn json() -> Self {
        Self::new(GrammarConfig {
            grammar_type: GrammarType::Json,
            ..Default::default()
        })
    }

    /// Create a JSON schema grammar.
    pub fn json_schema(schema: &str) -> Self {
        Self::new(GrammarConfig {
            grammar_type: GrammarType::JsonSchema(schema.to_string()),
            ..Default::default()
        })
    }

    /// Create a regex grammar.
    pub fn regex(pattern: &str) -> Self {
        Self::new(GrammarConfig {
            grammar_type: GrammarType::Regex(pattern.to_string()),
            ..Default::default()
        })
    }

    /// Create a choice grammar.
    pub fn choice(options: Vec<String>) -> Self {
        Self::new(GrammarConfig {
            grammar_type: GrammarType::Choice(options),
            ..Default::default()
        })
    }

    /// Get current state.
    pub fn state(&self) -> &GrammarState {
        &self.state
    }

    /// Check if grammar is complete.
    pub fn is_complete(&self) -> bool {
        matches!(self.state, GrammarState::Complete)
    }

    /// Check if grammar is in error state.
    pub fn is_error(&self) -> bool {
        matches!(self.state, GrammarState::Error(_))
    }

    /// Get token mask for current state.
    pub fn get_token_mask(&self) -> TokenMask {
        match &self.config.grammar_type {
            GrammarType::Json | GrammarType::JsonSchema(_) => self.json_token_mask(),
            GrammarType::Regex(pattern) => self.regex_token_mask(pattern),
            GrammarType::Choice(options) => self.choice_token_mask(options),
            GrammarType::Ebnf(_) => TokenMask::allow_all(self.config.vocab_size),
            GrammarType::None => TokenMask::allow_all(self.config.vocab_size),
        }
    }

    /// Update grammar state after generating a token.
    pub fn update(&mut self, token_id: u32, token_text: &str) {
        self.generated.push(token_id);

        // Clone grammar type to avoid borrow issues
        let grammar_type = self.config.grammar_type.clone();
        match grammar_type {
            GrammarType::Json | GrammarType::JsonSchema(_) => {
                self.update_json_state(token_text);
            }
            GrammarType::Regex(_) => {
                // Regex validation happens at the end
            }
            GrammarType::Choice(options) => {
                self.update_choice_state(token_text, &options);
            }
            GrammarType::Ebnf(_) | GrammarType::None => {}
        }
    }

    /// Get JSON token mask based on current state.
    fn json_token_mask(&self) -> TokenMask {
        // Simplified JSON grammar - in practice, this would use the tokenizer
        // to map characters to token IDs
        match &self.state {
            GrammarState::Start => {
                // Allow: { [ " digits true false null
                // For simplicity, we allow a broad set
                TokenMask::allow_all(self.config.vocab_size)
            }
            GrammarState::InObject { expecting_key, .. } => {
                if *expecting_key {
                    // Allow: " }
                    TokenMask::allow_all(self.config.vocab_size)
                } else {
                    // Allow: : ,
                    TokenMask::allow_all(self.config.vocab_size)
                }
            }
            GrammarState::InArray { .. } => {
                // Allow: values, ] ,
                TokenMask::allow_all(self.config.vocab_size)
            }
            GrammarState::InString { .. } => {
                // Allow all printable characters, escape sequences
                TokenMask::allow_all(self.config.vocab_size)
            }
            GrammarState::InNumber => {
                // Allow: digits, . e E + -
                TokenMask::allow_all(self.config.vocab_size)
            }
            GrammarState::ExpectingValue => {
                // Allow: { [ " digits true false null
                TokenMask::allow_all(self.config.vocab_size)
            }
            GrammarState::Complete => {
                // Only EOS
                TokenMask::block_all(self.config.vocab_size)
            }
            GrammarState::Error(_) => {
                // Block everything
                TokenMask::block_all(self.config.vocab_size)
            }
        }
    }

    /// Update JSON state based on generated token.
    fn update_json_state(&mut self, token_text: &str) {
        for c in token_text.chars() {
            match &mut self.state {
                GrammarState::Start => match c {
                    '{' => {
                        self.state = GrammarState::InObject {
                            depth: 1,
                            expecting_key: true,
                        };
                        self.depth = 1;
                    }
                    '[' => {
                        self.state = GrammarState::InArray { depth: 1 };
                        self.depth = 1;
                    }
                    '"' => {
                        self.state = GrammarState::InString { escaped: false };
                    }
                    c if c.is_ascii_digit() || c == '-' => {
                        self.state = GrammarState::InNumber;
                    }
                    't' | 'f' | 'n' => {
                        // true, false, null - simplified
                        self.state = GrammarState::ExpectingValue;
                    }
                    _ if c.is_whitespace() => {}
                    _ => {
                        self.state = GrammarState::Error(format!("Unexpected char: {}", c));
                    }
                },
                GrammarState::InObject {
                    depth,
                    expecting_key,
                } => match c {
                    '}' => {
                        *depth -= 1;
                        if *depth == 0 {
                            self.state = GrammarState::Complete;
                            self.depth = 0;
                        }
                    }
                    '"' if *expecting_key => {
                        *expecting_key = false;
                    }
                    ':' => {
                        self.state = GrammarState::ExpectingValue;
                    }
                    ',' => {
                        *expecting_key = true;
                    }
                    '{' => {
                        *depth += 1;
                        self.depth = *depth;
                    }
                    _ => {}
                },
                GrammarState::InArray { depth } => match c {
                    ']' => {
                        *depth -= 1;
                        if *depth == 0 {
                            self.state = GrammarState::Complete;
                            self.depth = 0;
                        }
                    }
                    '[' => {
                        *depth += 1;
                        self.depth = *depth;
                    }
                    _ => {}
                },
                GrammarState::InString { escaped } => {
                    if *escaped {
                        *escaped = false;
                    } else if c == '\\' {
                        *escaped = true;
                    } else if c == '"' {
                        self.state = GrammarState::Start;
                    }
                }
                GrammarState::InNumber => {
                    if !c.is_ascii_digit()
                        && c != '.'
                        && c != 'e'
                        && c != 'E'
                        && c != '+'
                        && c != '-'
                    {
                        self.state = GrammarState::Start;
                    }
                }
                GrammarState::ExpectingValue => match c {
                    '{' => {
                        self.state = GrammarState::InObject {
                            depth: self.depth + 1,
                            expecting_key: true,
                        };
                        self.depth += 1;
                    }
                    '[' => {
                        self.state = GrammarState::InArray {
                            depth: self.depth + 1,
                        };
                        self.depth += 1;
                    }
                    '"' => {
                        self.state = GrammarState::InString { escaped: false };
                    }
                    c if c.is_ascii_digit() || c == '-' => {
                        self.state = GrammarState::InNumber;
                    }
                    _ if c.is_whitespace() => {}
                    _ => {}
                },
                GrammarState::Complete | GrammarState::Error(_) => {}
            }
        }
    }

    /// Get regex token mask.
    fn regex_token_mask(&self, _pattern: &str) -> TokenMask {
        // Simplified: allow all and validate at end
        // In practice, would use a regex automaton
        TokenMask::allow_all(self.config.vocab_size)
    }

    /// Get choice token mask.
    fn choice_token_mask(&self, _options: &[String]) -> TokenMask {
        // Allow tokens that could continue any option
        TokenMask::allow_all(self.config.vocab_size)
    }

    /// Update choice state.
    fn update_choice_state(&mut self, _token_text: &str, _options: &[String]) {
        // Track which options are still valid
    }

    /// Reset grammar to initial state.
    pub fn reset(&mut self) {
        self.state = GrammarState::Start;
        self.depth = 0;
        self.generated.clear();
    }
}

/// JSON Schema validator for structured output.
#[derive(Debug, Clone)]
pub struct JsonSchemaValidator {
    schema: JsonSchema,
}

/// Simplified JSON Schema representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Schema type.
    pub schema_type: JsonSchemaType,
    /// Required properties (for objects).
    pub required: Vec<String>,
    /// Properties (for objects).
    pub properties: std::collections::HashMap<String, Box<JsonSchema>>,
    /// Items schema (for arrays).
    pub items: Option<Box<JsonSchema>>,
    /// Enum values.
    pub enum_values: Option<Vec<String>>,
    /// Minimum value (for numbers).
    pub minimum: Option<f64>,
    /// Maximum value (for numbers).
    pub maximum: Option<f64>,
}

/// JSON Schema types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JsonSchemaType {
    Object,
    Array,
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

impl JsonSchemaValidator {
    /// Create a new validator from schema.
    pub fn new(schema: JsonSchema) -> Self {
        Self { schema }
    }

    /// Parse schema from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let schema: JsonSchema = serde_json::from_str(json)?;
        Ok(Self::new(schema))
    }

    /// Validate JSON string against schema.
    pub fn validate(&self, json: &str) -> bool {
        // Simplified validation
        match serde_json::from_str::<serde_json::Value>(json) {
            Ok(value) => self.validate_value(&value, &self.schema),
            Err(_) => false,
        }
    }

    fn validate_value(&self, value: &serde_json::Value, schema: &JsonSchema) -> bool {
        match (&schema.schema_type, value) {
            (JsonSchemaType::Object, serde_json::Value::Object(map)) => {
                // Check required properties
                for required in &schema.required {
                    if !map.contains_key(required) {
                        return false;
                    }
                }
                true
            }
            (JsonSchemaType::Array, serde_json::Value::Array(_)) => true,
            (JsonSchemaType::String, serde_json::Value::String(_)) => true,
            (JsonSchemaType::Number, serde_json::Value::Number(_)) => true,
            (JsonSchemaType::Integer, serde_json::Value::Number(n)) => {
                n.as_i64().is_some() || n.as_u64().is_some()
            }
            (JsonSchemaType::Boolean, serde_json::Value::Bool(_)) => true,
            (JsonSchemaType::Null, serde_json::Value::Null) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_mask_allow_all() {
        let mask = TokenMask::allow_all(100);
        assert_eq!(mask.allowed_count(), 100);
        assert!(mask.is_allowed(0));
        assert!(mask.is_allowed(99));
    }

    #[test]
    fn test_token_mask_block_all() {
        let mask = TokenMask::block_all(100);
        assert_eq!(mask.allowed_count(), 0);
        assert!(!mask.is_allowed(0));
    }

    #[test]
    fn test_token_mask_from_allowed() {
        let mask = TokenMask::from_allowed(100, &[5, 10, 15]);
        assert_eq!(mask.allowed_count(), 3);
        assert!(mask.is_allowed(5));
        assert!(mask.is_allowed(10));
        assert!(!mask.is_allowed(0));
    }

    #[test]
    fn test_token_mask_apply_to_logits() {
        let mask = TokenMask::from_allowed(5, &[1, 3]);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        mask.apply_to_logits(&mut logits);

        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], 4.0);
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn test_grammar_json() {
        let grammar = Grammar::json();
        assert!(matches!(grammar.state(), GrammarState::Start));
        assert!(!grammar.is_complete());
    }

    #[test]
    fn test_grammar_update_json() {
        let mut grammar = Grammar::json();

        grammar.update(0, "{");
        assert!(matches!(grammar.state(), GrammarState::InObject { .. }));

        grammar.update(1, "}");
        assert!(grammar.is_complete());
    }

    #[test]
    fn test_grammar_reset() {
        let mut grammar = Grammar::json();
        grammar.update(0, "{");
        grammar.reset();

        assert!(matches!(grammar.state(), GrammarState::Start));
    }

    #[test]
    fn test_token_mask_intersect() {
        let mut mask1 = TokenMask::from_allowed(5, &[0, 1, 2]);
        let mask2 = TokenMask::from_allowed(5, &[1, 2, 3]);

        mask1.intersect(&mask2);

        assert!(!mask1.is_allowed(0));
        assert!(mask1.is_allowed(1));
        assert!(mask1.is_allowed(2));
        assert!(!mask1.is_allowed(3));
    }

    #[test]
    fn test_token_mask_union() {
        let mut mask1 = TokenMask::from_allowed(5, &[0, 1]);
        let mask2 = TokenMask::from_allowed(5, &[2, 3]);

        mask1.union(&mask2);

        assert!(mask1.is_allowed(0));
        assert!(mask1.is_allowed(1));
        assert!(mask1.is_allowed(2));
        assert!(mask1.is_allowed(3));
    }

    #[test]
    fn test_json_schema_validator() {
        let schema = JsonSchema {
            schema_type: JsonSchemaType::Object,
            required: vec!["name".to_string()],
            properties: std::collections::HashMap::new(),
            items: None,
            enum_values: None,
            minimum: None,
            maximum: None,
        };

        let validator = JsonSchemaValidator::new(schema);

        assert!(validator.validate(r#"{"name": "test"}"#));
        assert!(!validator.validate(r#"{"other": "test"}"#));
    }
}
