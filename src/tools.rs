//! Tool/Function Calling
//!
//! Implements OpenAI-compatible function calling for LLMs to invoke external
//! tools and APIs.
//!
//! # TGI Reference
//!
//! Based on TGI's tool calling support.
//! See: <https://github.com/huggingface/text-generation-inference>
//!
//! # Example
//!
//! ```rust
//! use tgi_gtc::tools::{Tool, ToolCall, ToolResult, ToolRegistry};
//!
//! // Define a tool
//! let tool = Tool::new("get_weather")
//!     .description("Get current weather for a location")
//!     .add_param("location", "string", "City name", true)
//!     .add_param("unit", "string", "Temperature unit (celsius/fahrenheit)", false);
//!
//! // Register tools
//! let mut registry = ToolRegistry::new();
//! registry.register(tool);
//!
//! // Parse a tool call from model output
//! let call = ToolCall::new("get_weather")
//!     .with_arg("location", "San Francisco")
//!     .with_arg("unit", "celsius");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A tool/function that can be called by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool name.
    pub name: String,

    /// Description of what the tool does.
    pub description: String,

    /// Parameters schema.
    pub parameters: ToolParameters,
}

impl Tool {
    /// Create a new tool.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            parameters: ToolParameters::default(),
        }
    }

    /// Set description.
    pub fn description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a parameter.
    pub fn add_param(
        mut self,
        name: &str,
        param_type: &str,
        description: &str,
        required: bool,
    ) -> Self {
        self.parameters.properties.insert(
            name.to_string(),
            ToolParameter {
                param_type: param_type.to_string(),
                description: description.to_string(),
                enum_values: None,
            },
        );

        if required {
            self.parameters.required.push(name.to_string());
        }

        self
    }

    /// Add a parameter with enum constraint.
    pub fn add_enum_param(
        mut self,
        name: &str,
        values: Vec<String>,
        description: &str,
        required: bool,
    ) -> Self {
        self.parameters.properties.insert(
            name.to_string(),
            ToolParameter {
                param_type: "string".to_string(),
                description: description.to_string(),
                enum_values: Some(values),
            },
        );

        if required {
            self.parameters.required.push(name.to_string());
        }

        self
    }

    /// Convert to OpenAI function format.
    pub fn to_openai_format(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters.properties.iter().map(|(k, v)| {
                        (k.clone(), serde_json::json!({
                            "type": v.param_type,
                            "description": v.description,
                            "enum": v.enum_values,
                        }))
                    }).collect::<HashMap<_, _>>(),
                    "required": self.parameters.required,
                }
            }
        })
    }
}

/// Parameters schema for a tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolParameters {
    /// Parameter definitions.
    pub properties: HashMap<String, ToolParameter>,

    /// Required parameter names.
    pub required: Vec<String>,
}

/// A single parameter definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    /// Parameter type (string, number, boolean, array, object).
    pub param_type: String,

    /// Description of the parameter.
    pub description: String,

    /// Allowed values for enum parameters.
    pub enum_values: Option<Vec<String>>,
}

/// A call to a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this call.
    pub id: String,

    /// Name of the tool being called.
    pub name: String,

    /// Arguments as JSON.
    pub arguments: HashMap<String, serde_json::Value>,
}

impl ToolCall {
    /// Create a new tool call.
    pub fn new(name: &str) -> Self {
        Self {
            id: Self::generate_id(),
            name: name.to_string(),
            arguments: HashMap::new(),
        }
    }

    /// Create with a specific ID.
    pub fn with_id(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            arguments: HashMap::new(),
        }
    }

    /// Add a string argument.
    pub fn with_arg(mut self, name: &str, value: &str) -> Self {
        self.arguments.insert(
            name.to_string(),
            serde_json::Value::String(value.to_string()),
        );
        self
    }

    /// Add a numeric argument.
    pub fn with_number_arg(mut self, name: &str, value: f64) -> Self {
        self.arguments
            .insert(name.to_string(), serde_json::json!(value));
        self
    }

    /// Add a boolean argument.
    pub fn with_bool_arg(mut self, name: &str, value: bool) -> Self {
        self.arguments
            .insert(name.to_string(), serde_json::Value::Bool(value));
        self
    }

    /// Add a JSON argument.
    pub fn with_json_arg(mut self, name: &str, value: serde_json::Value) -> Self {
        self.arguments.insert(name.to_string(), value);
        self
    }

    /// Get argument as string.
    pub fn get_string(&self, name: &str) -> Option<&str> {
        self.arguments.get(name).and_then(|v| v.as_str())
    }

    /// Get argument as number.
    pub fn get_number(&self, name: &str) -> Option<f64> {
        self.arguments.get(name).and_then(|v| v.as_f64())
    }

    /// Get argument as boolean.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.arguments.get(name).and_then(|v| v.as_bool())
    }

    /// Parse from JSON string (as returned by model).
    pub fn parse(json: &str) -> Result<Self, ToolError> {
        serde_json::from_str(json).map_err(|e| ToolError::ParseError(e.to_string()))
    }

    /// Generate a unique ID.
    fn generate_id() -> String {
        format!(
            "call_{:016x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        )
    }
}

/// Result of a tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// ID of the original call.
    pub call_id: String,

    /// Tool name.
    pub name: String,

    /// Result content (usually JSON).
    pub content: String,

    /// Whether the call succeeded.
    pub success: bool,

    /// Error message if failed.
    pub error: Option<String>,
}

impl ToolResult {
    /// Create a successful result.
    pub fn success(call: &ToolCall, content: &str) -> Self {
        Self {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content: content.to_string(),
            success: true,
            error: None,
        }
    }

    /// Create a failed result.
    pub fn error(call: &ToolCall, error: &str) -> Self {
        Self {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content: String::new(),
            success: false,
            error: Some(error.to_string()),
        }
    }

    /// Create from JSON value.
    pub fn from_json(call: &ToolCall, value: serde_json::Value) -> Self {
        Self {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content: value.to_string(),
            success: true,
            error: None,
        }
    }
}

/// Registry of available tools.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    /// Registered tools by name.
    tools: HashMap<String, Tool>,

    /// Tool choice mode.
    pub choice: ToolChoice,
}

/// How the model should choose tools.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ToolChoice {
    /// Model decides whether to use tools.
    #[default]
    Auto,
    /// Model must use tools.
    Required,
    /// Model cannot use tools.
    None,
    /// Model must use a specific tool.
    Specific(String),
}

impl ToolRegistry {
    /// Create a new registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool.
    pub fn register(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Unregister a tool.
    pub fn unregister(&mut self, name: &str) -> Option<Tool> {
        self.tools.remove(name)
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    /// List all tool names.
    pub fn list(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered tools.
    pub fn count(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Set tool choice mode.
    pub fn set_choice(&mut self, choice: ToolChoice) {
        self.choice = choice;
    }

    /// Validate a tool call against registered tools.
    pub fn validate(&self, call: &ToolCall) -> Result<(), ToolError> {
        let tool = self
            .get(&call.name)
            .ok_or_else(|| ToolError::UnknownTool(call.name.clone()))?;

        // Check required parameters
        for required in &tool.parameters.required {
            if !call.arguments.contains_key(required) {
                return Err(ToolError::MissingParameter(required.clone()));
            }
        }

        // Validate parameter types
        for (name, value) in &call.arguments {
            if let Some(param) = tool.parameters.properties.get(name) {
                if !self.validate_type(value, &param.param_type) {
                    return Err(ToolError::InvalidParameterType {
                        name: name.clone(),
                        expected: param.param_type.clone(),
                    });
                }

                // Check enum constraint
                if let Some(ref enum_values) = param.enum_values {
                    if let Some(s) = value.as_str() {
                        if !enum_values.contains(&s.to_string()) {
                            return Err(ToolError::InvalidEnumValue {
                                name: name.clone(),
                                value: s.to_string(),
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn validate_type(&self, value: &serde_json::Value, expected: &str) -> bool {
        match expected {
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => value.is_i64() || value.is_u64(),
            "boolean" => value.is_boolean(),
            "array" => value.is_array(),
            "object" => value.is_object(),
            "null" => value.is_null(),
            _ => true, // Unknown type, allow
        }
    }

    /// Convert all tools to OpenAI format.
    pub fn to_openai_format(&self) -> Vec<serde_json::Value> {
        self.tools.values().map(|t| t.to_openai_format()).collect()
    }
}

/// Error type for tool operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ToolError {
    /// Unknown tool name.
    #[error("Unknown tool: {0}")]
    UnknownTool(String),

    /// Missing required parameter.
    #[error("Missing required parameter: {0}")]
    MissingParameter(String),

    /// Invalid parameter type.
    #[error("Invalid type for parameter {name}: expected {expected}")]
    InvalidParameterType { name: String, expected: String },

    /// Invalid enum value.
    #[error("Invalid value for parameter {name}: {value}")]
    InvalidEnumValue { name: String, value: String },

    /// Failed to parse tool call.
    #[error("Failed to parse tool call: {0}")]
    ParseError(String),

    /// Tool execution failed.
    #[error("Tool execution failed: {0}")]
    ExecutionError(String),
}

/// Parser for extracting tool calls from model output.
#[derive(Debug)]
pub struct ToolCallParser {
    /// Start marker for tool calls.
    pub start_marker: String,

    /// End marker for tool calls.
    pub end_marker: String,
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self {
            start_marker: "<tool_call>".to_string(),
            end_marker: "</tool_call>".to_string(),
        }
    }
}

impl ToolCallParser {
    /// Create a parser with custom markers.
    pub fn new(start: &str, end: &str) -> Self {
        Self {
            start_marker: start.to_string(),
            end_marker: end.to_string(),
        }
    }

    /// Extract tool calls from text.
    pub fn parse(&self, text: &str) -> Vec<Result<ToolCall, ToolError>> {
        let mut results = Vec::new();

        let mut remaining = text;
        while let Some(start) = remaining.find(&self.start_marker) {
            remaining = &remaining[start + self.start_marker.len()..];

            if let Some(end) = remaining.find(&self.end_marker) {
                let json = &remaining[..end];
                results.push(ToolCall::parse(json.trim()));
                remaining = &remaining[end + self.end_marker.len()..];
            } else {
                break;
            }
        }

        results
    }

    /// Check if text contains tool calls.
    pub fn contains_tool_call(&self, text: &str) -> bool {
        text.contains(&self.start_marker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_creation() {
        let tool = Tool::new("test_tool")
            .description("A test tool")
            .add_param("arg1", "string", "First argument", true)
            .add_param("arg2", "number", "Second argument", false);

        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.parameters.properties.len(), 2);
        assert_eq!(tool.parameters.required.len(), 1);
    }

    #[test]
    fn test_tool_enum_param() {
        let tool = Tool::new("color_picker").add_enum_param(
            "color",
            vec!["red".to_string(), "green".to_string(), "blue".to_string()],
            "Choose a color",
            true,
        );

        let param = tool.parameters.properties.get("color").unwrap();
        assert!(param.enum_values.is_some());
        assert_eq!(param.enum_values.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_tool_call_creation() {
        let call = ToolCall::new("get_weather")
            .with_arg("location", "San Francisco")
            .with_number_arg("temperature", 72.5)
            .with_bool_arg("metric", true);

        assert_eq!(call.name, "get_weather");
        assert_eq!(call.get_string("location"), Some("San Francisco"));
        assert_eq!(call.get_number("temperature"), Some(72.5));
        assert_eq!(call.get_bool("metric"), Some(true));
    }

    #[test]
    fn test_tool_result_success() {
        let call = ToolCall::new("test");
        let result = ToolResult::success(&call, r#"{"status": "ok"}"#);

        assert!(result.success);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_tool_result_error() {
        let call = ToolCall::new("test");
        let result = ToolResult::error(&call, "Something went wrong");

        assert!(!result.success);
        assert_eq!(result.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();

        let tool1 = Tool::new("tool1").description("First tool");
        let tool2 = Tool::new("tool2").description("Second tool");

        registry.register(tool1);
        registry.register(tool2);

        assert_eq!(registry.count(), 2);
        assert!(registry.get("tool1").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_validate_success() {
        let mut registry = ToolRegistry::new();
        let tool = Tool::new("greet").add_param("name", "string", "Name to greet", true);
        registry.register(tool);

        let call = ToolCall::new("greet").with_arg("name", "Alice");
        assert!(registry.validate(&call).is_ok());
    }

    #[test]
    fn test_registry_validate_missing_param() {
        let mut registry = ToolRegistry::new();
        let tool = Tool::new("greet").add_param("name", "string", "Name to greet", true);
        registry.register(tool);

        let call = ToolCall::new("greet");
        let result = registry.validate(&call);
        assert!(matches!(result, Err(ToolError::MissingParameter(_))));
    }

    #[test]
    fn test_registry_validate_unknown_tool() {
        let registry = ToolRegistry::new();
        let call = ToolCall::new("unknown");
        let result = registry.validate(&call);
        assert!(matches!(result, Err(ToolError::UnknownTool(_))));
    }

    #[test]
    fn test_tool_call_parser() {
        let parser = ToolCallParser::default();

        let text = r#"Here is the result:
<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>
Done."#;

        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_tool_call_parser_contains() {
        let parser = ToolCallParser::default();

        assert!(parser.contains_tool_call("<tool_call>{}</tool_call>"));
        assert!(!parser.contains_tool_call("No tool calls here"));
    }

    #[test]
    fn test_tool_to_openai_format() {
        let tool = Tool::new("search").description("Search the web").add_param(
            "query",
            "string",
            "Search query",
            true,
        );

        let json = tool.to_openai_format();
        assert_eq!(json["function"]["name"], "search");
    }
}
