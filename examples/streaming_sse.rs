//! Streaming SSE Example
//!
//! Demonstrates TGI's Server-Sent Events streaming for
//! token-by-token delivery.
//!
//! # Run
//!
//! ```bash
//! cargo run --example streaming_sse
//! ```

use tgi_gtc::streaming::{CompleteEvent, FinishReason, SseFormatter, TokenEvent};

fn main() {
    println!("=== SSE Streaming Demo ===\n");

    // Create formatter with options
    let mut formatter = SseFormatter::new();
    formatter.include_token_ids = true;
    formatter.include_logprobs = true;

    println!("Formatter Configuration:");
    println!("  Include token IDs: {}", formatter.include_token_ids);
    println!("  Include logprobs: {}", formatter.include_logprobs);
    println!();

    // Simulate token generation
    let tokens = [
        ("Hello", 1234, Some(-50), false),
        (",", 44, Some(-10), false),
        (" world", 5678, Some(-75), false),
        ("!", 999, Some(-25), false),
        ("<|endoftext|>", 0, None, true),
    ];

    println!("Simulated Token Stream:");
    println!("{}", "=".repeat(60));

    for (token, id, logprob, special) in tokens {
        let event = TokenEvent {
            token: token.to_string(),
            token_id: id,
            logprob,
            special,
        };

        let sse = formatter.format_token(&event);
        print!("{}", sse);
    }

    // Send completion event
    let complete = CompleteEvent {
        generated_tokens: 5,
        finish_reason: FinishReason::EndOfSequence,
        generation_time_ms: 125,
    };

    let sse = formatter.format_complete(&complete);
    print!("{}", sse);

    println!("{}", "=".repeat(60));
    println!();

    // Demonstrate error formatting
    println!("Error Event Example:");
    println!("{}", "-".repeat(40));
    let error_sse = formatter.format_error("Model inference failed: out of memory");
    print!("{}", error_sse);
    println!("{}", "-".repeat(40));

    // Show finish reasons
    println!("\nFinish Reasons:");
    println!("  Length: {}", FinishReason::Length);
    println!("  Stop: {}", FinishReason::Stop);
    println!("  EOS: {}", FinishReason::EndOfSequence);

    println!("\n=== Demo Complete ===");
}
