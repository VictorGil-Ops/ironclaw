//! Standalone Reborn composition and adapter wiring.
//!
//! This crate is the Reborn-side home for adapters that intentionally bridge
//! to existing root IronClaw services while keeping the normal `/src` app graph
//! free of Reborn loop-support wiring.

pub mod driver_registry;

#[cfg(feature = "root-llm-provider")]
pub mod model_gateway;
pub mod text_loop_driver;
pub mod text_loop_host;

#[cfg(feature = "root-llm-provider")]
pub use model_gateway::{
    LlmModelProfilePolicy, LlmProviderModelGateway, ThreadBackedLoopModelGateway,
};
pub use text_loop_driver::{TextOnlyModelReplyDriver, TextOnlyModelReplyDriverConfig};
pub use text_loop_host::{TextOnlyLoopHost, TextOnlyLoopHostConfig, TextOnlyLoopHostPorts};
