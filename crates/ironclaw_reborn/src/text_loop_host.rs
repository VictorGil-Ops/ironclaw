//! Reborn-side text-only loop host composition.
//!
//! This module keeps the concrete Reborn loop-support wiring out of the root
//! `/src` app graph while giving callers one small factory for the context,
//! prompt, model, transcript, and empty capability ports needed by the text-only
//! loop path.

use std::sync::Arc;

use async_trait::async_trait;
use ironclaw_loop_support::{
    EmptyLoopCapabilityPort, HostManagedModelGateway, ThreadBackedLoopContextPort,
    ThreadBackedLoopModelPort, ThreadBackedLoopTranscriptPort,
};
use ironclaw_threads::{SessionThreadService, ThreadScope};
use ironclaw_turns::{
    LoopMessageRef, TurnCheckpointId,
    run_profile::{
        AgentLoopHostError, AgentLoopHostErrorKind, AppendCapabilityResultRef, BeginAssistantDraft,
        CapabilityBatchInvocation, CapabilityBatchOutcome, CapabilityInvocation, CapabilityOutcome,
        FinalizeAssistantMessage, HostManagedLoopPromptPort, LoopCapabilityPort,
        LoopCheckpointPort, LoopCheckpointRequest, LoopContextBundle, LoopContextPort,
        LoopContextRequest, LoopHostMilestoneSink, LoopInputBatch, LoopInputCursor, LoopInputPort,
        LoopModelPort, LoopModelRequest, LoopModelResponse, LoopProgressEvent, LoopProgressPort,
        LoopPromptBundle, LoopPromptBundleRequest, LoopPromptPort, LoopRunContext, LoopRunInfoPort,
        LoopTranscriptPort, UpdateAssistantDraft, VisibleCapabilityRequest,
        VisibleCapabilitySurface,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextOnlyLoopHostConfig {
    pub max_context_messages: usize,
    pub max_model_messages: usize,
}

impl Default for TextOnlyLoopHostConfig {
    fn default() -> Self {
        Self {
            max_context_messages: 16,
            max_model_messages: 16,
        }
    }
}

pub struct TextOnlyLoopHostPorts<S, G>
where
    S: SessionThreadService + ?Sized,
    G: HostManagedModelGateway + ?Sized,
{
    pub context: Arc<ThreadBackedLoopContextPort<S>>,
    pub prompt:
        HostManagedLoopPromptPort<ThreadBackedLoopContextPort<S>, dyn LoopHostMilestoneSink>,
    pub model: ThreadBackedLoopModelPort<S, G>,
    pub transcript: ThreadBackedLoopTranscriptPort<S>,
    pub capabilities: EmptyLoopCapabilityPort,
}

impl<S, G> TextOnlyLoopHostPorts<S, G>
where
    S: SessionThreadService + ?Sized,
    G: HostManagedModelGateway + ?Sized,
{
    pub fn new(
        thread_service: Arc<S>,
        thread_scope: ThreadScope,
        run_context: LoopRunContext,
        gateway: Arc<G>,
        milestone_sink: Arc<dyn LoopHostMilestoneSink>,
        config: TextOnlyLoopHostConfig,
    ) -> Self {
        let context = Arc::new(ThreadBackedLoopContextPort::new(
            Arc::clone(&thread_service),
            thread_scope.clone(),
            run_context.clone(),
            config.max_context_messages,
        ));
        let prompt = HostManagedLoopPromptPort::new(
            run_context.clone(),
            Arc::clone(&context),
            Arc::clone(&milestone_sink),
        )
        .with_default_message_limit(config.max_context_messages);
        Self {
            context,
            prompt,
            model: ThreadBackedLoopModelPort::with_milestone_sink(
                Arc::clone(&thread_service),
                thread_scope.clone(),
                run_context.clone(),
                gateway,
                config.max_model_messages,
                Arc::clone(&milestone_sink),
            ),
            transcript: ThreadBackedLoopTranscriptPort::with_milestone_sink(
                thread_service,
                thread_scope,
                run_context,
                milestone_sink,
            ),
            capabilities: EmptyLoopCapabilityPort,
        }
    }
}

pub struct TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized,
    G: HostManagedModelGateway + ?Sized,
{
    pub ports: TextOnlyLoopHostPorts<S, G>,
}

impl<S, G> TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized,
    G: HostManagedModelGateway + ?Sized,
{
    pub fn new(
        thread_service: Arc<S>,
        thread_scope: ThreadScope,
        run_context: LoopRunContext,
        gateway: Arc<G>,
        milestone_sink: Arc<dyn LoopHostMilestoneSink>,
        config: TextOnlyLoopHostConfig,
    ) -> Self {
        Self {
            ports: TextOnlyLoopHostPorts::new(
                thread_service,
                thread_scope,
                run_context,
                gateway,
                milestone_sink,
                config,
            ),
        }
    }

    pub fn from_ports(ports: TextOnlyLoopHostPorts<S, G>) -> Self {
        Self { ports }
    }
}

impl<S, G> LoopRunInfoPort for TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized + Send + Sync,
    G: HostManagedModelGateway + ?Sized + Send + Sync,
{
    fn run_context(&self) -> &LoopRunContext {
        self.ports.context.run_context()
    }
}

#[async_trait]
impl<S, G> LoopContextPort for TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized + Send + Sync,
    G: HostManagedModelGateway + ?Sized + Send + Sync,
{
    async fn load_loop_context(
        &self,
        request: LoopContextRequest,
    ) -> Result<LoopContextBundle, AgentLoopHostError> {
        self.ports.context.load_loop_context(request).await
    }
}

#[async_trait]
impl<S, G> LoopPromptPort for TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized + Send + Sync,
    G: HostManagedModelGateway + ?Sized + Send + Sync,
{
    async fn build_prompt_bundle(
        &self,
        request: LoopPromptBundleRequest,
    ) -> Result<LoopPromptBundle, AgentLoopHostError> {
        self.ports.prompt.build_prompt_bundle(request).await
    }
}

#[async_trait]
impl<S, G> LoopModelPort for TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized + Send + Sync,
    G: HostManagedModelGateway + ?Sized + Send + Sync,
{
    async fn stream_model(
        &self,
        request: LoopModelRequest,
    ) -> Result<LoopModelResponse, AgentLoopHostError> {
        self.ports.model.stream_model(request).await
    }
}

#[async_trait]
impl<S, G> LoopCapabilityPort for TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized + Send + Sync,
    G: HostManagedModelGateway + ?Sized + Send + Sync,
{
    async fn visible_capabilities(
        &self,
        request: VisibleCapabilityRequest,
    ) -> Result<VisibleCapabilitySurface, AgentLoopHostError> {
        self.ports.capabilities.visible_capabilities(request).await
    }

    async fn invoke_capability(
        &self,
        request: CapabilityInvocation,
    ) -> Result<CapabilityOutcome, AgentLoopHostError> {
        self.ports.capabilities.invoke_capability(request).await
    }

    async fn invoke_capability_batch(
        &self,
        request: CapabilityBatchInvocation,
    ) -> Result<CapabilityBatchOutcome, AgentLoopHostError> {
        self.ports
            .capabilities
            .invoke_capability_batch(request)
            .await
    }
}

#[async_trait]
impl<S, G> LoopTranscriptPort for TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized + Send + Sync,
    G: HostManagedModelGateway + ?Sized + Send + Sync,
{
    async fn begin_assistant_draft(
        &self,
        request: BeginAssistantDraft,
    ) -> Result<LoopMessageRef, AgentLoopHostError> {
        self.ports.transcript.begin_assistant_draft(request).await
    }

    async fn update_assistant_draft(
        &self,
        request: UpdateAssistantDraft,
    ) -> Result<(), AgentLoopHostError> {
        self.ports.transcript.update_assistant_draft(request).await
    }

    async fn finalize_assistant_message(
        &self,
        request: FinalizeAssistantMessage,
    ) -> Result<LoopMessageRef, AgentLoopHostError> {
        self.ports
            .transcript
            .finalize_assistant_message(request)
            .await
    }

    async fn append_capability_result_ref(
        &self,
        request: AppendCapabilityResultRef,
    ) -> Result<LoopMessageRef, AgentLoopHostError> {
        self.ports
            .transcript
            .append_capability_result_ref(request)
            .await
    }
}

#[async_trait]
impl<S, G> LoopInputPort for TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized + Send + Sync,
    G: HostManagedModelGateway + ?Sized + Send + Sync,
{
    async fn poll_inputs(
        &self,
        _after: LoopInputCursor,
        _limit: usize,
    ) -> Result<LoopInputBatch, AgentLoopHostError> {
        Err(unsupported_text_only_host_method("poll_inputs"))
    }

    async fn ack_inputs(&self, _cursor: LoopInputCursor) -> Result<(), AgentLoopHostError> {
        Err(unsupported_text_only_host_method("ack_inputs"))
    }
}

#[async_trait]
impl<S, G> LoopCheckpointPort for TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized + Send + Sync,
    G: HostManagedModelGateway + ?Sized + Send + Sync,
{
    async fn checkpoint(
        &self,
        _request: LoopCheckpointRequest,
    ) -> Result<TurnCheckpointId, AgentLoopHostError> {
        Err(unsupported_text_only_host_method("checkpoint"))
    }
}

#[async_trait]
impl<S, G> LoopProgressPort for TextOnlyLoopHost<S, G>
where
    S: SessionThreadService + ?Sized + Send + Sync,
    G: HostManagedModelGateway + ?Sized + Send + Sync,
{
    async fn emit_loop_progress(
        &self,
        _event: LoopProgressEvent,
    ) -> Result<(), AgentLoopHostError> {
        Err(unsupported_text_only_host_method("emit_loop_progress"))
    }
}

fn unsupported_text_only_host_method(method: &'static str) -> AgentLoopHostError {
    AgentLoopHostError::new(
        AgentLoopHostErrorKind::Unavailable,
        format!("text-only loop host method {method} is unavailable"),
    )
}
