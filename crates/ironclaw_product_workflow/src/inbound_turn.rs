//! InboundTurnService — the user-message turn submission path.
//!
//! This is the narrower user-message subset of [`ProductWorkflow`]. It
//! resolves the conversation binding, accepts the inbound message into the
//! session thread, and submits the turn to the coordinator.

use async_trait::async_trait;
use ironclaw_product_adapters::{ProductInboundAck, ProductInboundEnvelope};
use ironclaw_turns::{AcceptedMessageRef, TurnRunId};
use ironclaw_turns::{
    IdempotencyKey, ReplyTargetBindingRef, SourceBindingRef, SubmitTurnRequest,
    SubmitTurnResponse, TurnActor, TurnCoordinator, TurnScope,
};

use crate::binding::{ConversationBindingService, ResolveBindingRequest, ResolvedBinding};
use crate::error::ProductWorkflowError;

/// Result of the inbound turn submission flow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InboundTurnOutcome {
    /// Turn was accepted and submitted to the coordinator.
    Submitted {
        accepted_message_ref: AcceptedMessageRef,
        submitted_run_id: TurnRunId,
        binding: ResolvedBinding,
    },
    /// Turn submission was busy (thread already has an active run). The message
    /// was accepted but deferred.
    DeferredBusy {
        accepted_message_ref: AcceptedMessageRef,
        active_run_id: TurnRunId,
        binding: ResolvedBinding,
    },
}

impl InboundTurnOutcome {
    /// Convert to a product-safe acknowledgement for the adapter.
    pub fn to_ack(&self) -> ProductInboundAck {
        match self {
            Self::Submitted {
                accepted_message_ref,
                submitted_run_id,
                ..
            } => ProductInboundAck::Accepted {
                accepted_message_ref: accepted_message_ref.clone(),
                submitted_run_id: *submitted_run_id,
            },
            Self::DeferredBusy {
                accepted_message_ref,
                active_run_id,
                ..
            } => ProductInboundAck::DeferredBusy {
                accepted_message_ref: accepted_message_ref.clone(),
                active_run_id: *active_run_id,
            },
        }
    }
}

/// Port for the inbound turn submission path.
///
/// Implementations coordinate binding resolution, message acceptance into the
/// session thread service, and turn submission to the coordinator.
#[async_trait]
pub trait InboundTurnService: Send + Sync {
    /// Accept a user message envelope: resolve binding, stage message, submit turn.
    async fn accept_user_message(
        &self,
        envelope: &ProductInboundEnvelope,
    ) -> Result<InboundTurnOutcome, ProductWorkflowError>;
}

/// Default implementation that composes a [`ConversationBindingService`] with a
/// [`TurnCoordinator`].
pub struct DefaultInboundTurnService<B, C> {
    binding_service: B,
    turn_coordinator: C,
}

impl<B, C> DefaultInboundTurnService<B, C>
where
    B: ConversationBindingService,
    C: TurnCoordinator,
{
    pub fn new(binding_service: B, turn_coordinator: C) -> Self {
        Self {
            binding_service,
            turn_coordinator,
        }
    }
}

#[async_trait]
impl<B, C> InboundTurnService for DefaultInboundTurnService<B, C>
where
    B: ConversationBindingService,
    C: TurnCoordinator,
{
    async fn accept_user_message(
        &self,
        envelope: &ProductInboundEnvelope,
    ) -> Result<InboundTurnOutcome, ProductWorkflowError> {
        // Step 1: Resolve conversation binding.
        let binding = self
            .binding_service
            .resolve_binding(ResolveBindingRequest {
                adapter_id: envelope.adapter_id().clone(),
                installation_id: envelope.installation_id().clone(),
                external_actor_ref: envelope.external_actor_ref().clone(),
                external_conversation_ref: envelope.external_conversation_ref().clone(),
                auth_claim: envelope.auth_claim().clone(),
            })
            .await?;

        // Step 2: Build the turn submission request.
        let scope = TurnScope::new(
            binding.tenant_id.clone(),
            binding.agent_id.clone(),
            binding.project_id.clone(),
            binding.thread_id.clone(),
        );
        let actor = TurnActor::new(binding.user_id.clone());
        let source_binding_ref =
            SourceBindingRef::new(envelope.source_binding_key()).map_err(|e| {
                ProductWorkflowError::BindingResolutionFailed {
                    reason: format!("invalid source binding ref: {e}"),
                }
            })?;
        let accepted_message_ref = AcceptedMessageRef::new(format!(
            "msg:{}",
            envelope.external_event_id()
        ))
        .map_err(|e| ProductWorkflowError::TurnSubmissionRejected {
            reason: format!("invalid accepted message ref: {e}"),
        })?;
        let reply_target_binding_ref = ReplyTargetBindingRef::new(format!(
            "reply:{}",
            envelope.source_binding_key()
        ))
        .map_err(|e| ProductWorkflowError::TurnSubmissionRejected {
            reason: format!("invalid reply target binding ref: {e}"),
        })?;
        let idempotency_key = IdempotencyKey::new(format!(
            "turn:{}:{}:{}",
            envelope.adapter_id(),
            envelope.installation_id(),
            envelope.external_event_id()
        ))
        .map_err(|e| ProductWorkflowError::TurnSubmissionRejected {
            reason: format!("invalid idempotency key: {e}"),
        })?;

        let request = SubmitTurnRequest {
            scope,
            actor,
            accepted_message_ref: accepted_message_ref.clone(),
            source_binding_ref,
            reply_target_binding_ref,
            requested_run_profile: None,
            idempotency_key,
            received_at: envelope.received_at(),
        };

        // Step 3: Submit to turn coordinator.
        let response = self.turn_coordinator.submit_turn(request).await.map_err(|e| {
            ProductWorkflowError::TurnSubmissionRejected {
                reason: e.to_string(),
            }
        })?;

        match response {
            SubmitTurnResponse::Accepted { run_id, .. } => Ok(InboundTurnOutcome::Submitted {
                accepted_message_ref,
                submitted_run_id: run_id,
                binding,
            }),
        }
    }
}
