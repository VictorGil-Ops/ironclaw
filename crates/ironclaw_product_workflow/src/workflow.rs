//! Host-side `ProductWorkflow` implementation.
//!
//! This is the top-level product action orchestrator that dispatches inbound
//! envelopes to the appropriate downstream service based on payload kind.

use std::sync::Arc;

use async_trait::async_trait;
use ironclaw_product_adapters::{
    ProductAdapterError, ProductInboundAck, ProductInboundEnvelope, ProductInboundPayload,
    ProductWorkflow, ProjectionSubscriptionRequest,
};
use tracing::{debug, warn};

use crate::action::{ActionDispatchKind, ActionFingerprintKey};
use crate::error::ProductWorkflowError;
use crate::inbound_turn::InboundTurnService;
use crate::ledger::{IdempotencyDecision, IdempotencyLedger};

/// Host-side implementation of [`ProductWorkflow`] that dispatches inbound
/// envelopes through the idempotency ledger and routes to the appropriate
/// downstream service.
pub struct DefaultProductWorkflow {
    inbound_turn_service: Arc<dyn InboundTurnService>,
    idempotency_ledger: Arc<dyn IdempotencyLedger>,
}

impl DefaultProductWorkflow {
    pub fn new(
        inbound_turn_service: Arc<dyn InboundTurnService>,
        idempotency_ledger: Arc<dyn IdempotencyLedger>,
    ) -> Self {
        Self {
            inbound_turn_service,
            idempotency_ledger,
        }
    }
}

#[async_trait]
impl ProductWorkflow for DefaultProductWorkflow {
    async fn accept_inbound(
        &self,
        envelope: ProductInboundEnvelope,
    ) -> Result<ProductInboundAck, ProductAdapterError> {
        // Step 1: Check idempotency ledger.
        let fingerprint = ActionFingerprintKey::new(
            envelope.adapter_id(),
            envelope.installation_id(),
            &envelope.source_binding_key(),
            envelope.external_event_id(),
        );

        let decision = self
            .idempotency_ledger
            .begin_or_replay(fingerprint, envelope.received_at())
            .await
            .map_err(ProductAdapterError::from)?;

        match decision {
            IdempotencyDecision::Replay(action) => {
                debug!(
                    action_id = %action.action_id,
                    "replaying prior settled action"
                );
                if let Some(prior_outcome) = action.outcome {
                    return Ok(ProductInboundAck::Duplicate {
                        prior: Box::new(prior_outcome),
                    });
                }
                // Settled but no outcome recorded — treat as internal error.
                return Err(ProductAdapterError::Internal {
                    detail: ironclaw_product_adapters::RedactedString::new(
                        "settled action missing outcome",
                    ),
                });
            }
            IdempotencyDecision::New(mut action) => {
                // Step 2: Dispatch based on payload kind.
                let result = dispatch_payload(
                    &envelope,
                    &*self.inbound_turn_service,
                )
                .await;

                match result {
                    Ok(ack) => {
                        let dispatch_kind =
                            ActionDispatchKind::from_payload(envelope.payload());
                        action.mark_dispatched(dispatch_kind);
                        action.settle(ack.clone());
                        if let Err(e) = self.idempotency_ledger.settle(action).await {
                            warn!(error = %e, "failed to settle idempotency ledger entry");
                        }
                        Ok(ack)
                    }
                    Err(e) => Err(ProductAdapterError::from(e)),
                }
            }
        }
    }

    async fn resolve_projection_subscription(
        &self,
        _envelope: ProductInboundEnvelope,
    ) -> Result<ProjectionSubscriptionRequest, ProductAdapterError> {
        // Projection subscription resolution is a read-only operation.
        // Full implementation deferred to #3281 (EventStreamManager).
        Err(ProductAdapterError::Internal {
            detail: ironclaw_product_adapters::RedactedString::new(
                "projection subscription resolution not yet implemented",
            ),
        })
    }
}

async fn dispatch_payload(
    envelope: &ProductInboundEnvelope,
    inbound_turn_service: &dyn InboundTurnService,
) -> Result<ProductInboundAck, ProductWorkflowError> {
    match envelope.payload() {
        ProductInboundPayload::UserMessage(_) => {
            let outcome = inbound_turn_service
                .accept_user_message(envelope)
                .await?;
            Ok(outcome.to_ack())
        }
        ProductInboundPayload::Command(cmd) => {
            // Command routing seam — full matrix in #3286.
            Err(ProductWorkflowError::CommandRoutingUnavailable {
                command: cmd.command.clone(),
            })
        }
        ProductInboundPayload::ApprovalResolution(_) => {
            // Approval/auth interaction services — future work.
            Err(ProductWorkflowError::UnsupportedActionKind {
                kind: "approval_resolution".into(),
            })
        }
        ProductInboundPayload::AuthResolution(_) => {
            // Auth resolution interaction services — future work.
            Err(ProductWorkflowError::UnsupportedActionKind {
                kind: "auth_resolution".into(),
            })
        }
        ProductInboundPayload::SubscriptionRequest(_) => {
            // Read/subscription dispatch — #3281 owns streams.
            Err(ProductWorkflowError::UnsupportedActionKind {
                kind: "subscription_request".into(),
            })
        }
        ProductInboundPayload::LinkedThreadAction(_) => {
            // Linked thread action — future work.
            Err(ProductWorkflowError::UnsupportedActionKind {
                kind: "linked_thread_action".into(),
            })
        }
        ProductInboundPayload::NoOp => Ok(ProductInboundAck::NoOp),
    }
}
