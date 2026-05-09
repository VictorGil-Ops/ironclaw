//! Contract tests for the InboundTurnService.

use std::sync::Arc;

use chrono::Utc;
use ironclaw_product_adapters::{
    AdapterInstallationId, AuthRequirement, ExternalActorRef, ExternalConversationRef,
    ExternalEventId, ParsedProductInbound, ProductAdapterId, ProductInboundEnvelope,
    ProductInboundPayload, ProductTriggerReason, ProtocolAuthEvidence, TrustedInboundContext,
    UserMessagePayload,
};
use ironclaw_product_workflow::{
    DefaultInboundTurnService, FakeConversationBindingService, InboundTurnOutcome,
    InboundTurnService, ProductWorkflowError,
};
use ironclaw_turns::{DefaultTurnCoordinator, InMemoryTurnStateStore};

fn sample_user_message_envelope(event_suffix: &str) -> ProductInboundEnvelope {
    let evidence = ProtocolAuthEvidence::test_verified(
        AuthRequirement::SharedSecretHeader {
            header_name: "X-Secret".into(),
        },
        "install_alpha",
    );
    let context = TrustedInboundContext::from_verified_evidence(
        ProductAdapterId::new("test_adapter").expect("valid"),
        AdapterInstallationId::new("install_alpha").expect("valid"),
        Utc::now(),
        &evidence,
    )
    .expect("verified");

    let parsed = ParsedProductInbound::new(
        ExternalEventId::new(format!("evt:{event_suffix}")).expect("valid"),
        ExternalActorRef::new("test", "user1", Option::<String>::None).expect("valid"),
        ExternalConversationRef::new(None, "conv1", None, None).expect("valid"),
        ProductInboundPayload::UserMessage(
            UserMessagePayload::new("hello world", vec![], ProductTriggerReason::DirectChat)
                .expect("valid"),
        ),
    )
    .expect("parsed");

    ProductInboundEnvelope::from_trusted_parse(context, parsed).expect("envelope")
}

#[tokio::test]
async fn user_message_resolves_binding_and_submits_turn() {
    let binding_service = FakeConversationBindingService::new();
    let store = Arc::new(InMemoryTurnStateStore::default());
    let coordinator = DefaultTurnCoordinator::new(store);
    let service = DefaultInboundTurnService::new(binding_service, coordinator);

    let envelope = sample_user_message_envelope("turn1");
    let outcome: InboundTurnOutcome = service.accept_user_message(&envelope).await.expect("submit");

    match &outcome {
        InboundTurnOutcome::Submitted { binding, .. } => {
            assert!(!binding.tenant_id.as_str().is_empty());
            assert!(!binding.user_id.as_str().is_empty());
            assert!(!binding.thread_id.as_str().is_empty());
        }
        _ => panic!("expected Submitted, got {outcome:?}"),
    }

    let ack = outcome.to_ack();
    assert!(matches!(
        ack,
        ironclaw_product_adapters::ProductInboundAck::Accepted { .. }
    ));
}

#[tokio::test]
async fn binding_failure_surfaces_workflow_error() {
    let binding_service = FakeConversationBindingService::new();
    binding_service.force_failure(ProductWorkflowError::BindingResolutionFailed {
        reason: "no tenant found".into(),
    });

    let store = Arc::new(InMemoryTurnStateStore::default());
    let coordinator = DefaultTurnCoordinator::new(store);
    let service = DefaultInboundTurnService::new(binding_service, coordinator);

    let envelope = sample_user_message_envelope("fail1");
    let err = service
        .accept_user_message(&envelope)
        .await
        .expect_err("should fail");

    assert!(matches!(
        err,
        ProductWorkflowError::BindingResolutionFailed { .. }
    ));
}
