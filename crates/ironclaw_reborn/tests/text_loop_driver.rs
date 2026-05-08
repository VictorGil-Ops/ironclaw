use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ironclaw_host_api::{AgentId, CapabilityId, ProjectId, TenantId, ThreadId, UserId};
use ironclaw_loop_support::{
    HostManagedModelError, HostManagedModelErrorKind, HostManagedModelGateway,
    HostManagedModelRequest, HostManagedModelResponse,
};
use ironclaw_reborn::{TextOnlyLoopHost, TextOnlyLoopHostConfig, TextOnlyModelReplyDriver};
use ironclaw_threads::{
    AcceptInboundMessageRequest, EnsureThreadRequest, InMemorySessionThreadService, MessageContent,
    MessageKind, MessageStatus, SessionThreadService, ThreadHistoryRequest, ThreadScope,
};
use ironclaw_turns::{
    LoopCompletionKind, LoopExit, RunProfileResolutionRequest, RunProfileResolver, TurnId,
    TurnRunId, TurnScope,
    run_profile::{
        AgentLoopDriver, AgentLoopDriverError, AgentLoopDriverRunRequest, CapabilityCallCandidate,
        CapabilityInputRef, CapabilitySurfaceVersion, InMemoryLoopHostMilestoneSink,
        InMemoryRunProfileResolver, LoopHostMilestoneKind, LoopRunContext, ParentLoopOutput,
    },
};

#[tokio::test]
async fn text_only_model_reply_driver_runs_context_model_transcript_path() {
    let fixture = ThreadFixture::new_with_user_content(
        "RAW_PROMPT_TEXT_SENTINEL sk-prompt-secret /host/path tool_input",
    )
    .await;
    let milestone_sink = Arc::new(InMemoryLoopHostMilestoneSink::default());
    let gateway = Arc::new(RecordingGateway::reply(
        "RAW_ASSISTANT_CONTENT_SENTINEL sk-output-secret /host/path tool_input",
    ));
    let host = TextOnlyLoopHost::new(
        Arc::clone(&fixture.thread_service),
        fixture.thread_scope.clone(),
        fixture.run_context.clone(),
        gateway.clone(),
        milestone_sink.clone(),
        TextOnlyLoopHostConfig::default(),
    );
    let driver = TextOnlyModelReplyDriver::default();

    let exit = driver
        .run(driver_request(&fixture.run_context), &host)
        .await
        .unwrap();

    let LoopExit::Completed(completed) = exit else {
        panic!("expected completed final reply exit");
    };
    assert_eq!(completed.completion_kind, LoopCompletionKind::FinalReply);
    assert_eq!(completed.result_refs, Vec::new());
    assert_eq!(completed.reply_message_refs.len(), 1);
    let reply_ref = completed.reply_message_refs[0].clone();

    let history = fixture
        .thread_service
        .list_thread_history(ThreadHistoryRequest {
            scope: fixture.thread_scope.clone(),
            thread_id: fixture.thread_id.clone(),
        })
        .await
        .unwrap();
    let assistant = history
        .messages
        .iter()
        .find(|message| message.kind == MessageKind::Assistant)
        .expect("driver must persist assistant reply through transcript port");
    assert_eq!(assistant.status, MessageStatus::Finalized);
    assert_eq!(reply_ref.as_str(), format!("msg:{}", assistant.message_id));
    assert_eq!(
        assistant.content.as_deref(),
        Some("RAW_ASSISTANT_CONTENT_SENTINEL sk-output-secret /host/path tool_input")
    );

    let requests = gateway.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].messages.len(), 1);
    assert_eq!(
        requests[0].messages[0].content,
        "RAW_PROMPT_TEXT_SENTINEL sk-prompt-secret /host/path tool_input"
    );
    drop(requests);

    let milestones = milestone_sink.milestones();
    assert_eq!(milestones.len(), 4);
    assert_eq!(milestones[0].kind.kind_name(), "prompt_bundle_built");
    assert!(matches!(
        &milestones[1].kind,
        LoopHostMilestoneKind::ModelStarted {
            requested_model_profile_id: None
        }
    ));
    assert!(matches!(
        &milestones[2].kind,
        LoopHostMilestoneKind::ModelCompleted { effective_model_profile_id }
            if effective_model_profile_id == &fixture.run_context.resolved_run_profile.model_profile_id
    ));
    assert!(matches!(
        &milestones[3].kind,
        LoopHostMilestoneKind::AssistantReplyFinalized { message_ref }
            if message_ref == &reply_ref
    ));
    assert_serialized_outputs_hide_sentinels(&milestones);
    assert_serialized_outputs_hide_sentinels(&completed);
}

#[tokio::test]
async fn text_only_model_reply_driver_builds_prompt_bundle_before_model_request() {
    let fixture = ThreadFixture::new_with_user_content("prompt bundle source").await;
    let milestone_sink = Arc::new(InMemoryLoopHostMilestoneSink::default());
    let gateway = Arc::new(RecordingGateway::reply("model reply"));
    let host = TextOnlyLoopHost::new(
        Arc::clone(&fixture.thread_service),
        fixture.thread_scope.clone(),
        fixture.run_context.clone(),
        gateway,
        milestone_sink.clone(),
        TextOnlyLoopHostConfig::default(),
    );
    let driver = TextOnlyModelReplyDriver::new(ironclaw_reborn::TextOnlyModelReplyDriverConfig {
        context_limit: 1,
    });

    driver
        .run(driver_request(&fixture.run_context), &host)
        .await
        .unwrap();

    assert_eq!(
        milestone_sink
            .milestones()
            .iter()
            .map(|milestone| milestone.kind.kind_name())
            .collect::<Vec<_>>(),
        vec![
            "prompt_bundle_built",
            "model_started",
            "model_completed",
            "assistant_reply_finalized",
        ]
    );
}

#[tokio::test]
async fn text_only_model_reply_driver_rejects_mismatched_run_profile_driver() {
    let fixture = ThreadFixture::new_with_user_content("hello mismatch").await;
    let milestone_sink = Arc::new(InMemoryLoopHostMilestoneSink::default());
    let gateway = Arc::new(RecordingGateway::reply("should not be called"));
    let mut mismatched_context = fixture.run_context.clone();
    mismatched_context.resolved_run_profile = InMemoryRunProfileResolver::default()
        .resolve_run_profile(RunProfileResolutionRequest::interactive_default())
        .await
        .unwrap();
    let host = TextOnlyLoopHost::new(
        Arc::clone(&fixture.thread_service),
        fixture.thread_scope.clone(),
        mismatched_context.clone(),
        gateway.clone(),
        milestone_sink.clone(),
        TextOnlyLoopHostConfig::default(),
    );
    let driver = TextOnlyModelReplyDriver::default();

    let error = driver
        .run(driver_request(&mismatched_context), &host)
        .await
        .unwrap_err();

    assert!(matches!(error, AgentLoopDriverError::InvalidRequest { .. }));
    assert!(gateway.requests.lock().unwrap().is_empty());
    assert!(milestone_sink.milestones().is_empty());
    assert_driver_error_hides_sentinels(&error);
}

#[tokio::test]
async fn text_only_model_reply_driver_sanitizes_model_failures_and_skips_transcript_write() {
    let fixture = ThreadFixture::new_with_user_content("RAW_PROMPT_TEXT_SENTINEL").await;
    let milestone_sink = Arc::new(InMemoryLoopHostMilestoneSink::default());
    let gateway = Arc::new(RecordingGateway::deny(
        "RAW_PROVIDER_ERROR invalid api key sk-provider-secret /host/path tool_input",
    ));
    let host = TextOnlyLoopHost::new(
        Arc::clone(&fixture.thread_service),
        fixture.thread_scope.clone(),
        fixture.run_context.clone(),
        gateway,
        milestone_sink.clone(),
        TextOnlyLoopHostConfig::default(),
    );
    let driver = TextOnlyModelReplyDriver::default();

    let error = driver
        .run(driver_request(&fixture.run_context), &host)
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        AgentLoopDriverError::Failed { ref reason_kind } if reason_kind == "model_error"
    ));
    assert_driver_error_hides_sentinels(&error);
    assert_no_assistant_message(&fixture).await;
    let milestones = milestone_sink.milestones();
    assert_eq!(milestones.len(), 2);
    assert_eq!(milestones[0].kind.kind_name(), "prompt_bundle_built");
    assert!(matches!(
        &milestones[1].kind,
        LoopHostMilestoneKind::ModelStarted {
            requested_model_profile_id: None
        }
    ));
    assert_serialized_outputs_hide_sentinels(&milestones);
}

#[tokio::test]
async fn text_only_model_reply_driver_rejects_capability_calls_without_dispatching_tools() {
    let fixture = ThreadFixture::new_with_user_content("hello needs tool").await;
    let milestone_sink = Arc::new(InMemoryLoopHostMilestoneSink::default());
    let gateway = Arc::new(RecordingGateway::capability_calls());
    let host = TextOnlyLoopHost::new(
        Arc::clone(&fixture.thread_service),
        fixture.thread_scope.clone(),
        fixture.run_context.clone(),
        gateway,
        milestone_sink.clone(),
        TextOnlyLoopHostConfig::default(),
    );
    let driver = TextOnlyModelReplyDriver::default();

    let error = driver
        .run(driver_request(&fixture.run_context), &host)
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        AgentLoopDriverError::Failed { ref reason_kind } if reason_kind == "invalid_model_output"
    ));
    assert_no_assistant_message(&fixture).await;
    assert_eq!(
        milestone_sink
            .milestones()
            .iter()
            .map(|milestone| milestone.kind.kind_name())
            .collect::<Vec<_>>(),
        vec!["prompt_bundle_built", "model_started", "model_completed"]
    );
}

fn driver_request(context: &LoopRunContext) -> AgentLoopDriverRunRequest {
    AgentLoopDriverRunRequest {
        turn_id: context.turn_id,
        run_id: context.run_id,
        resolved_run_profile: context.resolved_run_profile.clone(),
    }
}

async fn assert_no_assistant_message(fixture: &ThreadFixture) {
    let history = fixture
        .thread_service
        .list_thread_history(ThreadHistoryRequest {
            scope: fixture.thread_scope.clone(),
            thread_id: fixture.thread_id.clone(),
        })
        .await
        .unwrap();
    assert!(
        !history
            .messages
            .iter()
            .any(|message| message.kind == MessageKind::Assistant)
    );
}

fn assert_serialized_outputs_hide_sentinels<T: serde::Serialize>(value: &T) {
    let wire = serde_json::to_string(value).unwrap();
    for forbidden in [
        "RAW_PROMPT_TEXT_SENTINEL",
        "RAW_ASSISTANT_CONTENT_SENTINEL",
        "RAW_PROVIDER_ERROR",
        "invalid api key",
        "sk-prompt-secret",
        "sk-output-secret",
        "sk-provider-secret",
        "/host/path",
        "tool_input",
    ] {
        assert!(
            !wire.contains(forbidden),
            "serialized value leaked {forbidden}"
        );
    }
}

fn assert_driver_error_hides_sentinels(error: &AgentLoopDriverError) {
    let debug = format!("{error:?}");
    for forbidden in [
        "RAW_PROMPT_TEXT_SENTINEL",
        "RAW_PROVIDER_ERROR",
        "invalid api key",
        "sk-provider-secret",
        "/host/path",
        "tool_input",
    ] {
        assert!(
            !debug.contains(forbidden),
            "driver error leaked {forbidden}"
        );
    }
}

struct ThreadFixture {
    thread_service: Arc<InMemorySessionThreadService>,
    thread_scope: ThreadScope,
    thread_id: ThreadId,
    run_context: LoopRunContext,
}

impl ThreadFixture {
    async fn new_with_user_content(user_content: &str) -> Self {
        let thread_service = Arc::new(InMemorySessionThreadService::default());
        let tenant_id = TenantId::new("tenant-reborn-driver").unwrap();
        let agent_id = AgentId::new("agent-reborn-driver").unwrap();
        let project_id = ProjectId::new("project-reborn-driver").unwrap();
        let user_id = UserId::new("user-reborn-driver").unwrap();
        let thread_id = ThreadId::new("thread-reborn-driver").unwrap();
        let thread_scope = ThreadScope {
            tenant_id: tenant_id.clone(),
            agent_id: agent_id.clone(),
            project_id: Some(project_id.clone()),
            owner_user_id: Some(user_id.clone()),
            mission_id: None,
        };
        thread_service
            .ensure_thread(EnsureThreadRequest {
                scope: thread_scope.clone(),
                thread_id: Some(thread_id.clone()),
                created_by_actor_id: user_id.as_str().to_string(),
                title: None,
                metadata_json: None,
            })
            .await
            .unwrap();
        thread_service
            .accept_inbound_message(AcceptInboundMessageRequest {
                scope: thread_scope.clone(),
                thread_id: thread_id.clone(),
                actor_id: user_id.as_str().to_string(),
                source_binding_id: Some("source-cli".to_string()),
                reply_target_binding_id: Some("reply-cli".to_string()),
                external_event_id: Some("event-driver".to_string()),
                content: MessageContent::text(user_content),
            })
            .await
            .unwrap();
        let turn_scope = TurnScope::new(
            tenant_id,
            Some(agent_id),
            Some(project_id),
            thread_id.clone(),
        );
        let mut resolved = InMemoryRunProfileResolver::default()
            .resolve_run_profile(RunProfileResolutionRequest::interactive_default())
            .await
            .unwrap();
        resolved.loop_driver = TextOnlyModelReplyDriver::default().descriptor();
        let run_context =
            LoopRunContext::new(turn_scope, TurnId::new(), TurnRunId::new(), resolved);
        Self {
            thread_service,
            thread_scope,
            thread_id,
            run_context,
        }
    }
}

struct RecordingGateway {
    requests: Mutex<Vec<HostManagedModelRequest>>,
    response: Result<HostManagedModelResponse, HostManagedModelError>,
}

impl RecordingGateway {
    fn reply(content: &str) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            response: Ok(HostManagedModelResponse::assistant_reply(content)),
        }
    }

    fn deny(raw_detail: &str) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            response: Err(HostManagedModelError::new(
                HostManagedModelErrorKind::PolicyDenied,
                raw_detail,
            )),
        }
    }

    fn capability_calls() -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            response: Ok(HostManagedModelResponse {
                safe_text_deltas: Vec::new(),
                output: ParentLoopOutput::CapabilityCalls(vec![CapabilityCallCandidate {
                    surface_version: CapabilitySurfaceVersion::new("surface-v1").unwrap(),
                    capability_id: CapabilityId::new("demo.echo").unwrap(),
                    input_ref: CapabilityInputRef::new("input:opaque-tool-call").unwrap(),
                }]),
            }),
        }
    }
}

#[async_trait]
impl HostManagedModelGateway for RecordingGateway {
    async fn stream_model(
        &self,
        request: HostManagedModelRequest,
    ) -> Result<HostManagedModelResponse, HostManagedModelError> {
        self.requests.lock().unwrap().push(request);
        self.response.clone()
    }
}
