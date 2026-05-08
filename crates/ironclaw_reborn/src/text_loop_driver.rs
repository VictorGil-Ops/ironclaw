//! Text-only Reborn loop driver.
//!
//! This driver is deliberately narrow: it asks the host to build a text prompt
//! bundle, streams one host-managed model response, accepts only a final assistant
//! reply, and finalizes that reply through the transcript port. Tool/capability
//! calls are rejected until the tool-capable loop slice exists.

use async_trait::async_trait;
use ironclaw_turns::{
    LoopCompleted, LoopCompletionKind, LoopExit, LoopExitId, LoopFailureKind, LoopMessageRef,
    RunProfileVersion,
    run_profile::{
        AgentLoopDriver, AgentLoopDriverDescriptor, AgentLoopDriverError, AgentLoopDriverHost,
        AgentLoopDriverResumeRequest, AgentLoopDriverRunRequest, AgentLoopHostError,
        AgentLoopHostErrorKind, FinalizeAssistantMessage, LoopModelRequest,
        LoopPromptBundleRequest, ParentLoopOutput, PromptMode,
    },
};

const TEXT_ONLY_DRIVER_ID: &str = "reborn:text-only-model-reply";
const TEXT_ONLY_DRIVER_VERSION: u64 = 1;
const DEFAULT_CONTEXT_LIMIT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextOnlyModelReplyDriverConfig {
    pub context_limit: usize,
}

impl Default for TextOnlyModelReplyDriverConfig {
    fn default() -> Self {
        Self {
            context_limit: DEFAULT_CONTEXT_LIMIT,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TextOnlyModelReplyDriver {
    config: TextOnlyModelReplyDriverConfig,
}

impl TextOnlyModelReplyDriver {
    pub fn new(config: TextOnlyModelReplyDriverConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl AgentLoopDriver for TextOnlyModelReplyDriver {
    fn descriptor(&self) -> AgentLoopDriverDescriptor {
        AgentLoopDriverDescriptor::from_trusted_static(
            TEXT_ONLY_DRIVER_ID,
            RunProfileVersion::new(TEXT_ONLY_DRIVER_VERSION),
        )
    }

    async fn run(
        &self,
        request: AgentLoopDriverRunRequest,
        host: &(dyn AgentLoopDriverHost + Send + Sync),
    ) -> Result<LoopExit, AgentLoopDriverError> {
        validate_run_request(&request, host, &self.descriptor())?;
        let prompt_bundle = host
            .build_prompt_bundle(LoopPromptBundleRequest {
                mode: PromptMode::TextOnly,
                context_cursor: None,
                surface_version: None,
                checkpoint_state_ref: None,
                max_messages: Some(self.config.context_limit as u32),
            })
            .await
            .map_err(|error| map_host_error("prompt", error))?;
        let model_response = host
            .stream_model(LoopModelRequest {
                messages: prompt_bundle.messages,
                surface_version: prompt_bundle.surface_version,
                model_preference: None,
            })
            .await
            .map_err(|error| map_host_error("model", error))?;
        let reply = match model_response.output {
            ParentLoopOutput::AssistantReply(reply) => reply,
            ParentLoopOutput::CapabilityCalls(_) => {
                return Err(AgentLoopDriverError::Failed {
                    reason_kind: loop_failure_kind_name(LoopFailureKind::InvalidModelOutput)
                        .to_string(),
                });
            }
        };
        let reply_ref = host
            .finalize_assistant_message(FinalizeAssistantMessage { reply })
            .await
            .map_err(|error| map_host_error("transcript", error))?;
        Ok(LoopExit::Completed(completed_final_reply(
            request.run_id,
            reply_ref,
        )?))
    }

    async fn resume(
        &self,
        _request: AgentLoopDriverResumeRequest,
        _host: &(dyn AgentLoopDriverHost + Send + Sync),
    ) -> Result<LoopExit, AgentLoopDriverError> {
        Err(AgentLoopDriverError::InvalidRequest {
            reason: "text-only model reply driver does not support resume".to_string(),
        })
    }
}

fn validate_run_request(
    request: &AgentLoopDriverRunRequest,
    host: &(dyn AgentLoopDriverHost + Send + Sync),
    descriptor: &AgentLoopDriverDescriptor,
) -> Result<(), AgentLoopDriverError> {
    let context = host.run_context();
    if request.turn_id != context.turn_id || request.run_id != context.run_id {
        return Err(AgentLoopDriverError::InvalidRequest {
            reason: "driver request does not match loop host run context".to_string(),
        });
    }
    if request.resolved_run_profile != context.resolved_run_profile {
        return Err(AgentLoopDriverError::InvalidRequest {
            reason: "driver request profile does not match loop host run context".to_string(),
        });
    }
    if request.resolved_run_profile.loop_driver != *descriptor {
        return Err(AgentLoopDriverError::InvalidRequest {
            reason: "driver request profile is not assigned to the text-only model reply driver"
                .to_string(),
        });
    }
    Ok(())
}

fn completed_final_reply(
    run_id: ironclaw_turns::TurnRunId,
    reply_ref: LoopMessageRef,
) -> Result<LoopCompleted, AgentLoopDriverError> {
    let exit_id = LoopExitId::new(format!("exit:{run_id}-final-reply")).map_err(|_| {
        AgentLoopDriverError::Failed {
            reason_kind: loop_failure_kind_name(LoopFailureKind::DriverBug).to_string(),
        }
    })?;
    Ok(LoopCompleted {
        completion_kind: LoopCompletionKind::FinalReply,
        reply_message_refs: vec![reply_ref],
        result_refs: Vec::new(),
        final_checkpoint_id: None,
        usage_summary_ref: None,
        exit_id,
    })
}

fn map_host_error(stage: &'static str, error: AgentLoopHostError) -> AgentLoopDriverError {
    match error.kind {
        AgentLoopHostErrorKind::InvalidInvocation | AgentLoopHostErrorKind::ScopeMismatch => {
            AgentLoopDriverError::InvalidRequest {
                reason: format!("{stage}: {}", error.kind.as_str()),
            }
        }
        AgentLoopHostErrorKind::Unavailable | AgentLoopHostErrorKind::Cancelled => {
            AgentLoopDriverError::Unavailable {
                reason: format!("{stage}: {}", error.kind.as_str()),
            }
        }
        AgentLoopHostErrorKind::TranscriptWriteFailed => AgentLoopDriverError::Failed {
            reason_kind: loop_failure_kind_name(LoopFailureKind::TranscriptWriteFailed).to_string(),
        },
        AgentLoopHostErrorKind::BudgetExceeded | AgentLoopHostErrorKind::PolicyDenied => {
            AgentLoopDriverError::Failed {
                reason_kind: loop_failure_kind_name(LoopFailureKind::ModelError).to_string(),
            }
        }
        AgentLoopHostErrorKind::CheckpointRejected => AgentLoopDriverError::Failed {
            reason_kind: loop_failure_kind_name(LoopFailureKind::CheckpointRejected).to_string(),
        },
        AgentLoopHostErrorKind::Unauthorized
        | AgentLoopHostErrorKind::StaleSurface
        | AgentLoopHostErrorKind::Internal => AgentLoopDriverError::Failed {
            reason_kind: loop_failure_kind_name(LoopFailureKind::DriverBug).to_string(),
        },
    }
}

fn loop_failure_kind_name(kind: LoopFailureKind) -> &'static str {
    match kind {
        LoopFailureKind::ModelError => "model_error",
        LoopFailureKind::ContextBuildFailed => "context_build_failed",
        LoopFailureKind::CapabilityProtocolError => "capability_protocol_error",
        LoopFailureKind::IterationLimit => "iteration_limit",
        LoopFailureKind::InvalidModelOutput => "invalid_model_output",
        LoopFailureKind::CheckpointRejected => "checkpoint_rejected",
        LoopFailureKind::TranscriptWriteFailed => "transcript_write_failed",
        LoopFailureKind::DriverBug => "driver_bug",
        LoopFailureKind::InterruptedUnexpectedly => "interrupted_unexpectedly",
    }
}
