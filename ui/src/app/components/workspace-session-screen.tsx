import { WorkspaceIde } from './workspace-ide';
import type { CanvasMessage, CanvasSendPayload } from './canvas';
import type {
  AutoState,
  DecisionRespondChoice,
  ModeTransitionsContract,
  PlanEnvelope,
  SessionMode,
  TaskExecutionState,
  UiDecision,
} from '../types';

type WorkspaceSessionScreenProps = {
  sessionId: string | null;
  sessionHeader: string;
  modelLabel: string;
  sessionPolicyLabel: string;
  sessionYoloActive: boolean;
  sessionSafeMode: boolean;
  messages: CanvasMessage[];
  sending: boolean;
  statusMessage: string | null;
  mode: SessionMode;
  activePlan: PlanEnvelope | null;
  activeTask: TaskExecutionState | null;
  autoState: AutoState | null;
  modeTransitions: ModeTransitionsContract | null;
  modeBusy: boolean;
  modeError: string | null;
  decision: UiDecision | null;
  decisionBusy: boolean;
  decisionError: string | null;
  refreshToken: number;
  explorerVisible: boolean;
  onBackToChat: () => void;
  onOpenSessionDrawer: () => void;
  onOpenRepositoryPanel: () => void;
  onSendAgentMessage: (payload: CanvasSendPayload) => Promise<boolean>;
  onChangeMode: (mode: SessionMode) => Promise<void>;
  onPlanDraft: (goal: string) => Promise<void>;
  onPlanApprove: () => Promise<void>;
  onPlanExecute: () => Promise<void>;
  onPlanCancel: () => Promise<void>;
  onDecisionRespond: (
    choice: DecisionRespondChoice,
    editedAction?: Record<string, unknown> | null,
  ) => void;
};

export function WorkspaceSessionScreen({
  sessionId,
  sessionHeader,
  modelLabel,
  sessionPolicyLabel,
  sessionYoloActive,
  sessionSafeMode,
  messages,
  sending,
  statusMessage,
  mode,
  activePlan,
  activeTask,
  autoState,
  modeTransitions,
  modeBusy,
  modeError,
  decision,
  decisionBusy,
  decisionError,
  refreshToken,
  explorerVisible,
  onBackToChat,
  onOpenSessionDrawer,
  onOpenRepositoryPanel,
  onSendAgentMessage,
  onChangeMode,
  onPlanDraft,
  onPlanApprove,
  onPlanExecute,
  onPlanCancel,
  onDecisionRespond,
}: WorkspaceSessionScreenProps) {
  return (
    <WorkspaceIde
      sessionId={sessionId}
      sessionHeader={sessionHeader}
      modelLabel={modelLabel}
      sessionPolicyLabel={sessionPolicyLabel}
      sessionYoloActive={sessionYoloActive}
      sessionSafeMode={sessionSafeMode}
      messages={messages}
      sending={sending}
      statusMessage={statusMessage}
      onBackToChat={onBackToChat}
      onOpenSessionDrawer={onOpenSessionDrawer}
      onOpenRepositoryPanel={onOpenRepositoryPanel}
      onSendAgentMessage={onSendAgentMessage}
      mode={mode}
      activePlan={activePlan}
      activeTask={activeTask}
      autoState={autoState}
      modeTransitions={modeTransitions}
      modeBusy={modeBusy}
      modeError={modeError}
      onChangeMode={onChangeMode}
      onPlanDraft={onPlanDraft}
      onPlanApprove={onPlanApprove}
      onPlanExecute={onPlanExecute}
      onPlanCancel={onPlanCancel}
      decision={decision}
      decisionBusy={decisionBusy}
      decisionError={decisionError}
      onDecisionRespond={onDecisionRespond}
      refreshToken={refreshToken}
      explorerVisible={explorerVisible}
    />
  );
}
