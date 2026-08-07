import { WorkspaceIde } from './workspace-ide';
import type { CanvasMessage } from './canvas';
import type {
  AutoState,
  ComputerActivityEvent,
  DecisionRespondChoice,
  PlanEnvelope,
  SessionMode,
  TaskExecutionState,
  UiDecision,
} from '../types';
import type { GitDecisionOutcome } from '../use-repository-actions';

type WorkspaceSessionScreenProps = {
  sessionId: string | null;
  sessionHeader: string;
  modelLabel: string;
  workspaceRoot: string;
  sessionPolicyLabel: string;
  sessionYoloActive: boolean;
  sessionSafeMode: boolean;
  messages: CanvasMessage[];
  computerEvents: ComputerActivityEvent[];
  statusMessage: string | null;
  mode: SessionMode;
  activePlan: PlanEnvelope | null;
  activeTask: TaskExecutionState | null;
  autoState: AutoState | null;
  decision: UiDecision | null;
  decisionBusy: boolean;
  decisionError: string | null;
  refreshToken: number;
  gitDecisionOutcome: GitDecisionOutcome | null;
  explorerVisible: boolean;
  onBackToChat: () => void;
  onOpenSessionDrawer: () => void;
  onOpenRepositoryPanel: () => void;
  onApplyWorkspaceRoot: (workspaceRoot: string) => void;
  onDecisionRespond: (
    choice: DecisionRespondChoice,
    editedAction?: Record<string, unknown> | null,
  ) => void;
};

export function WorkspaceSessionScreen({
  sessionId,
  sessionHeader,
  modelLabel,
  workspaceRoot,
  sessionPolicyLabel,
  sessionYoloActive,
  sessionSafeMode,
  messages,
  computerEvents,
  statusMessage,
  mode,
  activePlan,
  activeTask,
  autoState,
  decision,
  decisionBusy,
  decisionError,
  refreshToken,
  gitDecisionOutcome,
  explorerVisible,
  onBackToChat,
  onOpenSessionDrawer,
  onOpenRepositoryPanel,
  onApplyWorkspaceRoot,
  onDecisionRespond,
}: WorkspaceSessionScreenProps) {
  return (
    <WorkspaceIde
      sessionId={sessionId}
      sessionHeader={sessionHeader}
      modelLabel={modelLabel}
      workspaceRoot={workspaceRoot}
      sessionPolicyLabel={sessionPolicyLabel}
      sessionYoloActive={sessionYoloActive}
      sessionSafeMode={sessionSafeMode}
      messages={messages}
      computerEvents={computerEvents}
      statusMessage={statusMessage}
      onBackToChat={onBackToChat}
      onOpenSessionDrawer={onOpenSessionDrawer}
      onOpenRepositoryPanel={onOpenRepositoryPanel}
      onApplyWorkspaceRoot={onApplyWorkspaceRoot}
      mode={mode}
      activePlan={activePlan}
      activeTask={activeTask}
      autoState={autoState}
      decision={decision}
      decisionBusy={decisionBusy}
      decisionError={decisionError}
      onDecisionRespond={onDecisionRespond}
      refreshToken={refreshToken}
      gitDecisionOutcome={gitDecisionOutcome}
      explorerVisible={explorerVisible}
    />
  );
}
