import type {
  ChatMessage,
  MessageLane,
  PlanEnvelope,
  SelectedModel,
  SessionMode,
  TaskExecutionState,
  UiDecision,
  AutoState,
} from './types';
import type { SessionArtifactRecord } from './session-payload';

export type RuntimePayloadSnapshot = {
  decisionProvided?: boolean;
  decision: UiDecision | null;
  selectedModel: SelectedModel | null;
  modeProvided?: boolean;
  mode: SessionMode;
  activePlanProvided?: boolean;
  activePlan: PlanEnvelope | null;
  activeTaskProvided?: boolean;
  activeTask: TaskExecutionState | null;
  autoStateProvided?: boolean;
  autoState: AutoState | null;
};

export type SessionTransportBridge = {
  applyLoadedConversation: (snapshot: {
    chatMessages: ChatMessage[];
    workspaceMessages: ChatMessage[];
    outputContent: string | null;
    files: string[];
    artifacts: SessionArtifactRecord[];
  }) => void;
  applySessionPayload: (
    payload: unknown,
    options: { applyDisplay: boolean },
  ) => { lane: MessageLane };
  clearConversationState: () => void;
};
