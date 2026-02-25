import type {
  DecisionRespondChoice,
  MessageRuntimeMeta,
  MwvReportUi,
  UiDecision,
} from '../../app/types';

export type MessageRenderContext = 'chat' | 'workspace';

export type RenderableMessageRole = 'user' | 'assistant';

export type RenderableAttachment = {
  name: string;
  mime: string;
  content: string;
};

export type RenderableUiMessage = {
  id: string;
  messageId: string;
  role: RenderableMessageRole;
  content: string;
  createdAt?: string;
  traceId?: string | null;
  parentUserMessageId?: string | null;
  attachments?: RenderableAttachment[];
  transient?: boolean;
  runtimeMeta?: MessageRuntimeMeta | null;
};

export type RenderableMessage =
  | {
      kind: 'message';
      message: RenderableUiMessage;
      meta: MessageRuntimeMeta | null;
    }
  | {
      kind: 'decision';
      id: string;
      decision: UiDecision;
    };

export type TextMessageBlock = {
  kind: 'text';
  id: string;
  markdown: string;
};

export type CodeMessageBlock = {
  kind: 'code';
  id: string;
  language: string | null;
  code: string;
  isFinal: boolean;
};

export type ToolMessageBlock = {
  kind: 'tool';
  id: string;
  traceId: string;
  summary: string;
  report: MwvReportUi | null;
};

export type VerifierMessageBlock = {
  kind: 'verifier';
  id: string;
  summary: string;
  verifier: Record<string, unknown> | null;
  traceId: string | null;
};

export type DecisionMessageBlock = {
  kind: 'decision';
  id: string;
  decision: UiDecision;
};

export type MessageBlock =
  | TextMessageBlock
  | CodeMessageBlock
  | ToolMessageBlock
  | VerifierMessageBlock
  | DecisionMessageBlock;

export type MessageRendererProps = {
  context: MessageRenderContext;
  message: RenderableMessage;
  decisionBusy?: boolean;
  decisionError?: string | null;
  onDecisionRespond?: (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
  ) => Promise<void> | void;
};
