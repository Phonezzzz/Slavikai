export type ChatRole = 'user' | 'assistant' | 'system';
export type MessageLane = 'chat' | 'workspace';

export type ChatAttachment = {
  name: string;
  mime: string;
  content: string;
};

export type ChatMessage = {
  message_id: string;
  role: ChatRole;
  lane: MessageLane;
  content: string;
  created_at: string;
  trace_id: string | null;
  parent_user_message_id: string | null;
  attachments: ChatAttachment[];
};

export type UiDecisionStatus = 'pending' | 'approved' | 'rejected' | 'executing' | 'resolved';

export type UiDecisionOption = {
  id: string;
  title: string;
  action: string;
  payload: Record<string, unknown>;
  risk: string;
};

export type DecisionRespondChoice =
  | 'approve_once'
  | 'approve_session'
  | 'edit_and_approve'
  | 'edit_plan'
  | 'reject'
  | 'ask_user'
  | 'proceed_safe'
  | 'retry'
  | 'abort'
  | 'select_skill'
  | 'adjust_threshold'
  | 'create_candidate';

export type UiDecision = {
  id: string;
  kind: 'approval' | 'decision';
  decision_type: 'tool_approval' | 'plan_execute' | 'agent_decision' | null;
  status: UiDecisionStatus;
  blocking: boolean;
  reason: string;
  summary: string;
  proposed_action: Record<string, unknown>;
  options: UiDecisionOption[];
  default_option_id: string | null;
  context: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  resolved_at: string | null;
};

export const SESSION_MODE_VALUES = ['ask', 'plan', 'act', 'auto'] as const;

export type SessionMode = (typeof SESSION_MODE_VALUES)[number];

export const isSessionMode = (value: unknown): value is SessionMode =>
  typeof value === 'string' && SESSION_MODE_VALUES.some((mode) => mode === value);

export type AutoRunStatus =
  | 'idle'
  | 'planning'
  | 'coding'
  | 'merging'
  | 'verifying'
  | 'waiting_approval'
  | 'completed'
  | 'failed_conflict'
  | 'failed_verifier'
  | 'failed_worker'
  | 'failed_internal'
  | 'cancelled';

export type AutoState = {
  run_id: string;
  status: AutoRunStatus;
  goal: string;
  pool_size: number;
  started_at: string;
  updated_at: string;
  planner: Record<string, unknown>;
  plan: Record<string, unknown> | null;
  coders: Array<Record<string, unknown>>;
  merge: Record<string, unknown>;
  verifier: Record<string, unknown> | null;
  approval: Record<string, unknown> | null;
  error: string | null;
};

export type PlanStepStatus = 'todo' | 'doing' | 'waiting_approval' | 'blocked' | 'done' | 'failed';

export type PlanStatus = 'draft' | 'approved' | 'running' | 'completed' | 'failed' | 'cancelled';

export type TaskStatus = 'running' | 'completed' | 'failed' | 'cancelled';

export type PlanStep = {
  step_id: string;
  title: string;
  description: string;
  allowed_tool_kinds: string[];
  acceptance_checks: string[];
  status: PlanStepStatus;
  details: string | null;
};

export type PlanEnvelope = {
  plan_id: string;
  plan_hash: string;
  plan_revision: number;
  status: PlanStatus;
  goal: string;
  scope_in: string[];
  scope_out: string[];
  assumptions: string[];
  inputs_needed: string[];
  audit_log: unknown[];
  steps: PlanStep[];
  exit_criteria: string[];
  created_at: string;
  updated_at: string;
  approved_at: string | null;
  approved_by: string | null;
};

export type TaskExecutionState = {
  task_id: string;
  plan_id: string;
  plan_hash: string;
  current_step_id: string | null;
  status: TaskStatus;
  started_at: string;
  updated_at: string;
};

export type SessionSummary = {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  chat_message_count: number;
  workspace_message_count: number;
  last_message_lane: MessageLane | null;
  title_override?: string | null;
  folder_id?: string | null;
};

export type FolderSummary = {
  folder_id: string;
  name: string;
  created_at: string;
  updated_at: string;
};

export type SelectedModel = {
  provider: string;
  model: string;
};

export type ProviderModels = {
  provider: string;
  models: string[];
  error: string | null;
};

export type UploadPreviewType = 'text' | 'image' | 'binary';

export type UploadHistoryItem = {
  id: string;
  name: string;
  size: number;
  type: string;
  preview: string;
  previewType: UploadPreviewType;
  previewUrl: string | null;
  createdAt: string;
};
