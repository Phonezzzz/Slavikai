export type Message = {
  role: "user" | "assistant" | "system";
  content: string;
};

export type DecisionOptionView = {
  id: string;
  title: string;
  action: string;
  risk: string;
};

export type DecisionPacketView = {
  id: string;
  reason: string;
  summary: string;
  options: DecisionOptionView[];
  default_option_id?: string | null;
};

export type SelectedModel = {
  provider: string;
  model: string;
};

export type ProviderModels = {
  provider: string;
  models: string[];
  error?: string | null;
};

export type UIEvent = {
  id: string;
  type: string;
  ts: string;
  payload: unknown;
};

export type SessionSummary = {
  session_id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

export type ApprovalPromptView = {
  what: string;
  why: string;
  risk: string;
  changes: string[];
};

export type ApprovalRequestView = {
  category: string;
  required_categories: string[];
  tool: string;
  details: Record<string, unknown>;
  session_id: string | null;
  prompt: ApprovalPromptView;
};
