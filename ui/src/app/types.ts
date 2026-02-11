export type ChatRole = 'user' | 'assistant' | 'system';

export type ChatAttachment = {
  name: string;
  mime: string;
  content: string;
};

export type ChatMessage = {
  message_id: string;
  role: ChatRole;
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

export type UiDecision = {
  id: string;
  kind: 'approval' | 'decision';
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

export type SessionSummary = {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
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
