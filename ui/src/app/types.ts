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
