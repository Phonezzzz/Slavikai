export type ChatRole = 'user' | 'assistant' | 'system';

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type SessionSummary = {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
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
