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
