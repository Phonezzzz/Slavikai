import {
  Activity,
  Bot,
  BrainCircuit,
  Database,
  Download,
  KeyRound,
  Palette,
  RefreshCcw,
  TextCursorInput,
  Upload,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import type { ChangeEvent, ReactNode } from 'react';
import { useEffect, useRef, useState } from 'react';

import { StatusBadge, type StatusTone } from './ui/status-badge';
import { ToggleSwitch } from './ui/toggle-switch';
import { useFocusTrap } from '../use-focus-trap';

interface SettingsProps {
  isOpen: boolean;
  onClose: () => void;
  onSaved?: () => void;
}

type SettingsTab = 'api' | 'assistant' | 'appearance' | 'composer' | 'memory' | 'data' | 'diagnostics';
type ApiKeyProvider = 'xai' | 'openrouter' | 'local' | 'inception' | 'openai' | 'deepseek';
type ModelProvider = 'xai' | 'openrouter' | 'local' | 'inception' | 'deepseek';
type ApiKeySource = 'env' | 'file' | 'missing';
type EmbeddingsProvider = 'local' | 'openai';
type EmbeddingDownloadState = 'missing' | 'package_missing' | 'downloading' | 'ready' | 'error';
type ImportMode = 'replace' | 'merge';
type AppearanceTheme = 'default' | 'oled';

type ProviderSettings = {
  provider: ApiKeyProvider;
  api_key_env: string;
  api_key_set: boolean;
  api_key_stored: boolean;
  api_key_source: ApiKeySource;
  endpoint: string;
  api_key_valid: boolean | null;
  last_check_error: string | null;
  last_checked_at: string | null;
};

type ProviderRuntimeState = {
  modelsCount: number;
  error: string | null;
};

type TtsBackendSettings = {
  provider: 'openai';
  api_key_env: string;
  api_key_set: boolean;
  endpoint: string;
  model: string;
  voice: string;
  format: string;
  backend_ready: boolean;
};

type ParsedSettings = {
  providers: ProviderSettings[];
  ttsBackend: TtsBackendSettings;
  tone: string;
  systemPrompt: string;
  appearanceTheme: AppearanceTheme;
  longPasteToFileEnabled: boolean;
  longPasteThresholdChars: number;
  memoryInboxMaxItems: number;
  memoryInboxTtlDays: number;
  memoryInboxWritesPerMinute: number;
  embeddingsProvider: EmbeddingsProvider;
  embeddingsLocalModel: string;
  embeddingsOpenaiModel: string;
  defaultModelProvider: ModelProvider;
  defaultModelId: string;
};

type EmbeddingRuntime = {
  model: string;
  state: EmbeddingDownloadState;
  error: string | null;
};

type ProviderRuntimeByModel = Record<ModelProvider, ProviderRuntimeState | null>;

type ImportPreview = {
  fileName: string;
  sessionsCount: number;
  messagesCount: number;
};

const DEFAULT_SYSTEM_PROMPT =
  'You are SlavikAI, a helpful AI assistant with MWV architecture.';

const API_KEY_PROVIDERS: ApiKeyProvider[] = ['xai', 'openrouter', 'local', 'inception', 'openai', 'deepseek'];

const EMPTY_API_KEYS: Record<ApiKeyProvider, string> = {
  xai: '',
  openrouter: '',
  local: '',
  inception: '',
  openai: '',
  deepseek: '',
};

const EMPTY_PROVIDER_DIRTY: Record<ApiKeyProvider, boolean> = {
  xai: false,
  openrouter: false,
  local: false,
  inception: false,
  openai: false,
  deepseek: false,
};

const PROVIDER_LABELS: Record<ApiKeyProvider, string> = {
  xai: 'xAI',
  openrouter: 'OpenRouter',
  local: 'Local',
  inception: 'Inception',
  openai: 'OpenAI',
  deepseek: 'DeepSeek',
};

const RESPONSE_STYLE_OPTIONS = [
  {
    value: 'balanced',
    label: 'Balanced',
    description: 'Default mix of concise answers and useful context.',
  },
  {
    value: 'professional',
    label: 'Professional',
    description: 'More formal tone with structured and direct wording.',
  },
  {
    value: 'technical',
    label: 'Technical',
    description: 'Prioritizes precision, implementation details, and engineering vocabulary.',
  },
  {
    value: 'friendly',
    label: 'Friendly',
    description: 'Warmer wording while keeping the same assistant behavior.',
  },
  {
    value: 'casual',
    label: 'Casual',
    description: 'More conversational phrasing for low-friction everyday use.',
  },
];

const TAB_ITEMS: Array<{ id: SettingsTab; title: string }> = [
  { id: 'api', title: 'API Keys' },
  { id: 'assistant', title: 'Assistant' },
  { id: 'appearance', title: 'Appearance' },
  { id: 'composer', title: 'Composer' },
  { id: 'memory', title: 'Memory' },
  { id: 'data', title: 'Data' },
  { id: 'diagnostics', title: 'Diagnostics' },
];

const TAB_ICONS: Record<SettingsTab, typeof KeyRound> = {
  api: KeyRound,
  assistant: Bot,
  appearance: Palette,
  composer: TextCursorInput,
  memory: BrainCircuit,
  data: Database,
  diagnostics: Activity,
};

const THRESHOLD_PRESETS = [8000, 12000, 25000];

const DEFAULT_LONG_PASTE_TO_FILE_ENABLED = true;
const DEFAULT_LONG_PASTE_THRESHOLD_CHARS = 12000;
const DEFAULT_APPEARANCE_THEME: AppearanceTheme = 'default';
const APPEARANCE_THEME_OPTIONS: Array<{
  value: AppearanceTheme;
  label: string;
  description: string;
}> = [
  {
    value: 'default',
    label: 'Default',
    description: 'Current dark UI with the existing panel tones.',
  },
  {
    value: 'oled',
    label: 'OLED',
    description: 'Pure black app and chat background for OLED screens.',
  },
];
const DEFAULT_EMBEDDINGS_PROVIDER: EmbeddingsProvider = 'local';
const DEFAULT_EMBEDDINGS_LOCAL_MODEL = 'all-MiniLM-L6-v2';
const DEFAULT_EMBEDDINGS_OPENAI_MODEL = 'text-embedding-3-small';
const DEFAULT_MEMORY_INBOX_MAX_ITEMS = 200;
const DEFAULT_MEMORY_INBOX_TTL_DAYS = 30;
const DEFAULT_MEMORY_INBOX_WRITES_PER_MINUTE = 6;

const DEFAULT_PROVIDER_SETTINGS: ProviderSettings[] = [
  {
    provider: 'xai',
    api_key_env: 'XAI_API_KEY',
    api_key_set: false,
    api_key_stored: false,
    api_key_source: 'missing',
    endpoint: 'https://api.x.ai/v1/models',
    api_key_valid: null,
    last_check_error: null,
    last_checked_at: null,
  },
  {
    provider: 'openrouter',
    api_key_env: 'OPENROUTER_API_KEY',
    api_key_set: false,
    api_key_stored: false,
    api_key_source: 'missing',
    endpoint: 'https://openrouter.ai/api/v1/models',
    api_key_valid: null,
    last_check_error: null,
    last_checked_at: null,
  },
  {
    provider: 'local',
    api_key_env: 'LOCAL_LLM_API_KEY',
    api_key_set: false,
    api_key_stored: false,
    api_key_source: 'missing',
    endpoint: 'http://localhost:11434/v1/models',
    api_key_valid: null,
    last_check_error: null,
    last_checked_at: null,
  },
  {
    provider: 'openai',
    api_key_env: 'OPENAI_API_KEY',
    api_key_set: false,
    api_key_stored: false,
    api_key_source: 'missing',
    endpoint: 'https://api.openai.com/v1/audio/transcriptions',
    api_key_valid: null,
    last_check_error: null,
    last_checked_at: null,
  },
  {
    provider: 'deepseek',
    api_key_env: 'DEEPSEEK_API_KEY',
    api_key_set: false,
    api_key_stored: false,
    api_key_source: 'missing',
    endpoint: 'https://api.deepseek.com/models',
    api_key_valid: null,
    last_check_error: null,
    last_checked_at: null,
  },
  {
    provider: 'inception',
    api_key_env: 'INCEPTION_API_KEY',
    api_key_set: false,
    api_key_stored: false,
    api_key_source: 'missing',
    endpoint: 'https://api.inceptionlabs.ai/v1/models',
    api_key_valid: null,
    last_check_error: null,
    last_checked_at: null,
  },
];

const DEFAULT_TTS_BACKEND: TtsBackendSettings = {
  provider: 'openai',
  api_key_env: 'OPENAI_API_KEY',
  api_key_set: false,
  endpoint: 'https://api.openai.com/v1/audio/speech',
  model: 'gpt-4o-mini-tts',
  voice: 'alloy',
  format: 'mp3',
  backend_ready: false,
};

const DEFAULT_PROVIDER_RUNTIME: ProviderRuntimeByModel = {
  xai: null,
  openrouter: null,
  local: null,
  inception: null,
  deepseek: null,
};

const isApiKeyProvider = (value: unknown): value is ApiKeyProvider =>
  value === 'xai' || value === 'openrouter' || value === 'local' || value === 'inception' || value === 'openai' || value === 'deepseek';

const isApiKeySource = (value: unknown): value is ApiKeySource =>
  value === 'env' || value === 'file' || value === 'missing';

const isModelProvider = (value: unknown): value is ModelProvider =>
  value === 'xai' || value === 'openrouter' || value === 'local' || value === 'inception' || value === 'deepseek';

const isEmbeddingsProvider = (value: unknown): value is EmbeddingsProvider =>
  value === 'local' || value === 'openai';

const extractErrorMessage = (payload: unknown, fallback: string): string => {
  if (!payload || typeof payload !== 'object') {
    return fallback;
  }
  const body = payload as { error?: { message?: unknown } };
  if (body.error && typeof body.error.message === 'string' && body.error.message.trim()) {
    return body.error.message;
  }
  return fallback;
};

const parseSettingsPayload = (payload: unknown): ParsedSettings => {
  const defaults: ParsedSettings = {
    providers: DEFAULT_PROVIDER_SETTINGS,
    ttsBackend: DEFAULT_TTS_BACKEND,
    tone: 'balanced',
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
    appearanceTheme: DEFAULT_APPEARANCE_THEME,
    longPasteToFileEnabled: DEFAULT_LONG_PASTE_TO_FILE_ENABLED,
    longPasteThresholdChars: DEFAULT_LONG_PASTE_THRESHOLD_CHARS,
    memoryInboxMaxItems: DEFAULT_MEMORY_INBOX_MAX_ITEMS,
    memoryInboxTtlDays: DEFAULT_MEMORY_INBOX_TTL_DAYS,
    memoryInboxWritesPerMinute: DEFAULT_MEMORY_INBOX_WRITES_PER_MINUTE,
    embeddingsProvider: DEFAULT_EMBEDDINGS_PROVIDER,
    embeddingsLocalModel: DEFAULT_EMBEDDINGS_LOCAL_MODEL,
    embeddingsOpenaiModel: DEFAULT_EMBEDDINGS_OPENAI_MODEL,
    defaultModelProvider: 'deepseek',
    defaultModelId: '',
  };

  if (!payload || typeof payload !== 'object') {
    return defaults;
  }
  const settings = (payload as { settings?: unknown }).settings;
  if (!settings || typeof settings !== 'object') {
    return defaults;
  }

  let appearanceTheme = defaults.appearanceTheme;
  let defaultModelProvider = defaults.defaultModelProvider;
  let defaultModelId = defaults.defaultModelId;
  const defaultModelRaw = (settings as { model?: unknown }).model;
  if (defaultModelRaw && typeof defaultModelRaw === 'object') {
    const providerRaw = (defaultModelRaw as { provider?: unknown }).provider;
    const modelRaw = (defaultModelRaw as { model?: unknown }).model;
    if (isModelProvider(providerRaw)) {
      defaultModelProvider = providerRaw;
    }
    if (typeof modelRaw === 'string' && modelRaw.trim()) {
      defaultModelId = modelRaw.trim();
    }
  }
  const appearance = (settings as { appearance?: unknown }).appearance;
  if (appearance && typeof appearance === 'object') {
    const themeRaw = (appearance as { theme?: unknown }).theme;
    if (themeRaw === 'oled' || themeRaw === 'default') {
      appearanceTheme = themeRaw;
    }
  }

  let tone = defaults.tone;
  let systemPrompt = defaults.systemPrompt;
  const personalization = (settings as { personalization?: unknown }).personalization;
  if (personalization && typeof personalization === 'object') {
    const toneRaw = (personalization as { tone?: unknown }).tone;
    const promptRaw = (personalization as { system_prompt?: unknown }).system_prompt;
    if (typeof toneRaw === 'string' && toneRaw.trim()) {
      tone = toneRaw.trim();
    }
    if (typeof promptRaw === 'string') {
      systemPrompt = promptRaw;
    }
  }

  let longPasteToFileEnabled = defaults.longPasteToFileEnabled;
  let longPasteThresholdChars = defaults.longPasteThresholdChars;
  const composer = (settings as { composer?: unknown }).composer;
  if (composer && typeof composer === 'object') {
    const enabledRaw = (composer as { long_paste_to_file_enabled?: unknown }).long_paste_to_file_enabled;
    const thresholdRaw = (composer as { long_paste_threshold_chars?: unknown }).long_paste_threshold_chars;
    if (typeof enabledRaw === 'boolean') {
      longPasteToFileEnabled = enabledRaw;
    }
    if (typeof thresholdRaw === 'number' && Number.isFinite(thresholdRaw) && thresholdRaw > 0) {
      longPasteThresholdChars = Math.floor(thresholdRaw);
    }
  }

  let memoryInboxMaxItems = defaults.memoryInboxMaxItems;
  let memoryInboxTtlDays = defaults.memoryInboxTtlDays;
  let memoryInboxWritesPerMinute = defaults.memoryInboxWritesPerMinute;
  let embeddingsProvider = defaults.embeddingsProvider;
  let embeddingsLocalModel = defaults.embeddingsLocalModel;
  let embeddingsOpenaiModel = defaults.embeddingsOpenaiModel;
  const memory = (settings as { memory?: unknown }).memory;
  if (memory && typeof memory === 'object') {
    const inboxMaxItemsRaw = (memory as { inbox_max_items?: unknown }).inbox_max_items;
    const inboxTtlDaysRaw = (memory as { inbox_ttl_days?: unknown }).inbox_ttl_days;
    const inboxWritesRaw = (memory as { inbox_writes_per_minute?: unknown }).inbox_writes_per_minute;
    if (typeof inboxMaxItemsRaw === 'number' && Number.isFinite(inboxMaxItemsRaw) && inboxMaxItemsRaw > 0) {
      memoryInboxMaxItems = Math.floor(inboxMaxItemsRaw);
    }
    if (typeof inboxTtlDaysRaw === 'number' && Number.isFinite(inboxTtlDaysRaw) && inboxTtlDaysRaw > 0) {
      memoryInboxTtlDays = Math.floor(inboxTtlDaysRaw);
    }
    if (typeof inboxWritesRaw === 'number' && Number.isFinite(inboxWritesRaw) && inboxWritesRaw > 0) {
      memoryInboxWritesPerMinute = Math.floor(inboxWritesRaw);
    }

    const embeddingsRaw = (memory as { embeddings?: unknown }).embeddings;
    if (embeddingsRaw && typeof embeddingsRaw === 'object') {
      const providerRaw = (embeddingsRaw as { provider?: unknown }).provider;
      const localModelRaw = (embeddingsRaw as { local_model?: unknown }).local_model;
      const openaiModelRaw = (embeddingsRaw as { openai_model?: unknown }).openai_model;
      if (isEmbeddingsProvider(providerRaw)) {
        embeddingsProvider = providerRaw;
      }
      if (typeof localModelRaw === 'string' && localModelRaw.trim()) {
        embeddingsLocalModel = localModelRaw.trim();
      }
      if (typeof openaiModelRaw === 'string' && openaiModelRaw.trim()) {
        embeddingsOpenaiModel = openaiModelRaw.trim();
      }
    }
  }

  const providersRaw = (settings as { providers?: unknown }).providers;
  const providersMap = new Map<ApiKeyProvider, ProviderSettings>();
  for (const item of DEFAULT_PROVIDER_SETTINGS) {
    providersMap.set(item.provider, item);
  }
  if (Array.isArray(providersRaw)) {
    for (const item of providersRaw) {
      if (!item || typeof item !== 'object') {
        continue;
      }
      const providerRaw = (item as { provider?: unknown }).provider;
      if (!isApiKeyProvider(providerRaw)) {
        continue;
      }
      const current = providersMap.get(providerRaw);
      const apiKeyEnv = (item as { api_key_env?: unknown }).api_key_env;
      const endpoint = (item as { endpoint?: unknown }).endpoint;
      const apiKeySet = (item as { api_key_set?: unknown }).api_key_set;
      const apiKeyStored = (item as { api_key_stored?: unknown }).api_key_stored;
      const sourceRaw = (item as { api_key_source?: unknown }).api_key_source;
      const apiKeyValid = (item as { api_key_valid?: unknown }).api_key_valid;
      const lastCheckError = (item as { last_check_error?: unknown }).last_check_error;
      const lastCheckedAt = (item as { last_checked_at?: unknown }).last_checked_at;
      providersMap.set(providerRaw, {
        provider: providerRaw,
        api_key_env: typeof apiKeyEnv === 'string' && apiKeyEnv.trim() ? apiKeyEnv : current?.api_key_env || '',
        endpoint: typeof endpoint === 'string' && endpoint.trim() ? endpoint : current?.endpoint || '',
        api_key_set: typeof apiKeySet === 'boolean' ? apiKeySet : current?.api_key_set || false,
        api_key_stored:
          typeof apiKeyStored === 'boolean' ? apiKeyStored : current?.api_key_stored || false,
        api_key_source: isApiKeySource(sourceRaw) ? sourceRaw : current?.api_key_source || 'missing',
        api_key_valid: typeof apiKeyValid === 'boolean' ? apiKeyValid : current?.api_key_valid ?? null,
        last_check_error:
          typeof lastCheckError === 'string' ? lastCheckError : current?.last_check_error ?? null,
        last_checked_at:
          typeof lastCheckedAt === 'string' ? lastCheckedAt : current?.last_checked_at ?? null,
      });
    }
  }
  const providers = API_KEY_PROVIDERS.map(
    (provider) =>
      providersMap.get(provider)
      || DEFAULT_PROVIDER_SETTINGS.find((item) => item.provider === provider)
      || DEFAULT_PROVIDER_SETTINGS[0],
  );

  let ttsBackend = defaults.ttsBackend;
  const audioRaw = (settings as { audio?: unknown }).audio;
  if (audioRaw && typeof audioRaw === 'object') {
    const ttsRaw = (audioRaw as { tts?: unknown }).tts;
    if (ttsRaw && typeof ttsRaw === 'object') {
      const provider = (ttsRaw as { provider?: unknown }).provider;
      const apiKeyEnv = (ttsRaw as { api_key_env?: unknown }).api_key_env;
      const apiKeySet = (ttsRaw as { api_key_set?: unknown }).api_key_set;
      const endpoint = (ttsRaw as { endpoint?: unknown }).endpoint;
      const model = (ttsRaw as { model?: unknown }).model;
      const voice = (ttsRaw as { voice?: unknown }).voice;
      const format = (ttsRaw as { format?: unknown }).format;
      const backendReady = (ttsRaw as { backend_ready?: unknown }).backend_ready;
      ttsBackend = {
        provider: provider === 'openai' ? 'openai' : defaults.ttsBackend.provider,
        api_key_env:
          typeof apiKeyEnv === 'string' && apiKeyEnv.trim()
            ? apiKeyEnv
            : defaults.ttsBackend.api_key_env,
        api_key_set: typeof apiKeySet === 'boolean' ? apiKeySet : defaults.ttsBackend.api_key_set,
        endpoint:
          typeof endpoint === 'string' && endpoint.trim() ? endpoint : defaults.ttsBackend.endpoint,
        model: typeof model === 'string' && model.trim() ? model : defaults.ttsBackend.model,
        voice: typeof voice === 'string' && voice.trim() ? voice : defaults.ttsBackend.voice,
        format: typeof format === 'string' && format.trim() ? format : defaults.ttsBackend.format,
        backend_ready:
          typeof backendReady === 'boolean' ? backendReady : defaults.ttsBackend.backend_ready,
      };
    }
  }

  return {
    providers,
    ttsBackend,
    tone,
    systemPrompt,
    appearanceTheme,
    longPasteToFileEnabled,
    longPasteThresholdChars,
    memoryInboxMaxItems,
    memoryInboxTtlDays,
    memoryInboxWritesPerMinute,
    embeddingsProvider,
    embeddingsLocalModel,
    embeddingsOpenaiModel,
    defaultModelProvider,
    defaultModelId,
  };
};

const parseEmbeddingRuntime = (payload: unknown): EmbeddingRuntime => {
  const fallback: EmbeddingRuntime = {
    model: DEFAULT_EMBEDDINGS_LOCAL_MODEL,
    state: 'missing',
    error: null,
  };
  if (!payload || typeof payload !== 'object') {
    return fallback;
  }
  const model = (payload as { model?: unknown }).model;
  const state = (payload as { state?: unknown }).state;
  const error = (payload as { error?: unknown }).error;
  const validState = state === 'missing'
    || state === 'package_missing'
    || state === 'downloading'
    || state === 'ready'
    || state === 'error';
  return {
    model: typeof model === 'string' && model.trim() ? model : fallback.model,
    state: validState ? state : fallback.state,
    error: typeof error === 'string' ? error : null,
  };
};

const parseProviderRuntimePayload = (payload: unknown): ProviderRuntimeByModel => {
  const result: ProviderRuntimeByModel = { ...DEFAULT_PROVIDER_RUNTIME };
  if (!payload || typeof payload !== 'object') {
    return result;
  }
  const providersRaw = (payload as { providers?: unknown }).providers;
  if (!Array.isArray(providersRaw)) {
    return result;
  }
  for (const item of providersRaw) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const providerRaw = (item as { provider?: unknown }).provider;
    if (!isModelProvider(providerRaw)) {
      continue;
    }
    const modelsRaw = (item as { models?: unknown }).models;
    const errorRaw = (item as { error?: unknown }).error;
    const modelsCount = Array.isArray(modelsRaw)
      ? modelsRaw.filter((entry): entry is string => typeof entry === 'string' && entry.trim().length > 0).length
      : 0;
    const error =
      typeof errorRaw === 'string' && errorRaw.trim().length > 0 ? errorRaw.trim() : null;
    result[providerRaw] = { modelsCount, error };
  }
  return result;
};

const sourceLabel = (source: ApiKeySource): string => {
  if (source === 'env') {
    return 'Environment';
  }
  if (source === 'file') {
    return 'Saved file';
  }
  return 'Missing';
};

const providerKeyPlaceholder = (provider: ProviderSettings): string => {
  if (provider.api_key_stored) {
    return '•••••••••••• (saved)';
  }
  if (provider.api_key_source === 'env') {
    return `${provider.api_key_env} is active; enter a backup file key`;
  }
  return `Enter ${PROVIDER_LABELS[provider.provider]} API key`;
};

const formatThresholdPreset = (value: number): string => `${Math.round(value / 1000)}k`;

const parseImportPayloadText = (text: string): ImportPreview & { payload: { sessions: unknown[] } } => {
  const parsed = JSON.parse(text) as { sessions?: unknown };
  if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.sessions)) {
    throw new Error('Import file must be a JSON object with a "sessions" array.');
  }
  const sessionsCount = parsed.sessions.length;
  const messagesCount = parsed.sessions.reduce((count, session) => {
    if (!session || typeof session !== 'object') {
      return count;
    }
    const messages = (session as { messages?: unknown }).messages;
    return count + (Array.isArray(messages) ? messages.length : 0);
  }, 0);
  return {
    fileName: '',
    sessionsCount,
    messagesCount,
    payload: {
      sessions: parsed.sessions,
    },
  };
};

type ScopeBadgeProps = {
  children: string;
};

function ScopeBadge({ children }: ScopeBadgeProps) {
  const normalized = children.toLowerCase();
  const tone: StatusTone =
    normalized === 'advanced'
      ? 'waiting'
      : normalized.includes('danger')
        ? 'error'
        : normalized.includes('read')
          ? 'neutral'
          : 'neutral';
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.14em] ${
        tone === 'waiting'
          ? 'border-amber-500/30 bg-amber-500/10 text-amber-300'
          : tone === 'error'
            ? 'border-rose-500/30 bg-rose-500/10 text-rose-300'
            : 'border-zinc-700 bg-zinc-800/60 text-zinc-400'
      }`}
    >
      {children}
    </span>
  );
}

type SectionCardProps = {
  title: string;
  description: string;
  scope?: string;
  children: ReactNode;
};

function SectionCard({ title, description, scope, children }: SectionCardProps) {
  return (
    <section className="border-t border-zinc-800/70 pt-4 pb-5 first:border-t-0 first:pt-0">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[13px] font-medium text-zinc-100">{title}</div>
          <p className="mt-0.5 text-xs text-zinc-500">{description}</p>
        </div>
        {scope ? <ScopeBadge>{scope}</ScopeBadge> : null}
      </div>
      {children}
    </section>
  );
}

function Row({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2.5 first:pt-0 last:pb-0">
      <div className="min-w-0">
        <div className="text-[13px] text-zinc-200">{label}</div>
        {description ? <div className="mt-0.5 text-xs text-zinc-500">{description}</div> : null}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

export function Settings({
  isOpen,
  onClose,
  onSaved,
}: SettingsProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>('assistant');
  const [providers, setProviders] = useState<ProviderSettings[]>(DEFAULT_PROVIDER_SETTINGS);
  const [apiKeys, setApiKeys] = useState<Record<ApiKeyProvider, string>>(EMPTY_API_KEYS);
  const [providerDirty, setProviderDirty] = useState<Record<ApiKeyProvider, boolean>>(
    EMPTY_PROVIDER_DIRTY,
  );
  const [ttsBackend, setTtsBackend] = useState<TtsBackendSettings>(DEFAULT_TTS_BACKEND);
  const [providerRuntime, setProviderRuntime] = useState<ProviderRuntimeByModel>(DEFAULT_PROVIDER_RUNTIME);
  const [providerRuntimeLoading, setProviderRuntimeLoading] = useState(false);
  const [providerRuntimeError, setProviderRuntimeError] = useState<string | null>(null);
  const [defaultModelProvider, setDefaultModelProvider] = useState<ModelProvider>('deepseek');
  const [defaultModelId, setDefaultModelId] = useState('');
  const [defaultModelOptions, setDefaultModelOptions] = useState<string[]>([]);
  const [defaultModelsLoading, setDefaultModelsLoading] = useState(false);
  const [tone, setTone] = useState('balanced');
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [appearanceTheme, setAppearanceTheme] = useState<AppearanceTheme>(
    DEFAULT_APPEARANCE_THEME,
  );
  const [showAssistantAdvanced, setShowAssistantAdvanced] = useState(false);
  const [longPasteToFileEnabled, setLongPasteToFileEnabled] = useState(DEFAULT_LONG_PASTE_TO_FILE_ENABLED);
  const [longPasteThresholdChars, setLongPasteThresholdChars] = useState(DEFAULT_LONG_PASTE_THRESHOLD_CHARS);
  const [memoryInboxMaxItems, setMemoryInboxMaxItems] = useState(DEFAULT_MEMORY_INBOX_MAX_ITEMS);
  const [memoryInboxTtlDays, setMemoryInboxTtlDays] = useState(DEFAULT_MEMORY_INBOX_TTL_DAYS);
  const [memoryInboxWritesPerMinute, setMemoryInboxWritesPerMinute] = useState(DEFAULT_MEMORY_INBOX_WRITES_PER_MINUTE);
  const [embeddingsProvider, setEmbeddingsProvider] = useState<EmbeddingsProvider>(DEFAULT_EMBEDDINGS_PROVIDER);
  const [embeddingsLocalModel, setEmbeddingsLocalModel] = useState(DEFAULT_EMBEDDINGS_LOCAL_MODEL);
  const [embeddingsOpenaiModel, setEmbeddingsOpenaiModel] = useState(DEFAULT_EMBEDDINGS_OPENAI_MODEL);
  const [embeddingRuntime, setEmbeddingRuntime] = useState<EmbeddingRuntime>({
    model: DEFAULT_EMBEDDINGS_LOCAL_MODEL,
    state: 'missing',
    error: null,
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [exportingChats, setExportingChats] = useState(false);
  const [importingChats, setImportingChats] = useState(false);
  const [importMode, setImportMode] = useState<ImportMode>('merge');
  const [importPreview, setImportPreview] = useState<ImportPreview | null>(null);
  const [importPayloadText, setImportPayloadText] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);

  const presetIsCustom = !THRESHOLD_PRESETS.includes(longPasteThresholdChars);

  useFocusTrap(isOpen, modalRef);

  const applyParsedSettings = (parsed: ParsedSettings): void => {
    setProviders(parsed.providers);
    setApiKeys({ ...EMPTY_API_KEYS });
    setProviderDirty({ ...EMPTY_PROVIDER_DIRTY });
    setTtsBackend(parsed.ttsBackend);
    setTone(parsed.tone);
    setSystemPrompt(parsed.systemPrompt);
    setAppearanceTheme(parsed.appearanceTheme);
    setLongPasteToFileEnabled(parsed.longPasteToFileEnabled);
    setLongPasteThresholdChars(parsed.longPasteThresholdChars);
    setMemoryInboxMaxItems(parsed.memoryInboxMaxItems);
    setMemoryInboxTtlDays(parsed.memoryInboxTtlDays);
    setMemoryInboxWritesPerMinute(parsed.memoryInboxWritesPerMinute);
    setEmbeddingsProvider(parsed.embeddingsProvider);
    setEmbeddingsLocalModel(parsed.embeddingsLocalModel);
    setEmbeddingsOpenaiModel(parsed.embeddingsOpenaiModel);
    setDefaultModelProvider(parsed.defaultModelProvider);
    setDefaultModelId(parsed.defaultModelId);
    setDefaultModelOptions(parsed.defaultModelId ? [parsed.defaultModelId] : []);
  };

  const refreshEmbeddingRuntime = async (): Promise<void> => {
    const response = await fetch('/ui/api/embeddings/status');
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to load embedding model status.'));
    }
    setEmbeddingRuntime(parseEmbeddingRuntime(payload));
  };

  const loadSettings = async (): Promise<void> => {
    setLoading(true);
    setStatus(null);
    setProviderRuntimeLoading(true);
    setProviderRuntimeError(null);
    try {
      const [settingsResponse, providerRuntimeResponse, embeddingResponse] = await Promise.all([
        fetch('/ui/api/settings'),
        fetch('/ui/api/models'),
        fetch('/ui/api/embeddings/status'),
      ]);
      const settingsPayload: unknown = await settingsResponse.json();
      if (!settingsResponse.ok) {
        throw new Error(extractErrorMessage(settingsPayload, 'Failed to load settings.'));
      }
      applyParsedSettings(parseSettingsPayload(settingsPayload));

      const providerRuntimePayload: unknown = await providerRuntimeResponse.json();
      if (!providerRuntimeResponse.ok) {
        throw new Error(extractErrorMessage(providerRuntimePayload, 'Failed to load provider diagnostics.'));
      }
      setProviderRuntime(parseProviderRuntimePayload(providerRuntimePayload));

      const embeddingPayload: unknown = await embeddingResponse.json();
      if (!embeddingResponse.ok) {
        throw new Error(extractErrorMessage(embeddingPayload, 'Failed to load embedding model status.'));
      }
      setEmbeddingRuntime(parseEmbeddingRuntime(embeddingPayload));
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load settings.';
      setProviderRuntime({ ...DEFAULT_PROVIDER_RUNTIME });
      setProviderRuntimeError(message);
      setStatus(message);
    } finally {
      setProviderRuntimeLoading(false);
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    void loadSettings();
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  useEffect(() => {
    if (!isOpen || embeddingRuntime.state !== 'downloading') {
      return undefined;
    }
    const timer = window.setInterval(() => {
      void refreshEmbeddingRuntime().catch((error: unknown) => {
        setStatus(error instanceof Error ? error.message : 'Failed to refresh embedding model status.');
      });
    }, 1500);
    return () => window.clearInterval(timer);
  }, [embeddingRuntime.state, isOpen]);

  const handleLoadDefaultModels = async (): Promise<void> => {
    if (defaultModelsLoading) {
      return;
    }
    setDefaultModelsLoading(true);
    setStatus(null);
    try {
      if (providerDirty[defaultModelProvider]) {
        const apiKey = apiKeys[defaultModelProvider].trim();
        const saveKeyResponse = await fetch('/ui/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            providers: {
              [defaultModelProvider]: apiKey ? { api_key: apiKey } : null,
            },
          }),
        });
        const saveKeyPayload: unknown = await saveKeyResponse.json();
        if (!saveKeyResponse.ok) {
          throw new Error(extractErrorMessage(saveKeyPayload, `Failed to save ${defaultModelProvider} key.`));
        }
        const parsed = parseSettingsPayload(saveKeyPayload);
        setProviders(parsed.providers);
        setApiKeys((current) => ({ ...current, [defaultModelProvider]: '' }));
        setProviderDirty((current) => ({ ...current, [defaultModelProvider]: false }));
      }
      const params = new URLSearchParams({ provider: defaultModelProvider, strict: '1' });
      const response = await fetch(`/ui/api/models?${params.toString()}`);
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, `Failed to load ${defaultModelProvider} models.`));
      }
      const providersRaw = (payload as { providers?: unknown }).providers;
      const item = Array.isArray(providersRaw) ? providersRaw[0] : null;
      const modelsRaw = item && typeof item === 'object' ? (item as { models?: unknown }).models : null;
      const errorRaw = item && typeof item === 'object' ? (item as { error?: unknown }).error : null;
      if (typeof errorRaw === 'string' && errorRaw.trim()) {
        throw new Error(errorRaw);
      }
      const models = Array.isArray(modelsRaw)
        ? modelsRaw.filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
        : [];
      setDefaultModelOptions(models);
      if (!models.includes(defaultModelId)) {
        setDefaultModelId(models[0] ?? '');
      }
      setStatus(models.length > 0 ? `Loaded ${models.length} ${defaultModelProvider} models.` : 'No models returned.');
    } catch (error) {
      setDefaultModelOptions(defaultModelId ? [defaultModelId] : []);
      setStatus(error instanceof Error ? error.message : 'Failed to load provider models.');
    } finally {
      setDefaultModelsLoading(false);
    }
  };

  const handleDownloadEmbeddings = async (): Promise<void> => {
    setStatus(null);
    try {
      const response = await fetch('/ui/api/embeddings/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true, model: embeddingsLocalModel.trim() }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to download embedding model.'));
      }
      const parsed = parseEmbeddingRuntime(payload);
      setEmbeddingRuntime(parsed);
      setStatus(parsed.state === 'ready' ? 'Embedding model is ready.' : 'Embedding model download started.');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Failed to download embedding model.');
    }
  };

  const handleSave = async () => {
    if (saving) {
      return;
    }
    setSaving(true);
    setStatus(null);
    try {
      const providersPayload: Record<string, { api_key: string } | null> = {};
      let hasProviderChanges = false;
      for (const provider of API_KEY_PROVIDERS) {
        if (!providerDirty[provider]) {
          continue;
        }
        const apiKey = apiKeys[provider].trim();
        providersPayload[provider] = apiKey ? { api_key: apiKey } : null;
        hasProviderChanges = true;
      }

      const payload: Record<string, unknown> = {
        model: defaultModelId.trim()
          ? { provider: defaultModelProvider, model: defaultModelId.trim() }
          : null,
        personalization: {
          tone: tone.trim() || 'balanced',
          system_prompt: systemPrompt,
        },
        appearance: {
          theme: appearanceTheme,
        },
        composer: {
          long_paste_to_file_enabled: longPasteToFileEnabled,
          long_paste_threshold_chars: Math.max(1000, Math.min(80000, longPasteThresholdChars)),
        },
        memory: {
          inbox_max_items: Math.max(1, Math.floor(memoryInboxMaxItems)),
          inbox_ttl_days: Math.max(1, Math.floor(memoryInboxTtlDays)),
          inbox_writes_per_minute: Math.max(1, Math.floor(memoryInboxWritesPerMinute)),
          embeddings: {
            provider: embeddingsProvider,
            local_model: embeddingsLocalModel.trim() || DEFAULT_EMBEDDINGS_LOCAL_MODEL,
            openai_model: embeddingsOpenaiModel.trim() || DEFAULT_EMBEDDINGS_OPENAI_MODEL,
          },
        },
      };
      if (hasProviderChanges) {
        payload.providers = providersPayload;
      }
      const response = await fetch('/ui/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      const body: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(body, 'Failed to save settings.'));
      }
      applyParsedSettings(parseSettingsPayload(body));
      setStatus('Global settings saved.');
      onSaved?.();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save settings.';
      setStatus(message);
    } finally {
      setSaving(false);
    }
  };

  const handleExportChats = async () => {
    if (exportingChats) {
      return;
    }
    setExportingChats(true);
    setStatus(null);
    try {
      const response = await fetch('/ui/api/settings/chats/export');
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to export chats.'));
      }
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `slavikai-chats-${new Date().toISOString().slice(0, 10)}.json`;
      link.click();
      window.URL.revokeObjectURL(url);
      setStatus('Chats export downloaded.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to export chats.';
      setStatus(message);
    } finally {
      setExportingChats(false);
    }
  };

  const handlePickImportFile = () => {
    fileInputRef.current?.click();
  };

  const handleImportFileSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const text = await file.text();
      const parsed = parseImportPayloadText(text);
      setImportPayloadText(text);
      setImportPreview({
        fileName: file.name,
        sessionsCount: parsed.sessionsCount,
        messagesCount: parsed.messagesCount,
      });
      setStatus(`Loaded import file: ${file.name}`);
    } catch (error) {
      setImportPayloadText(null);
      setImportPreview(null);
      const message = error instanceof Error ? error.message : 'Failed to read import file.';
      setStatus(message);
    } finally {
      event.target.value = '';
    }
  };

  const handleImportChats = async () => {
    if (importingChats || !importPayloadText) {
      return;
    }
    setImportingChats(true);
    setStatus(null);
    try {
      const parsed = parseImportPayloadText(importPayloadText);
      const response = await fetch('/ui/api/settings/chats/import', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mode: importMode,
          sessions: parsed.payload.sessions,
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to import chats.'));
      }
      const importedCount =
        typeof (payload as { imported?: unknown }).imported === 'number'
          ? (payload as { imported?: number }).imported
          : null;
      setImportPayloadText(null);
      setImportPreview(null);
      setStatus(
        importedCount !== null
          ? `Imported ${importedCount} session${importedCount === 1 ? '' : 's'}.`
          : 'Chats imported.',
      );
      onSaved?.();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to import chats.';
      setStatus(message);
    } finally {
      setImportingChats(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen ? (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm"
            onClick={onClose}
          />

          <motion.div
            ref={modalRef}
            initial={{ opacity: 0, scale: 0.97, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 20 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            role="dialog"
            aria-modal="true"
            aria-label="Settings"
          >
            <div className="flex max-h-[88vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl shadow-black/60">
              <div className="flex items-center justify-between gap-4 border-b border-zinc-800 px-5 py-4">
                <div className="min-w-0">
                  <h2 className="text-base font-semibold text-zinc-100">Settings</h2>
                  <p className="mt-0.5 truncate text-xs text-zinc-500">
                    Assistant, providers, memory, and data tools.
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      void loadSettings();
                    }}
                    className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-800"
                  >
                    Refresh
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      void handleSave();
                    }}
                    disabled={saving || loading}
                    className="rounded-md border border-zinc-600 bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-100 transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {saving ? 'Saving...' : 'Save changes'}
                  </button>
                  <button
                    type="button"
                    onClick={onClose}
                    className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-800"
                  >
                    Close
                  </button>
                </div>
              </div>

              <div className="flex min-h-0 flex-1 overflow-hidden">
                <nav className="w-48 shrink-0 border-r border-zinc-800 p-2.5">
                  <div className="space-y-0.5">
                    {TAB_ITEMS.map((tab) => (
                      <button
                        key={tab.id}
                        type="button"
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-[13px] transition-colors ${
                          activeTab === tab.id
                            ? 'bg-zinc-800 text-zinc-100'
                            : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200'
                        }`}
                      >
                        {(() => {
                          const Icon = TAB_ICONS[tab.id];
                          return <Icon className="h-4 w-4 shrink-0" />;
                        })()}
                        {tab.title}
                      </button>
                    ))}
                  </div>
                </nav>

                <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5" data-scrollbar="auto">
                  {status ? (
                    <div className="mb-4 rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-xs text-zinc-300">
                      {status}
                    </div>
                  ) : null}
                  {loading ? (
                    <div className="text-sm text-zinc-400">Loading settings...</div>
                  ) : null}

                  {!loading && activeTab === 'api' ? (
                    <div>
                      <SectionCard
                        title="Provider API keys"
                        description="Keys are validated before they are saved to config/api_keys.json. Environment variables always take priority."
                        scope="Global"
                      >
                        <p className="mb-3 text-xs text-zinc-500">
                          Saved keys stay masked and are never returned by the API. Environment
                          variables always take priority.
                        </p>
                        <div>
                          {providers.map((provider) => (
                            <div
                              key={provider.provider}
                              className="border-b border-zinc-800/60 py-3 first:pt-0 last:border-b-0 last:pb-0"
                            >
                              <div className="flex items-center justify-between gap-3">
                                <div className="flex min-w-0 items-center gap-2">
                                  <span className="text-[13px] font-medium text-zinc-100">
                                    {PROVIDER_LABELS[provider.provider]}
                                  </span>
                                  <span className="truncate font-mono text-[11px] text-zinc-500">
                                    {provider.api_key_env}
                                  </span>
                                </div>
                                <div className="flex shrink-0 items-center gap-2">
                                  <StatusBadge tone={provider.api_key_set ? 'success' : 'waiting'}>
                                    {provider.api_key_set ? 'key set' : 'key missing'}
                                  </StatusBadge>
                                  <span className="text-[11px] text-zinc-500">
                                    {sourceLabel(provider.api_key_source)}
                                  </span>
                                  {providerDirty[provider.provider] ? (
                                    <StatusBadge tone="active">unsaved</StatusBadge>
                                  ) : null}
                                </div>
                              </div>
                              <div className="mt-2 flex items-center gap-2">
                                <input
                                  type="password"
                                  autoComplete="new-password"
                                  aria-label={`${PROVIDER_LABELS[provider.provider]} API key`}
                                  value={apiKeys[provider.provider]}
                                  onChange={(event) => {
                                    const value = event.target.value;
                                    setApiKeys((current) => ({
                                      ...current,
                                      [provider.provider]: value,
                                    }));
                                    setProviderDirty((current) => ({
                                      ...current,
                                      [provider.provider]: true,
                                    }));
                                  }}
                                  placeholder={providerKeyPlaceholder(provider)}
                                  className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
                                />
                                {provider.api_key_stored ? (
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setApiKeys((current) => ({
                                        ...current,
                                        [provider.provider]: '',
                                      }));
                                      setProviderDirty((current) => ({
                                        ...current,
                                        [provider.provider]: true,
                                      }));
                                    }}
                                    className="rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-800"
                                  >
                                    Remove
                                  </button>
                                ) : null}
                              </div>
                              {provider.api_key_source === 'env' ? (
                                <div className="mt-1.5 text-xs text-zinc-500">
                                  {provider.api_key_env} overrides any saved file key.
                                </div>
                              ) : null}
                              {provider.last_check_error ? (
                                <div className="mt-1.5 text-xs text-rose-300">
                                  {provider.last_check_error}
                                </div>
                              ) : provider.api_key_valid === true ? (
                                <div className="mt-1.5 text-xs text-emerald-300">
                                  The saved key passed provider validation.
                                </div>
                              ) : null}
                            </div>
                          ))}
                        </div>
                      </SectionCard>

                      <SectionCard
                        title="Default chat model"
                        description="Choose the provider and model used by every new chat. DeepSeek is the recommended beta starting point."
                        scope="Global"
                      >
                        <div className="grid gap-3 md:grid-cols-[minmax(0,0.7fr)_minmax(0,1.3fr)_auto]">
                          <label>
                            <span className="mb-2 block text-sm font-medium text-zinc-300">Provider</span>
                            <select
                              value={defaultModelProvider}
                              onChange={(event) => {
                                const provider = event.target.value;
                                if (isModelProvider(provider)) {
                                  setDefaultModelProvider(provider);
                                  setDefaultModelId('');
                                  setDefaultModelOptions([]);
                                }
                              }}
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                            >
                              {(['deepseek', 'openrouter', 'xai', 'inception', 'local'] as ModelProvider[]).map((provider) => (
                                <option key={provider} value={provider}>{PROVIDER_LABELS[provider]}</option>
                              ))}
                            </select>
                          </label>
                          <label>
                            <span className="mb-2 block text-sm font-medium text-zinc-300">Model</span>
                            <select
                              value={defaultModelId}
                              onChange={(event) => setDefaultModelId(event.target.value)}
                              disabled={defaultModelsLoading || defaultModelOptions.length === 0}
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none disabled:opacity-50"
                            >
                              {defaultModelOptions.length === 0 ? <option value="">Load models first</option> : null}
                              {defaultModelOptions.map((model) => <option key={model} value={model}>{model}</option>)}
                            </select>
                          </label>
                          <button
                            type="button"
                            onClick={() => { void handleLoadDefaultModels(); }}
                            disabled={defaultModelsLoading}
                            className="self-end rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm text-zinc-100 transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {defaultModelsLoading ? 'Loading...' : 'Load models'}
                          </button>
                        </div>
                        <p className="mt-3 text-xs text-zinc-500">
                          A newly entered provider key is validated and saved automatically when you load models.
                        </p>
                      </SectionCard>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'assistant' ? (
                    <div>
                      <SectionCard
                        title="Response style"
                        description="Choose the default tone for answers in the UI."
                        scope="Global"
                      >
                        <div>
                          {RESPONSE_STYLE_OPTIONS.map((option) => (
                            <button
                              key={option.value}
                              type="button"
                              onClick={() => setTone(option.value)}
                              className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-left transition-colors ${
                                tone === option.value
                                  ? 'bg-zinc-800/70'
                                  : 'hover:bg-zinc-900'
                              }`}
                            >
                              <span
                                className={`h-3.5 w-3.5 shrink-0 rounded-full border ${
                                  tone === option.value
                                    ? 'border-zinc-200 bg-zinc-200'
                                    : 'border-zinc-600'
                                }`}
                              />
                              <span className="min-w-0">
                                <span
                                  className={`block text-[13px] ${
                                    tone === option.value ? 'text-zinc-100' : 'text-zinc-300'
                                  }`}
                                >
                                  {option.label}
                                </span>
                                <span className="block text-xs text-zinc-500">
                                  {option.description}
                                </span>
                              </span>
                            </button>
                          ))}
                        </div>
                      </SectionCard>

                      <SectionCard
                        title="Advanced assistant behavior"
                        description="Optional high-impact customization for advanced users."
                        scope="Global"
                      >
                        <button
                          type="button"
                          onClick={() => setShowAssistantAdvanced((current) => !current)}
                          className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-900"
                        >
                          {showAssistantAdvanced ? 'Hide advanced editor' : 'Show advanced editor'}
                        </button>
                        {showAssistantAdvanced ? (
                          <div className="mt-4 space-y-3">
                            <div className="rounded-xl border border-amber-500/30 bg-amber-950/10 px-4 py-3 text-xs text-amber-100/90">
                              Custom instructions apply globally and can significantly change agent behavior. Use this only when the default style options are not enough.
                            </div>
                            <label className="block">
                              <span className="mb-2 block text-sm font-medium text-zinc-300">Custom instructions</span>
                              <textarea
                                value={systemPrompt}
                                onChange={(event) => setSystemPrompt(event.target.value)}
                                rows={8}
                                className="w-full resize-none rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
                                placeholder="Add advanced global instructions for the assistant..."
                              />
                            </label>
                          </div>
                        ) : null}
                      </SectionCard>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'appearance' ? (
                    <div>
                      <SectionCard
                        title="Theme"
                        description="Choose the app background style."
                        scope="Global"
                      >
                        <div className="inline-flex rounded-md border border-zinc-700 bg-zinc-900 p-0.5">
                          {APPEARANCE_THEME_OPTIONS.map((option) => (
                            <button
                              key={option.value}
                              type="button"
                              onClick={() => setAppearanceTheme(option.value)}
                              className={`rounded px-3 py-1.5 text-[13px] transition-colors ${
                                appearanceTheme === option.value
                                  ? 'bg-zinc-700 text-zinc-100'
                                  : 'text-zinc-400 hover:text-zinc-200'
                              }`}
                            >
                              {option.label}
                            </button>
                          ))}
                        </div>
                        <p className="mt-2 text-xs text-zinc-500">
                          {
                            APPEARANCE_THEME_OPTIONS.find(
                              (option) => option.value === appearanceTheme,
                            )?.description
                          }
                        </p>
                      </SectionCard>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'composer' ? (
                    <div>
                      <SectionCard
                        title="Long paste handling"
                        description="Decide when large pasted text should become a file attachment in the message composer."
                        scope="Global"
                      >
                        <Row label="Convert long paste to file" description="Applies to the message composer before sending.">
                          <ToggleSwitch
                            checked={longPasteToFileEnabled}
                            label="Convert long paste to file"
                            onToggle={() => setLongPasteToFileEnabled((current) => !current)}
                          />
                        </Row>
                        <div className="border-t border-zinc-800/60 pt-3">
                          <div className="mb-2 text-[13px] text-zinc-200">
                            Treat pasted text as a file when longer than
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="flex flex-wrap gap-1">
                              {THRESHOLD_PRESETS.map((preset) => (
                                <button
                                  key={preset}
                                  type="button"
                                  onClick={() => setLongPasteThresholdChars(preset)}
                                  className={`rounded-md border px-2.5 py-1 text-xs transition-colors ${
                                    longPasteThresholdChars === preset
                                      ? 'border-zinc-500 bg-zinc-800 text-zinc-100'
                                      : 'border-zinc-700 text-zinc-400 hover:bg-zinc-900'
                                  }`}
                                >
                                  {formatThresholdPreset(preset)}
                                </button>
                              ))}
                            </div>
                            <input
                              type="number"
                              min={1000}
                              max={80000}
                              value={longPasteThresholdChars}
                              onChange={(event) => {
                                const next = Number.parseInt(event.target.value, 10);
                                if (!Number.isNaN(next)) {
                                  setLongPasteThresholdChars(next);
                                }
                              }}
                              className={`w-28 rounded-md border bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 focus:outline-none ${
                                presetIsCustom ? 'border-zinc-500' : 'border-zinc-700'
                              }`}
                            />
                          </div>
                        </div>
                      </SectionCard>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'memory' ? (
                    <div>
                      <SectionCard
                        title="Memory behavior"
                        description="Memory writes require a separate preview and explicit confirmation."
                        scope="Global"
                      >
                        <p className="mb-3 text-xs text-zinc-500">
                          Slavik previews each requested Memory change. Nothing is saved until
                          you confirm or edit it.
                        </p>
                        <Row label="Inbox size" description="Maximum items kept in the inbox.">
                          <input
                            type="number"
                            min={1}
                            value={memoryInboxMaxItems}
                            onChange={(event) => {
                              const next = Number.parseInt(event.target.value, 10);
                              if (!Number.isNaN(next)) {
                                setMemoryInboxMaxItems(next);
                              }
                            }}
                            className="w-28 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                          />
                        </Row>
                        <Row
                          label="Retention window"
                          description="How many days inbox entries stay available."
                        >
                          <input
                            type="number"
                            min={1}
                            value={memoryInboxTtlDays}
                            onChange={(event) => {
                              const next = Number.parseInt(event.target.value, 10);
                              if (!Number.isNaN(next)) {
                                setMemoryInboxTtlDays(next);
                              }
                            }}
                            className="w-28 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                          />
                        </Row>
                        <Row
                          label="Write rate limit"
                          description="Maximum memory writes allowed per minute."
                        >
                          <input
                            type="number"
                            min={1}
                            value={memoryInboxWritesPerMinute}
                            onChange={(event) => {
                              const next = Number.parseInt(event.target.value, 10);
                              if (!Number.isNaN(next)) {
                                setMemoryInboxWritesPerMinute(next);
                              }
                            }}
                            className="w-28 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                          />
                        </Row>
                      </SectionCard>

                      <SectionCard
                        title="Advanced indexing"
                        description="Embedding provider and model used for indexing and semantic retrieval."
                        scope="Advanced"
                      >
                        <div className="inline-flex rounded-md border border-zinc-700 bg-zinc-900 p-0.5">
                          {(['local', 'openai'] as EmbeddingsProvider[]).map((provider) => (
                            <button
                              key={provider}
                              type="button"
                              onClick={() => setEmbeddingsProvider(provider)}
                              className={`rounded px-3 py-1.5 text-[13px] transition-colors ${
                                embeddingsProvider === provider
                                  ? 'bg-zinc-700 text-zinc-100'
                                  : 'text-zinc-400 hover:text-zinc-200'
                              }`}
                            >
                              {provider === 'local' ? 'Local' : 'OpenAI'}
                            </button>
                          ))}
                        </div>
                        <p className="mt-2 text-xs text-zinc-500">
                          {embeddingsProvider === 'local'
                            ? 'Uses a local sentence-transformer model.'
                            : 'Uses the OpenAI key from the environment or API Keys settings.'}
                        </p>

                        {embeddingsProvider === 'local' ? (
                          <div className="mt-3 space-y-3">
                            <Row label="Local embedding model">
                              <input
                                type="text"
                                value={embeddingsLocalModel}
                                onChange={(event) => setEmbeddingsLocalModel(event.target.value)}
                                className="w-64 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                              />
                            </Row>
                            <Row
                              label="Model files"
                              description={
                                embeddingRuntime.error
                                  ? embeddingRuntime.error
                                  : embeddingRuntime.state === 'ready'
                                    ? `${embeddingRuntime.model} is available locally.`
                                    : 'Save the model name first, then download it once.'
                              }
                            >
                              <StatusBadge
                                tone={
                                  embeddingRuntime.state === 'ready'
                                    ? 'success'
                                    : embeddingRuntime.state === 'error' ||
                                        embeddingRuntime.state === 'package_missing'
                                      ? 'error'
                                      : 'waiting'
                                }
                              >
                                {embeddingRuntime.state.replace('_', ' ')}
                              </StatusBadge>
                              <button
                                type="button"
                                onClick={() => {
                                  void handleDownloadEmbeddings();
                                }}
                                disabled={
                                  embeddingRuntime.state === 'downloading' ||
                                  !embeddingsLocalModel.trim()
                                }
                                className="ml-3 inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-100 transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
                              >
                                <Download className="h-3.5 w-3.5" />
                                {embeddingRuntime.state === 'downloading'
                                  ? 'Downloading...'
                                  : embeddingRuntime.state === 'ready'
                                    ? 'Check again'
                                    : 'Download model'}
                              </button>
                            </Row>
                          </div>
                        ) : (
                          <div className="mt-3">
                            <Row label="OpenAI embedding model">
                            <input
                              type="text"
                              value={embeddingsOpenaiModel}
                              onChange={(event) => setEmbeddingsOpenaiModel(event.target.value)}
                                className="w-64 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                            />
                            </Row>
                          </div>
                        )}
                      </SectionCard>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'data' ? (
                    <div>
                      <SectionCard
                        title="Export chats"
                        description="Download the current chat database as JSON."
                        scope="Global"
                      >
                        <div className="flex items-center justify-between gap-4">
                          <div className="text-xs text-zinc-500">
                            Export includes session history and message data available to this UI principal.
                          </div>
                          <button
                            type="button"
                            onClick={() => {
                              void handleExportChats();
                            }}
                            disabled={exportingChats}
                            className="inline-flex shrink-0 items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-100 transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            <Download className="h-4 w-4" />
                            {exportingChats ? 'Exporting...' : 'Export chats'}
                          </button>
                        </div>
                      </SectionCard>

                      <SectionCard
                        title="Import chats"
                        description="Load a previously exported chats JSON file."
                        scope="Global"
                      >
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="application/json,.json"
                          onChange={(event) => {
                            void handleImportFileSelected(event);
                          }}
                          className="hidden"
                        />
                        <div className="flex flex-wrap items-center gap-3">
                          <button
                            type="button"
                            onClick={handlePickImportFile}
                            className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm text-zinc-100 transition-colors hover:bg-zinc-800"
                          >
                            <Upload className="h-4 w-4" />
                            Choose file
                          </button>
                          <label className="flex items-center gap-2 text-sm text-zinc-300">
                            <span>Mode</span>
                            <select
                              value={importMode}
                              onChange={(event) => setImportMode(event.target.value as ImportMode)}
                              className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                            >
                              <option value="merge">Merge</option>
                              <option value="replace">Replace</option>
                            </select>
                          </label>
                          <button
                            type="button"
                            onClick={() => {
                              void handleImportChats();
                            }}
                            disabled={!importPayloadText || importingChats}
                            className="rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm text-zinc-100 transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {importingChats ? 'Importing...' : 'Import chats'}
                          </button>
                        </div>

                        {importPreview ? (
                          <div className="mt-3 text-xs text-zinc-400">
                            <div className="font-medium text-zinc-200">{importPreview.fileName}</div>
                            <div className="mt-1 text-zinc-500">
                              {importPreview.sessionsCount} session{importPreview.sessionsCount === 1 ? '' : 's'} •{' '}
                              {importPreview.messagesCount} message{importPreview.messagesCount === 1 ? '' : 's'}
                            </div>
                          </div>
                        ) : null}
                      </SectionCard>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'diagnostics' ? (
                    <div>
                      <SectionCard
                        title="Read-only diagnostics"
                        description="Connection status for providers and audio backends."
                        scope="Read only"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-xs text-zinc-500">
                            Runtime checks use environment keys first, then keys saved under API Keys.
                          </div>
                          <button
                            type="button"
                            onClick={() => {
                              void loadSettings();
                            }}
                            className="inline-flex shrink-0 items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-100 transition-colors hover:bg-zinc-800"
                          >
                            <RefreshCcw className="h-4 w-4" />
                            Recheck connections
                          </button>
                        </div>
                      </SectionCard>

                      <SectionCard
                        title="Speech backend"
                        description="Status for the text-to-speech backend configured from environment variables."
                        scope="Read only"
                      >
                        <div>
                          <div className="flex items-center justify-between">
                            <span className="text-[13px] font-medium text-zinc-100">
                              OpenAI TTS backend
                            </span>
                            <div className="flex items-center gap-2">
                              <StatusBadge
                                tone={ttsBackend.backend_ready ? 'success' : 'waiting'}
                              >
                                {ttsBackend.backend_ready ? 'ready' : 'not ready'}
                              </StatusBadge>
                              <StatusBadge tone={ttsBackend.api_key_set ? 'success' : 'waiting'}>
                                {ttsBackend.api_key_set ? 'key set' : 'key missing'}
                              </StatusBadge>
                            </div>
                          </div>
                          <div className="mt-2 space-y-1 text-xs text-zinc-500">
                            <div className="break-all">Endpoint: {ttsBackend.endpoint}</div>
                            <div>
                              Model: <span className="font-mono text-zinc-300">{ttsBackend.model}</span>
                              {' · '}
                              Voice: <span className="font-mono text-zinc-300">{ttsBackend.voice}</span>
                              {' · '}
                              Format: <span className="font-mono text-zinc-300">{ttsBackend.format}</span>
                            </div>
                          </div>
                        </div>
                      </SectionCard>

                      <SectionCard
                        title="Provider status"
                        description="API key presence, validation status, and model endpoint probes."
                        scope="Read only"
                      >
                        <div>
                          {providers.map((provider) => {
                            const runtime = isModelProvider(provider.provider)
                              ? providerRuntime[provider.provider]
                              : null;
                            const runtimeLabel = providerRuntimeLoading
                              ? 'checking'
                              : runtime?.error
                                ? 'models error'
                                : runtime && runtime.modelsCount > 0
                                  ? `${runtime.modelsCount} models`
                                  : runtime
                                    ? '0 models'
                                    : providerRuntimeError
                                      ? 'probe failed'
                                      : 'unknown';
                            const runtimeTone: StatusTone = providerRuntimeLoading
                              ? 'neutral'
                              : runtime?.error || providerRuntimeError
                                ? 'error'
                                : runtime && runtime.modelsCount > 0
                                  ? 'success'
                                  : 'waiting';
                            const runtimeDetail = runtime?.error ?? providerRuntimeError;

                            return (
                              <div
                                key={provider.provider}
                                className="border-b border-zinc-800/60 py-3 first:pt-0 last:border-b-0 last:pb-0"
                              >
                                <div className="flex items-center justify-between gap-3">
                                  <div className="flex min-w-0 items-center gap-2">
                                    <span className="text-[13px] font-medium text-zinc-100">
                                      {PROVIDER_LABELS[provider.provider]}
                                    </span>
                                    <span className="font-mono text-[11px] text-zinc-500">
                                      {provider.api_key_env}
                                    </span>
                                  </div>
                                  <div className="flex shrink-0 items-center gap-2">
                                    <StatusBadge
                                      tone={provider.api_key_set ? 'success' : 'waiting'}
                                    >
                                      {provider.api_key_set ? 'key set' : 'key missing'}
                                    </StatusBadge>
                                    <StatusBadge
                                      tone={
                                        provider.api_key_valid === true
                                          ? 'success'
                                          : provider.api_key_valid === false
                                            ? 'error'
                                            : 'neutral'
                                      }
                                    >
                                      {provider.api_key_valid === true
                                        ? 'key valid'
                                        : provider.api_key_valid === false
                                          ? 'key invalid'
                                          : 'unchecked'}
                                    </StatusBadge>
                                    {isModelProvider(provider.provider) ? (
                                      <StatusBadge tone={runtimeTone}>{runtimeLabel}</StatusBadge>
                                    ) : null}
                                  </div>
                                </div>
                                <div className="mt-1 space-y-1 text-xs text-zinc-500">
                                  <div className="break-all">Endpoint: {provider.endpoint}</div>
                                  {provider.last_checked_at ? <div>Last check: {provider.last_checked_at}</div> : null}
                                  {provider.last_check_error ? (
                                    <div className="text-rose-300">Validation: {provider.last_check_error}</div>
                                  ) : null}
                                  {isModelProvider(provider.provider) && runtimeDetail ? (
                                    <div className="text-rose-300">Runtime: {runtimeDetail}</div>
                                  ) : null}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </SectionCard>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          </motion.div>
        </>
      ) : null}
    </AnimatePresence>
  );
}
