import {
  Bot,
  BrainCircuit,
  Database,
  Download,
  KeyRound,
  RefreshCcw,
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

type SettingsTab = 'api' | 'personal' | 'memory' | 'data';
type ApiKeyProvider = 'xai' | 'openrouter' | 'local' | 'inception' | 'openai' | 'deepseek';
type ModelProvider = string;
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

type ProviderInstance = {
  provider: string;
  display_name: string;
  base_url: string;
};

type CatalogSource = 'endpoint' | 'documented' | 'unverified' | 'not_checked';

const CATALOG_SOURCE_LABELS: Record<CatalogSource, string> = {
  endpoint: 'API response',
  documented: 'Documented list',
  unverified: 'Unverified',
  not_checked: 'Not checked',
};

type ProviderRuntimeState = {
  catalogSource: CatalogSource;
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
  api_key_source: ApiKeySource;
};

type ParsedSettings = {
  preferencesEtag: string | null;
  preferencesSubject: string | null;
  canManageInstallation: boolean;
  providers: ProviderSettings[];
  providerInstances: ProviderInstance[];
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
  { id: 'api', title: 'Models and connections' },
  { id: 'personal', title: 'Conversation and interface' },
  { id: 'memory', title: 'Memory' },
  { id: 'data', title: 'Data' },
];

const TAB_ICONS: Record<SettingsTab, typeof KeyRound> = {
  api: KeyRound,
  personal: Bot,
  memory: BrainCircuit,
  data: Database,
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
  api_key_source: 'missing',
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
  typeof value === 'string' && (
    value === 'xai' || value === 'openrouter' || value === 'local'
    || value === 'inception' || value === 'deepseek' || /^custom-[0-9a-f]{32}$/.test(value)
  );

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
    preferencesEtag: null,
    preferencesSubject: null,
    canManageInstallation: false,
    providers: DEFAULT_PROVIDER_SETTINGS,
    providerInstances: [],
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
  const providerInstances: ProviderInstance[] = Array.isArray(providersRaw)
    ? providersRaw.flatMap((item): ProviderInstance[] => {
      if (!item || typeof item !== 'object') return [];
      const entry = item as { provider?: unknown; display_name?: unknown; base_url?: unknown };
      if (typeof entry.provider !== 'string' || !/^custom-[0-9a-f]{32}$/.test(entry.provider)
        || typeof entry.display_name !== 'string' || typeof entry.base_url !== 'string'
      ) return [];
      return [{
        provider: entry.provider,
        display_name: entry.display_name,
        base_url: entry.base_url,
      }];
    })
    : [];

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
      const apiKeySource = (ttsRaw as { api_key_source?: unknown }).api_key_source;
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
        api_key_source: isApiKeySource(apiKeySource) ? apiKeySource : defaults.ttsBackend.api_key_source,
      };
    }
  }

  const metadata = settings as { preferences_etag?: unknown; preferences_subject?: unknown; can_manage_installation?: unknown };
  return {
    preferencesEtag: typeof metadata.preferences_etag === 'string' ? metadata.preferences_etag : null,
    preferencesSubject: typeof metadata.preferences_subject === 'string' ? metadata.preferences_subject : null,
    canManageInstallation: metadata.can_manage_installation === true,
    providers,
    providerInstances,
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
    const source = (item as { catalog_source?: unknown }).catalog_source;
    const catalogSource: CatalogSource = source === 'endpoint' || source === 'documented' || source === 'not_checked'
      ? source : 'unverified';
    result[providerRaw] = { modelsCount, error, catalogSource };
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

function ProviderConnectionStatus({ provider, runtime, loading, error }: {
  provider: ProviderSettings;
  runtime: ProviderRuntimeState | null;
  loading: boolean;
  error: string | null;
}) {
  const runtimeLabel = loading
    ? 'checking'
    : runtime?.catalogSource === 'not_checked'
      ? 'not checked'
      : runtime?.error
        ? 'models error'
      : runtime && runtime.modelsCount > 0
        ? `${runtime.modelsCount} models`
        : runtime
          ? '0 models'
          : error
            ? 'probe failed'
            : 'unknown';
  const runtimeTone: StatusTone = loading
    ? 'neutral'
    : runtime?.error || error
      ? 'error'
      : runtime && runtime.modelsCount > 0 && runtime.catalogSource === 'endpoint'
        ? 'success'
        : 'waiting';
  const runtimeDetail = runtime?.error ?? error;

  return (
    <div className="mt-2 space-y-2 text-xs text-zinc-500">
      <div className="flex flex-wrap gap-2">
        <StatusBadge tone={provider.api_key_valid === true ? 'success' : provider.api_key_valid === false ? 'error' : 'neutral'}>
          {provider.api_key_valid === true ? 'key valid' : provider.api_key_valid === false ? 'key invalid' : 'unchecked'}
        </StatusBadge>
        {isModelProvider(provider.provider) ? <StatusBadge tone={runtimeTone}>{runtimeLabel}</StatusBadge> : null}
      </div>
      {runtime ? <div>Catalog source: {CATALOG_SOURCE_LABELS[runtime.catalogSource]}</div>
        : isModelProvider(provider.provider) ? <div>Catalog source: {error ? 'Unverified' : 'Not checked'}</div> : null}
      {isModelProvider(provider.provider) && runtimeDetail ? <div className="text-rose-300">Runtime: {runtimeDetail}</div> : null}
      <details>
        <summary className="cursor-pointer text-zinc-400">Connection details</summary>
        <div className="mt-2 space-y-1">
          <div className="break-all">Endpoint: {provider.endpoint}</div>
          <div>Credential variable: {provider.api_key_env}</div>
          {provider.last_checked_at ? <div>Last key check: {provider.last_checked_at}</div> : null}
        </div>
      </details>
    </div>
  );
}

export function Settings({
  isOpen,
  onClose,
  onSaved,
}: SettingsProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>('personal');
  const [canManageInstallation, setCanManageInstallation] = useState(false);
  const [settingsReady, setSettingsReady] = useState(false);
  const preferencesEtag = useRef<string | null>(null);
  const preferencesSubject = useRef<string | null>(null);
  const [providers, setProviders] = useState<ProviderSettings[]>(DEFAULT_PROVIDER_SETTINGS);
  const [providerInstances, setProviderInstances] = useState<ProviderInstance[]>([]);
  const [newProviderName, setNewProviderName] = useState('');
  const [newProviderBaseUrl, setNewProviderBaseUrl] = useState('');
  const [newProviderKey, setNewProviderKey] = useState('');
  const [newProviderProbeStatus, setNewProviderProbeStatus] = useState<string | null>(null);
  const [providerOnboardingBusy, setProviderOnboardingBusy] = useState(false);
  const [apiKeys, setApiKeys] = useState<Record<ApiKeyProvider, string>>(EMPTY_API_KEYS);
  const [providerDirty, setProviderDirty] = useState<Record<ApiKeyProvider, boolean>>(
    EMPTY_PROVIDER_DIRTY,
  );
  const [ttsBackend, setTtsBackend] = useState<TtsBackendSettings>(DEFAULT_TTS_BACKEND);
  const [credentialDiagnosticsLoading, setCredentialDiagnosticsLoading] = useState(false);
  const [credentialDiagnosticsError, setCredentialDiagnosticsError] = useState<string | null>(null);
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
  const [showInputAdvanced, setShowInputAdvanced] = useState(false);
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
  const [embeddingRuntimeLoading, setEmbeddingRuntimeLoading] = useState(false);
  const [embeddingRuntimeError, setEmbeddingRuntimeError] = useState<string | null>(null);
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
  const onboardingEpoch = useRef(0);
  const providerInstancesRevision = useRef(0);
  const settingsRequest = useRef<AbortController | null>(null);
  const credentialDiagnosticsRequest = useRef<AbortController | null>(null);
  const providerRuntimeRequest = useRef<AbortController | null>(null);
  const embeddingStatusRequest = useRef<AbortController | null>(null);
  const pendingSettingsWrites = useRef(new Set<Promise<void>>());
  const formEdits = useRef(0);
  // Reconciliation updates untouched preferences without replacing a newer draft.
  const dirtyFields = useRef(new Set<string>());
  const reconciliationNeeded = useRef(false);
  const pendingEmbeddingWrites = useRef(new Set<Promise<void>>());

  const presetIsCustom = !THRESHOLD_PRESETS.includes(longPasteThresholdChars);

  useFocusTrap(isOpen, modalRef);

  const invalidateDiagnostics = (credentialsChanged = true): void => {
    if (credentialsChanged) {
      credentialDiagnosticsRequest.current?.abort();
      setCredentialDiagnosticsLoading(false);
      setCredentialDiagnosticsError(null);
      providerRuntimeRequest.current?.abort();
      setProviderRuntime({ ...DEFAULT_PROVIDER_RUNTIME });
      setProviderRuntimeLoading(false);
      setProviderRuntimeError(null);
    }
    embeddingStatusRequest.current?.abort();
    setEmbeddingRuntimeLoading(false);
    setEmbeddingRuntimeError(null);
  };

  const applyCredentialSnapshot = (parsed: ParsedSettings): void => {
    setProviders(parsed.providers);
    setTtsBackend(parsed.ttsBackend);
  };

  const applyParsedSettings = (parsed: ParsedSettings, providerRevisionAtRequestStart: number, preserveDraft = false): void => {
    const subjectChanged = preferencesSubject.current !== null && preferencesSubject.current !== parsed.preferencesSubject;
    preferencesSubject.current = parsed.preferencesSubject;
    if (subjectChanged) {
      onboardingEpoch.current += 1;
      invalidateDiagnostics();
      setNewProviderName('');
      setNewProviderBaseUrl('');
      setNewProviderKey('');
      setNewProviderProbeStatus(null);
      setProviderOnboardingBusy(false);
      preserveDraft = false;
      dirtyFields.current.clear();
    }
    if (!parsed.canManageInstallation) {
      setActiveTab((tab) => ['api', 'memory'].includes(tab) ? 'personal' : tab);
      setApiKeys({ ...EMPTY_API_KEYS });
      setProviderDirty({ ...EMPTY_PROVIDER_DIRTY });
    }
    setSettingsReady(true);
    setCanManageInstallation(parsed.canManageInstallation);
    const privateDirty = ['tone', 'systemPrompt', 'appearanceTheme', 'longPasteToFileEnabled', 'longPasteThresholdChars'].some((key) => dirtyFields.current.has(key));
    if (!preserveDraft || !privateDirty) preferencesEtag.current = parsed.preferencesEtag;
    applyCredentialSnapshot(parsed);
    setProviderInstances((current) => {
      if (subjectChanged || providerInstancesRevision.current === providerRevisionAtRequestStart) {
        return parsed.providerInstances;
      }
      // A provider confirmed after this snapshot was requested must survive
      // a late settings response. Keep the server's entries when IDs overlap.
      const snapshotIds = new Set(parsed.providerInstances.map((item) => item.provider));
      return [...parsed.providerInstances, ...current.filter((item) => !snapshotIds.has(item.provider))];
    });
    if (!preserveDraft) setApiKeys({ ...EMPTY_API_KEYS });
    if (!preserveDraft) setProviderDirty({ ...EMPTY_PROVIDER_DIRTY });
    if (!preserveDraft || !dirtyFields.current.has('tone')) setTone(parsed.tone);
    if (!preserveDraft || !dirtyFields.current.has('systemPrompt')) setSystemPrompt(parsed.systemPrompt);
    if (!preserveDraft || !dirtyFields.current.has('appearanceTheme')) setAppearanceTheme(parsed.appearanceTheme);
    if (!preserveDraft || !dirtyFields.current.has('longPasteToFileEnabled')) setLongPasteToFileEnabled(parsed.longPasteToFileEnabled);
    if (!preserveDraft || !dirtyFields.current.has('longPasteThresholdChars')) setLongPasteThresholdChars(parsed.longPasteThresholdChars);
    if (!preserveDraft || !dirtyFields.current.has('memoryInboxMaxItems')) setMemoryInboxMaxItems(parsed.memoryInboxMaxItems);
    if (!preserveDraft || !dirtyFields.current.has('memoryInboxTtlDays')) setMemoryInboxTtlDays(parsed.memoryInboxTtlDays);
    if (!preserveDraft || !dirtyFields.current.has('memoryInboxWritesPerMinute')) setMemoryInboxWritesPerMinute(parsed.memoryInboxWritesPerMinute);
    if (!preserveDraft || !dirtyFields.current.has('embeddingsProvider')) setEmbeddingsProvider(parsed.embeddingsProvider);
    if (!preserveDraft || !dirtyFields.current.has('embeddingsLocalModel')) setEmbeddingsLocalModel(parsed.embeddingsLocalModel);
    if (!preserveDraft || !dirtyFields.current.has('embeddingsOpenaiModel')) setEmbeddingsOpenaiModel(parsed.embeddingsOpenaiModel);
    if (!preserveDraft || !dirtyFields.current.has('defaultModelProvider')) setDefaultModelProvider(parsed.defaultModelProvider);
    if (!preserveDraft || !dirtyFields.current.has('defaultModelId')) setDefaultModelId(parsed.defaultModelId);
    if (!preserveDraft || !dirtyFields.current.has('defaultModelId')) setDefaultModelOptions(parsed.defaultModelId ? [parsed.defaultModelId] : []);
    if (!preserveDraft) dirtyFields.current.clear();
  };

  const refreshEmbeddingRuntime = async (restart = false): Promise<void> => {
    if (restart) {
      embeddingStatusRequest.current?.abort();
    } else if (embeddingStatusRequest.current && !embeddingStatusRequest.current.signal.aborted) {
      return;
    }
    const controller = new AbortController();
    embeddingStatusRequest.current = controller;
    setEmbeddingRuntimeLoading(true);
    setEmbeddingRuntimeError(null);
    try {
      await Promise.all([...pendingSettingsWrites.current, ...pendingEmbeddingWrites.current]);
      if (controller.signal.aborted) return;
      const response = await fetch('/ui/api/embeddings/status', { signal: controller.signal });
      const payload: unknown = await response.json();
      if (controller.signal.aborted) return;
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to load embedding model status.'));
      }
      setEmbeddingRuntime(parseEmbeddingRuntime(payload));
    } catch (error) {
      if (controller.signal.aborted) return;
      setEmbeddingRuntimeError(error instanceof Error ? error.message : 'Failed to load embedding model status.');
    } finally {
      if (!controller.signal.aborted) {
        embeddingStatusRequest.current = null;
        setEmbeddingRuntimeLoading(false);
      }
    }
  };

  const beginSettingsWrite = (credentialsChanged: boolean): (() => void) => {
    // Cancel observations of the previous configuration before starting a write.
    // A later Recheck waits for all owned writes, then probes the committed state.
    invalidateDiagnostics(credentialsChanged);
    settingsRequest.current?.abort();
    settingsRequest.current = null;
    setLoading(false);
    let finish!: () => void;
    const pending = new Promise<void>((resolve) => { finish = resolve; });
    pendingSettingsWrites.current.add(pending);
    void refreshEmbeddingRuntime();
    return () => {
      pendingSettingsWrites.current.delete(pending);
      finish();
    };
  };

  const loadCredentialDiagnostics = async (): Promise<void> => {
    credentialDiagnosticsRequest.current?.abort();
    const controller = new AbortController();
    credentialDiagnosticsRequest.current = controller;
    setCredentialDiagnosticsLoading(true);
    setCredentialDiagnosticsError(null);
    try {
      await Promise.all([...pendingSettingsWrites.current]);
      if (controller.signal.aborted) return;
      const response = await fetch('/ui/api/settings', { signal: controller.signal });
      const payload: unknown = await response.json();
      if (controller.signal.aborted) return;
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to refresh credential diagnostics.'));
      }
      const parsed = parseSettingsPayload(payload);
      // Recheck refreshes read-only metadata without applying editable preferences.
      setTtsBackend(parsed.ttsBackend);
      setProviders(parsed.providers);
    } catch (error) {
      if (controller.signal.aborted) return;
      setCredentialDiagnosticsError(error instanceof Error ? error.message : 'Failed to refresh credential diagnostics.');
    } finally {
      if (!controller.signal.aborted) {
        credentialDiagnosticsRequest.current = null;
        setCredentialDiagnosticsLoading(false);
      }
    }
  };

  const loadSettings = async (preserveDraft = false): Promise<void> => {
    if (!preserveDraft) invalidateDiagnostics();
    settingsRequest.current?.abort();
    const controller = new AbortController();
    settingsRequest.current = controller;
    const providerRevisionAtRequestStart = providerInstancesRevision.current;
    const writesBeforeSnapshot = [...pendingSettingsWrites.current];
    if (writesBeforeSnapshot.length > 0) reconciliationNeeded.current = true;
    if (!preserveDraft) { setLoading(true); setSettingsReady(false); }
    if (!preserveDraft) setStatus(null);
    if (!preserveDraft) void refreshEmbeddingRuntime();
    try {
      if (preserveDraft) {
        await Promise.all(writesBeforeSnapshot);
        if (controller.signal.aborted) return;
      }
      const settingsResponse = await fetch('/ui/api/settings', { signal: controller.signal });
      const settingsPayload: unknown = await settingsResponse.json();
      if (controller.signal.aborted) return;
      if (!settingsResponse.ok) {
        throw new Error(extractErrorMessage(settingsPayload, 'Failed to load settings.'));
      }
      applyParsedSettings(parseSettingsPayload(settingsPayload), providerRevisionAtRequestStart, preserveDraft);
      setLoading(false);
      if (writesBeforeSnapshot.length > 0 && !preserveDraft) {
        // Show the current snapshot immediately, then reconcile the completed write.
        // Preserve a draft created while waiting rather than silently replacing it.
        const editsAtSnapshot = formEdits.current;
        await Promise.all(writesBeforeSnapshot);
        if (controller.signal.aborted) return;
        const refreshedResponse = await fetch('/ui/api/settings', { signal: controller.signal });
        const refreshedPayload: unknown = await refreshedResponse.json();
        if (controller.signal.aborted) return;
        if (!refreshedResponse.ok) {
          throw new Error(extractErrorMessage(refreshedPayload, 'Failed to refresh saved settings.'));
        }
        const refreshed = parseSettingsPayload(refreshedPayload);
        if (formEdits.current === editsAtSnapshot) {
          applyParsedSettings(refreshed, providerRevisionAtRequestStart);
        } else {
          applyParsedSettings(refreshed, providerRevisionAtRequestStart, true);
          setStatus('Settings changed on the server. Your draft was kept; Refresh loads the saved settings.');
        }
      }
      reconciliationNeeded.current = false;
    } catch (error) {
      if (controller.signal.aborted) return;
      setStatus(error instanceof Error ? error.message : 'Failed to load settings.');
    } finally {
      if (!controller.signal.aborted) {
        settingsRequest.current = null;
        setLoading(false);
      }
    }
  };

  const loadProviderDiagnostics = async (): Promise<void> => {
    providerRuntimeRequest.current?.abort();
    const controller = new AbortController();
    providerRuntimeRequest.current = controller;
    setProviderRuntimeLoading(true);
    setProviderRuntimeError(null);
    try {
      await Promise.all([...pendingSettingsWrites.current]);
      if (controller.signal.aborted) return;
      const providerRuntimeResponse = await fetch('/ui/api/models?skip_custom_probe=1&discovery_only=1', {
        signal: controller.signal,
      });
      const providerRuntimePayload: unknown = await providerRuntimeResponse.json();
      if (controller.signal.aborted) return;
      if (!providerRuntimeResponse.ok) {
        throw new Error(extractErrorMessage(providerRuntimePayload, 'Failed to load provider diagnostics.'));
      }
      setProviderRuntime(parseProviderRuntimePayload(providerRuntimePayload));
    } catch (error) {
      if (controller.signal.aborted) return;
      setProviderRuntime({ ...DEFAULT_PROVIDER_RUNTIME });
      setProviderRuntimeError(error instanceof Error ? error.message : 'Failed to load provider diagnostics.');
    } finally {
      if (!controller.signal.aborted) {
        providerRuntimeRequest.current = null;
        setProviderRuntimeLoading(false);
      }
    }
  };

  useEffect(() => {
    if (!isOpen) {
      setSettingsReady(false);
      setShowInputAdvanced(false);
      setCanManageInstallation(false);
      preferencesEtag.current = null;
      setTone('balanced');
      setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
      setAppearanceTheme(DEFAULT_APPEARANCE_THEME);
      setLongPasteToFileEnabled(DEFAULT_LONG_PASTE_TO_FILE_ENABLED);
      setLongPasteThresholdChars(DEFAULT_LONG_PASTE_THRESHOLD_CHARS);
      setApiKeys({ ...EMPTY_API_KEYS });
      setProviderDirty({ ...EMPTY_PROVIDER_DIRTY });
      setImportPayloadText(null);
      setImportPreview(null);
      dirtyFields.current.clear();
      onboardingEpoch.current += 1;
      // Release the busy state owned by the invalidated operation so a
      // reopened Settings is never stuck disabled by a stale probe/save.
      setProviderOnboardingBusy(false);
      setNewProviderName('');
      setNewProviderBaseUrl('');
      setNewProviderKey('');
      setNewProviderProbeStatus(null);
      return;
    }
    setProviderRuntime({ ...DEFAULT_PROVIDER_RUNTIME });
    setProviderRuntimeLoading(false);
    setProviderRuntimeError(null);
    void loadSettings();
    return () => {
      settingsRequest.current?.abort();
      credentialDiagnosticsRequest.current?.abort();
      providerRuntimeRequest.current?.abort();
      embeddingStatusRequest.current?.abort();
    };
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
      void refreshEmbeddingRuntime();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [embeddingRuntime.state, isOpen]);

  const handleLoadDefaultModels = async (): Promise<void> => {
    if (defaultModelsLoading) {
      return;
    }
    setDefaultModelsLoading(true);
    setStatus(null);
    const epoch = onboardingEpoch.current;
    let finishWrite: (() => void) | undefined;
    try {
      if (isApiKeyProvider(defaultModelProvider) && providerDirty[defaultModelProvider]) {
        const apiKey = apiKeys[defaultModelProvider].trim();
        finishWrite = beginSettingsWrite(true);
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
        if (epoch !== onboardingEpoch.current) return;
        if (!saveKeyResponse.ok) {
          throw new Error(extractErrorMessage(saveKeyPayload, `Failed to save ${defaultModelProvider} key.`));
        }
        const parsed = parseSettingsPayload(saveKeyPayload);
        applyCredentialSnapshot(parsed);
        setApiKeys((current) => ({ ...current, [defaultModelProvider]: '' }));
        setProviderDirty((current) => ({ ...current, [defaultModelProvider]: false }));
        finishWrite();
        finishWrite = undefined;
        if (reconciliationNeeded.current) await loadSettings(true);
      }
      await Promise.all([...pendingSettingsWrites.current]);
      if (epoch !== onboardingEpoch.current) return;
      const params = new URLSearchParams({ provider: defaultModelProvider, strict: '1' });
      const response = await fetch(`/ui/api/models?${params.toString()}`);
      const payload: unknown = await response.json();
      if (epoch !== onboardingEpoch.current) return;
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, `Failed to load ${defaultModelProvider} models.`));
      }
      const providersRaw = (payload as { providers?: unknown }).providers;
      const item = Array.isArray(providersRaw) ? providersRaw[0] : null;
      const modelsRaw = item && typeof item === 'object' ? (item as { models?: unknown }).models : null;
      const errorRaw = item && typeof item === 'object' ? (item as { error?: unknown }).error : null;
      if (typeof errorRaw === 'string' && errorRaw.trim() && !defaultModelProvider.startsWith('custom-')) {
        throw new Error(errorRaw);
      }
      const models = Array.isArray(modelsRaw)
        ? modelsRaw.filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
        : [];
      setDefaultModelOptions(models);
      if (!models.includes(defaultModelId)) {
        setDefaultModelId(models[0] ?? '');
      }
      setStatus(typeof errorRaw === 'string' && errorRaw.trim()
        ? errorRaw
        : models.length > 0 ? `Loaded ${models.length} ${defaultModelProvider} models.` : 'No models returned.');
    } catch (error) {
      if (epoch !== onboardingEpoch.current) return;
      setDefaultModelOptions(defaultModelId ? [defaultModelId] : []);
      setStatus(error instanceof Error ? error.message : 'Failed to load provider models.');
    } finally {
      if (finishWrite) {
        finishWrite();
        if (epoch === onboardingEpoch.current && reconciliationNeeded.current) void loadSettings(true);
      }
      setDefaultModelsLoading(false);
    }
  };

  const handleProbeNewProvider = async (): Promise<void> => {
    const epoch = onboardingEpoch.current;
    setProviderOnboardingBusy(true);
    setNewProviderProbeStatus(null);
    try {
      const response = await fetch('/ui/api/provider-instances/probe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base_url: newProviderBaseUrl, api_key: newProviderKey }),
      });
      const payload: unknown = await response.json();
      if (epoch !== onboardingEpoch.current) return;
      if (!response.ok) throw new Error(extractErrorMessage(payload, 'Provider probe failed.'));
      const result = payload as { base_url?: unknown; models?: unknown; status?: unknown; message?: unknown };
      const models = Array.isArray(result.models)
        ? result.models.filter((item): item is string => typeof item === 'string' && item.length > 0)
        : [];
      if (typeof result.base_url === 'string') setNewProviderBaseUrl(result.base_url);
      setNewProviderProbeStatus(typeof result.message === 'string' && result.message
        ? result.message
        : `Connected. Found ${models.length} models.`);
    } catch (error) {
      if (epoch !== onboardingEpoch.current) return;
      setNewProviderProbeStatus(error instanceof Error ? error.message : 'Provider probe failed.');
    } finally {
      // Only the operation that owns the current epoch may release the busy
      // state; a stale probe must not clear a newer request's busy flag.
      if (epoch === onboardingEpoch.current) {
        setProviderOnboardingBusy(false);
      }
    }
  };

  const handleSaveNewProvider = async (): Promise<void> => {
    const epoch = onboardingEpoch.current;
    setProviderOnboardingBusy(true);
    setStatus(null);
    try {
      const response = await fetch('/ui/api/provider-instances', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          display_name: newProviderName,
          base_url: newProviderBaseUrl,
          api_key: newProviderKey,
        }),
      });
      const payload: unknown = await response.json();
      const stale = epoch !== onboardingEpoch.current;
      if (!response.ok) {
        // A stale error must remain silent for the new epoch.
        if (stale) return;
        throw new Error(extractErrorMessage(payload, 'Could not save provider.'));
      }
      const created = payload as { provider?: unknown; display_name?: unknown; base_url?: unknown };
      if (typeof created.provider !== 'string' || typeof created.display_name !== 'string'
        || typeof created.base_url !== 'string') {
        if (stale) return;
        throw new Error('Provider saved, but response was incomplete.');
      }
      const providerId = created.provider;
      const displayName = created.display_name;
      const baseUrl = created.base_url;
      providerInstancesRevision.current += 1;
      // The server may have persisted the provider even when the response is
      // stale for the new epoch: reconcile it into the list (deduplicated)
      // and notify the picker, but never touch the new epoch's form fields,
      // probe status, status message, or busy state.
      setProviderInstances((current) => current.some((item) => item.provider === providerId)
        ? current
        : [...current, {
          provider: providerId,
          display_name: displayName,
          base_url: baseUrl,
        }]);
      onSaved?.();
      if (stale) return;
      setNewProviderName('');
      setNewProviderBaseUrl('');
      setNewProviderKey('');
      setNewProviderProbeStatus(null);
      setStatus('Provider saved. Select it in the chat model picker when ready.');
    } catch (error) {
      // Stale error must not write its status into the new epoch.
      if (epoch !== onboardingEpoch.current) return;
      setStatus(error instanceof Error ? error.message : 'Could not save provider.');
    } finally {
      // Same ownership rule as the probe: a stale save must not clear the
      // busy state of a newer onboarding operation.
      if (epoch === onboardingEpoch.current) {
        setProviderOnboardingBusy(false);
      }
    }
  };

  const handleDownloadEmbeddings = async (): Promise<void> => {
    embeddingStatusRequest.current?.abort();
    setEmbeddingRuntimeLoading(false);
    setEmbeddingRuntimeError(null);
    setStatus(null);
    const epoch = onboardingEpoch.current;
    let finish!: () => void;
    const pending = new Promise<void>((resolve) => { finish = resolve; });
    pendingEmbeddingWrites.current.add(pending);
    try {
      const response = await fetch('/ui/api/embeddings/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true, model: embeddingsLocalModel.trim() }),
      });
      const payload: unknown = await response.json();
      if (epoch !== onboardingEpoch.current) return;
      if (!response.ok) {
        if (payload && typeof payload === 'object' && 'error' in payload) {
          const error = payload.error;
          if (error && typeof error === 'object' && 'details' in error) {
            const details = error.details;
            if (details && typeof details === 'object' && 'state' in details && details.state === 'error') {
              setEmbeddingRuntime(parseEmbeddingRuntime(details));
            }
          }
        }
        throw new Error(extractErrorMessage(payload, 'Failed to download embedding model.'));
      }
      const parsed = parseEmbeddingRuntime(payload);
      setEmbeddingRuntime(parsed);
      setStatus(parsed.state === 'ready' ? 'Embedding model is ready.' : 'Embedding model download started.');
    } catch (error) {
      if (epoch === onboardingEpoch.current) setStatus(error instanceof Error ? error.message : 'Failed to download embedding model.');
    } finally {
      pendingEmbeddingWrites.current.delete(pending);
      finish();
    }
  };

  const handleSave = async () => {
    if (saving) {
      return;
    }
    const providerRevisionAtRequestStart = providerInstancesRevision.current;
    const epoch = onboardingEpoch.current;
    const privateEtagAtRequestStart = preferencesEtag.current;
    setSaving(true);
    setStatus(null);
    let finishWrite: (() => void) | undefined;
    let installationCommitted = false;
    let personalPending = false;
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
      const changed = (fields: string[]) => fields.some((field) => dirtyFields.current.has(field));
      const personal: Record<string, unknown> = {};
      if (changed(['tone', 'systemPrompt'])) personal.personalization = payload.personalization;
      if (changed(['appearanceTheme'])) personal.appearance = payload.appearance;
      if (changed(['longPasteToFileEnabled', 'longPasteThresholdChars'])) personal.composer = payload.composer;
      const installation: Record<string, unknown> = {};
      if (changed(['defaultModelProvider', 'defaultModelId'])) installation.model = payload.model;
      if (changed(['memoryInboxMaxItems', 'memoryInboxTtlDays', 'memoryInboxWritesPerMinute', 'embeddingsProvider', 'embeddingsLocalModel', 'embeddingsOpenaiModel'])) installation.memory = payload.memory;
      if (hasProviderChanges) installation.providers = payload.providers;
      const requests: Array<{ payload: Record<string, unknown>; personal: boolean }> = [];
      if (Object.keys(installation).length > 0) {
        if (!canManageInstallation) throw new Error('Owner access required for installation settings.');
        requests.push({ payload: installation, personal: false });
      }
      if (Object.keys(personal).length > 0) {
        if (!privateEtagAtRequestStart) throw new Error('Refresh Settings before saving preferences.');
        requests.push({ payload: personal, personal: true });
        personalPending = true;
      }
      if (requests.length === 0) { setStatus('No changes to save.'); return; }
      finishWrite = beginSettingsWrite(hasProviderChanges);
      let lastBody: unknown;
      for (const item of requests) {
        if (epoch !== onboardingEpoch.current) {
          if (installationCommitted) onSaved?.();
          return;
        }
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (item.personal && privateEtagAtRequestStart) headers['If-Match'] = privateEtagAtRequestStart;
        const response = await fetch('/ui/api/settings', {
          method: 'POST', headers, body: JSON.stringify(item.payload),
        });
        lastBody = await response.json();
        if (!response.ok) {
          throw new Error(extractErrorMessage(lastBody, 'Failed to save settings.'));
        }
        if (item.personal) personalPending = false;
        else installationCommitted = true;
      }
      onSaved?.();
      if (epoch !== onboardingEpoch.current) return;
      applyParsedSettings(parseSettingsPayload(lastBody), providerRevisionAtRequestStart);
      setStatus('Settings saved.');
    } catch (error) {
      if (installationCommitted) onSaved?.();
      if (epoch !== onboardingEpoch.current) return;
      const message = error instanceof Error ? error.message : 'Failed to save settings.';
      setStatus(installationCommitted && personalPending
        ? `Installation settings saved. Preferences were not saved: ${message}`
        : message);
    } finally {
      finishWrite?.();
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
            onChangeCapture={(event) => {
              if (event.target instanceof Element) {
                const field = event.target.closest('[data-settings-field]')?.getAttribute('data-settings-field');
                if (field) {
                  dirtyFields.current.add(field);
                  if (field === 'defaultModelProvider') dirtyFields.current.add('defaultModelId');
                  formEdits.current += 1;
                }
              }
            }}
            onClickCapture={(event) => {
              if (event.target instanceof Element && event.target.closest('[data-edits-settings], [role="switch"]')) {
                const field = event.target.closest('[data-settings-field]')?.getAttribute('data-settings-field');
                if (field) dirtyFields.current.add(field);
                formEdits.current += 1;
              }
            }}
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
                    disabled={saving || loading || !settingsReady}
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
                    {TAB_ITEMS.filter((tab) => canManageInstallation || !['api', 'memory'].includes(tab.id)).map((tab) => (
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

                  {canManageInstallation && settingsReady && !loading && activeTab === 'api' ? (
                    <div>
                      <SectionCard
                        title="Connection checks"
                        description="Connection status for providers and audio backends."
                        scope="Read only"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-xs text-zinc-500">
                            Runtime checks use environment keys first, then saved provider keys.
                          </div>
                          <button
                            type="button"
                            onClick={() => {
                              void loadProviderDiagnostics();
                              void loadCredentialDiagnostics();
                              void refreshEmbeddingRuntime(true);
                            }}
                            className="inline-flex shrink-0 items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-100 transition-colors hover:bg-zinc-800"
                          >
                            <RefreshCcw className="h-4 w-4" />
                            Recheck connections
                          </button>
                        </div>
                      </SectionCard>
                      <SectionCard
                        title="Provider connections"
                        description="Manage credentials and inspect catalog evidence for each connection. Key and catalog checks do not verify inference or tool support."
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
                              role="group"
                              aria-label={`${PROVIDER_LABELS[provider.provider]} connection`}
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
                                  data-settings-field="apiKeys"
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
                                    data-edits-settings
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
                              <ProviderConnectionStatus
                                provider={provider}
                                runtime={isModelProvider(provider.provider) ? providerRuntime[provider.provider] : null}
                                loading={providerRuntimeLoading}
                                error={providerRuntimeError}
                              />
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
                        title="Add provider"
                        description="Connect an OpenAI-compatible endpoint. Saving it does not switch the active model."
                        scope="Global"
                      >
                        <div className="grid gap-3">
                          <input
                            aria-label="Provider display name"
                            placeholder="Display name"
                            value={newProviderName}
                            onChange={(event) => setNewProviderName(event.target.value)}
                            className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100"
                          />
                          <input
                            aria-label="Provider Base URL"
                            placeholder="Base URL (https://host or https://host/v1)"
                            value={newProviderBaseUrl}
                            onChange={(event) => {
                              setNewProviderBaseUrl(event.target.value);
                              setNewProviderProbeStatus(null);
                            }}
                            className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100"
                          />
                          <input
                            type="password"
                            autoComplete="new-password"
                            aria-label="New provider API key"
                            placeholder="API key"
                            value={newProviderKey}
                            onChange={(event) => {
                              setNewProviderKey(event.target.value);
                              setNewProviderProbeStatus(null);
                            }}
                            className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100"
                          />
                          <button
                            type="button"
                            onClick={() => { void handleProbeNewProvider(); }}
                            disabled={providerOnboardingBusy || !newProviderBaseUrl.trim() || !newProviderKey.trim()}
                            className="w-fit rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm text-zinc-100 disabled:opacity-40"
                          >
                            {providerOnboardingBusy ? 'Checking...' : 'Test connection'}
                          </button>
                          {newProviderProbeStatus ? <p className="text-xs text-zinc-400">{newProviderProbeStatus}</p> : null}
                          <button
                            type="button"
                            onClick={() => { void handleSaveNewProvider(); }}
                            disabled={providerOnboardingBusy || !newProviderName.trim()
                              || !newProviderBaseUrl.trim() || !newProviderKey.trim()}
                            className="w-fit rounded-lg border border-zinc-600 bg-zinc-800 px-4 py-2 text-sm text-zinc-100 disabled:opacity-40"
                          >
                            Save provider
                          </button>
                        </div>
                        {providerInstances.length > 0 ? (
                          <div className="mt-4 space-y-2 text-xs text-zinc-400">
                            {providerInstances.map((item) => (
                              <div key={item.provider}>
                                <span className="text-zinc-200">{item.display_name}</span>
                                {' · '}{item.base_url}{' · key saved'}
                              </div>
                            ))}
                          </div>
                        ) : null}
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
                              data-settings-field="defaultModelProvider"
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
                                <option key={provider} value={provider}>{isApiKeyProvider(provider) ? PROVIDER_LABELS[provider] : provider}</option>
                              ))}
                            </select>
                          </label>
                          <label>
                            <span className="mb-2 block text-sm font-medium text-zinc-300">Model</span>
                            <select
                              data-settings-field="defaultModelId"
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
                            data-edits-settings
                            data-settings-field="defaultModelId"
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

                  {settingsReady && !loading && activeTab === 'personal' ? (
                    <div>
                      <SectionCard
                        title="Response style"
                        description="Applies to your next browser chat request. Existing runs keep their original preferences."
                        scope="You"
                      >
                        <div>
                          {RESPONSE_STYLE_OPTIONS.map((option) => (
                            <button
                              key={option.value}
                              type="button"
                              data-edits-settings
                              data-settings-field="tone"
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
                        title="Your instructions"
                        description="Private instructions for your next browser chat request. System policy and approvals still apply."
                        scope="You"
                      >
                        <div className="mt-4 space-y-3">
                          <div className="rounded-xl border border-amber-500/30 bg-amber-950/10 px-4 py-3 text-xs text-amber-100/90">
                            These instructions apply to your browser chats, below system policy. They do not grant tool permissions or replace task approvals.
                          </div>
                          <label className="block">
                            <span className="mb-2 block text-sm font-medium text-zinc-300">Custom instructions</span>
                            <textarea
                              data-settings-field="systemPrompt"
                              value={systemPrompt}
                              onChange={(event) => setSystemPrompt(event.target.value)}
                              rows={8}
                              className="w-full resize-none rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
                              placeholder="Save your private instructions..."
                            />
                          </label>
                        </div>
                      </SectionCard>
                      <SectionCard
                        title="Theme"
                        description="Choose the app background style."
                        scope="You"
                      >
                        <div className="inline-flex rounded-md border border-zinc-700 bg-zinc-900 p-0.5">
                          {APPEARANCE_THEME_OPTIONS.map((option) => (
                            <button
                              key={option.value}
                              type="button"
                              data-edits-settings
                              data-settings-field="appearanceTheme"
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
                      <SectionCard
                        title="Long paste handling"
                        description="Decide when large pasted text should become a file attachment in the message composer."
                        scope="You"
                      >
                        <Row label="Convert long paste to file" description="Applies to the message composer before sending.">
                          <ToggleSwitch
                            checked={longPasteToFileEnabled}
                            label="Convert long paste to file"
                            onToggle={() => {
                              dirtyFields.current.add('longPasteToFileEnabled');
                              setLongPasteToFileEnabled((current) => !current);
                            }}
                          />
                        </Row>
                        <button
                          type="button"
                          aria-expanded={showInputAdvanced}
                          onClick={() => setShowInputAdvanced((current) => !current)}
                          className="mt-3 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-900"
                        >
                          Advanced input settings
                        </button>
                        {showInputAdvanced ? (
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
                                    data-edits-settings
                                    data-settings-field="longPasteThresholdChars"
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
                                aria-label="Long paste threshold (characters)"
                                data-settings-field="longPasteThresholdChars"
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
                        ) : null}
                      </SectionCard>
                    </div>
                  ) : null}

                  {canManageInstallation && settingsReady && !loading && activeTab === 'memory' ? (
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
                            data-settings-field="memoryInboxMaxItems"
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
                          label="Inbox retention window"
                          description="Applies only to inbox entries, not chat history, confirmed Memory, or vectors."
                        >
                          <input
                            type="number"
                            min={1}
                            data-settings-field="memoryInboxTtlDays"
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
                            data-settings-field="memoryInboxWritesPerMinute"
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
                              data-edits-settings
                              data-settings-field="embeddingsProvider"
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
                            ? 'Processes embeddings locally using downloaded model files. Without the package or files, semantic search is unavailable; chat remains available. Download is explicit.'
                            : 'Sends text for indexing and search queries to OpenAI. Select and save this provider to enable remote embeddings using your configured OpenAI key.'}
                        </p>

                        {embeddingsProvider === 'local' ? (
                          <div className="mt-3 space-y-3">
                            <Row label="Local embedding model">
                              <input
                                type="text"
                                data-settings-field="embeddingsLocalModel"
                                value={embeddingsLocalModel}
                                onChange={(event) => setEmbeddingsLocalModel(event.target.value)}
                                className="w-64 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                              />
                            </Row>
                            <Row
                              label="Model files"
                              description={
                                embeddingRuntimeError ||
                                  (embeddingRuntime.state === 'package_missing'
                                    ? 'Local embeddings library is missing. On the SlavikAI server, run make install-beta, restart the server, then recheck. Automatic installation is unavailable.'
                                    : embeddingRuntime.error) ||
                                  (embeddingRuntime.state === 'ready'
                                    ? `${embeddingRuntime.model} is available locally.`
                                    : 'Save the model name first, then download it once.')
                              }
                            >
                              <StatusBadge
                                tone={
                                  embeddingRuntimeLoading
                                    ? 'neutral'
                                    : embeddingRuntimeError
                                      ? 'error'
                                      : embeddingRuntime.state === 'ready'
                                        ? 'success'
                                        : embeddingRuntime.state === 'error' ||
                                            embeddingRuntime.state === 'package_missing'
                                          ? 'error'
                                          : 'waiting'
                                }
                              >
                                {embeddingRuntimeLoading
                                  ? 'checking'
                                  : embeddingRuntimeError
                                    ? 'error'
                                    : embeddingRuntime.state.replace('_', ' ')}
                              </StatusBadge>
                              {embeddingRuntime.state !== 'package_missing' ? <button
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
                              </button> : null}
                            </Row>
                          </div>
                        ) : (
                          <div className="mt-3">
                            <Row label="OpenAI embedding model">
                            <input
                              type="text"
                              data-settings-field="embeddingsOpenaiModel"
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

                  {settingsReady && !loading && activeTab === 'data' ? (
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

                  {canManageInstallation && settingsReady && !loading && activeTab === 'api' ? (
                    <div>
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
                                tone={credentialDiagnosticsLoading ? 'neutral' : credentialDiagnosticsError ? 'error' : ttsBackend.api_key_set ? 'neutral' : 'waiting'}
                              >
                                {credentialDiagnosticsLoading ? 'checking' : credentialDiagnosticsError ? 'check failed' : ttsBackend.api_key_set ? 'Configured' : 'Not configured'}
                              </StatusBadge>
                              <StatusBadge tone={ttsBackend.api_key_set ? 'success' : 'waiting'}>
                                {ttsBackend.api_key_set ? 'key set' : 'key missing'}
                              </StatusBadge>
                            </div>
                          </div>
                          <div className="mt-2 space-y-1 text-xs text-zinc-500">
                            <div>Key source: {sourceLabel(ttsBackend.api_key_source)}</div>
                            <div>Speech inference has not been tested.</div>
                            {credentialDiagnosticsError ? <div>Credential check: {credentialDiagnosticsError}</div> : null}
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
