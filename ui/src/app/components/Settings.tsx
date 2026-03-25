import { Download, RefreshCcw, Upload } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import type { ChangeEvent, ReactNode } from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';

interface SettingsProps {
  isOpen: boolean;
  onClose: () => void;
  onSaved?: () => void;
}

type SettingsTab = 'assistant' | 'composer' | 'memory' | 'data' | 'diagnostics';
type ApiKeyProvider = 'xai' | 'openrouter' | 'local' | 'inception' | 'openai';
type ModelProvider = 'xai' | 'openrouter' | 'local' | 'inception';
type ApiKeySource = 'env' | 'missing';
type EmbeddingsProvider = 'local' | 'openai';
type ImportMode = 'replace' | 'merge';

type ProviderSettings = {
  provider: ApiKeyProvider;
  api_key_env: string;
  api_key_set: boolean;
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
  longPasteToFileEnabled: boolean;
  longPasteThresholdChars: number;
  memoryAutoSaveDialogue: boolean;
  memoryInboxMaxItems: number;
  memoryInboxTtlDays: number;
  memoryInboxWritesPerMinute: number;
  embeddingsProvider: EmbeddingsProvider;
  embeddingsLocalModel: string;
  embeddingsOpenaiModel: string;
};

type ProviderRuntimeByModel = Record<ModelProvider, ProviderRuntimeState | null>;

type ImportPreview = {
  fileName: string;
  sessionsCount: number;
  messagesCount: number;
};

const DEFAULT_SYSTEM_PROMPT =
  'You are SlavikAI, a helpful AI assistant with MWV architecture.';

const API_KEY_PROVIDERS: ApiKeyProvider[] = ['xai', 'openrouter', 'local', 'inception', 'openai'];

const PROVIDER_LABELS: Record<ApiKeyProvider, string> = {
  xai: 'xAI',
  openrouter: 'OpenRouter',
  local: 'Local',
  inception: 'Inception',
  openai: 'OpenAI',
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
  { id: 'assistant', title: 'Assistant' },
  { id: 'composer', title: 'Composer' },
  { id: 'memory', title: 'Memory' },
  { id: 'data', title: 'Data' },
  { id: 'diagnostics', title: 'Diagnostics' },
];

const THRESHOLD_PRESETS = [8000, 12000, 25000];

const DEFAULT_LONG_PASTE_TO_FILE_ENABLED = true;
const DEFAULT_LONG_PASTE_THRESHOLD_CHARS = 12000;
const DEFAULT_EMBEDDINGS_PROVIDER: EmbeddingsProvider = 'local';
const DEFAULT_EMBEDDINGS_LOCAL_MODEL = 'all-MiniLM-L6-v2';
const DEFAULT_EMBEDDINGS_OPENAI_MODEL = 'text-embedding-3-small';
const DEFAULT_MEMORY_AUTO_SAVE_DIALOGUE = false;
const DEFAULT_MEMORY_INBOX_MAX_ITEMS = 200;
const DEFAULT_MEMORY_INBOX_TTL_DAYS = 30;
const DEFAULT_MEMORY_INBOX_WRITES_PER_MINUTE = 6;

const DEFAULT_PROVIDER_SETTINGS: ProviderSettings[] = [
  {
    provider: 'xai',
    api_key_env: 'XAI_API_KEY',
    api_key_set: false,
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
    api_key_source: 'missing',
    endpoint: 'https://api.openai.com/v1/audio/transcriptions',
    api_key_valid: null,
    last_check_error: null,
    last_checked_at: null,
  },
  {
    provider: 'inception',
    api_key_env: 'INCEPTION_API_KEY',
    api_key_set: false,
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
};

const isApiKeyProvider = (value: unknown): value is ApiKeyProvider =>
  value === 'xai' || value === 'openrouter' || value === 'local' || value === 'inception' || value === 'openai';

const isApiKeySource = (value: unknown): value is ApiKeySource =>
  value === 'env' || value === 'missing';

const isModelProvider = (value: unknown): value is ModelProvider =>
  value === 'xai' || value === 'openrouter' || value === 'local' || value === 'inception';

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
    longPasteToFileEnabled: DEFAULT_LONG_PASTE_TO_FILE_ENABLED,
    longPasteThresholdChars: DEFAULT_LONG_PASTE_THRESHOLD_CHARS,
    memoryAutoSaveDialogue: DEFAULT_MEMORY_AUTO_SAVE_DIALOGUE,
    memoryInboxMaxItems: DEFAULT_MEMORY_INBOX_MAX_ITEMS,
    memoryInboxTtlDays: DEFAULT_MEMORY_INBOX_TTL_DAYS,
    memoryInboxWritesPerMinute: DEFAULT_MEMORY_INBOX_WRITES_PER_MINUTE,
    embeddingsProvider: DEFAULT_EMBEDDINGS_PROVIDER,
    embeddingsLocalModel: DEFAULT_EMBEDDINGS_LOCAL_MODEL,
    embeddingsOpenaiModel: DEFAULT_EMBEDDINGS_OPENAI_MODEL,
  };

  if (!payload || typeof payload !== 'object') {
    return defaults;
  }
  const settings = (payload as { settings?: unknown }).settings;
  if (!settings || typeof settings !== 'object') {
    return defaults;
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

  let memoryAutoSaveDialogue = defaults.memoryAutoSaveDialogue;
  let memoryInboxMaxItems = defaults.memoryInboxMaxItems;
  let memoryInboxTtlDays = defaults.memoryInboxTtlDays;
  let memoryInboxWritesPerMinute = defaults.memoryInboxWritesPerMinute;
  let embeddingsProvider = defaults.embeddingsProvider;
  let embeddingsLocalModel = defaults.embeddingsLocalModel;
  let embeddingsOpenaiModel = defaults.embeddingsOpenaiModel;
  const memory = (settings as { memory?: unknown }).memory;
  if (memory && typeof memory === 'object') {
    const autoSaveRaw = (memory as { auto_save_dialogue?: unknown }).auto_save_dialogue;
    const inboxMaxItemsRaw = (memory as { inbox_max_items?: unknown }).inbox_max_items;
    const inboxTtlDaysRaw = (memory as { inbox_ttl_days?: unknown }).inbox_ttl_days;
    const inboxWritesRaw = (memory as { inbox_writes_per_minute?: unknown }).inbox_writes_per_minute;
    if (typeof autoSaveRaw === 'boolean') {
      memoryAutoSaveDialogue = autoSaveRaw;
    }
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
      const sourceRaw = (item as { api_key_source?: unknown }).api_key_source;
      const apiKeyValid = (item as { api_key_valid?: unknown }).api_key_valid;
      const lastCheckError = (item as { last_check_error?: unknown }).last_check_error;
      const lastCheckedAt = (item as { last_checked_at?: unknown }).last_checked_at;
      providersMap.set(providerRaw, {
        provider: providerRaw,
        api_key_env: typeof apiKeyEnv === 'string' && apiKeyEnv.trim() ? apiKeyEnv : current?.api_key_env || '',
        endpoint: typeof endpoint === 'string' && endpoint.trim() ? endpoint : current?.endpoint || '',
        api_key_set: typeof apiKeySet === 'boolean' ? apiKeySet : current?.api_key_set || false,
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
    longPasteToFileEnabled,
    longPasteThresholdChars,
    memoryAutoSaveDialogue,
    memoryInboxMaxItems,
    memoryInboxTtlDays,
    memoryInboxWritesPerMinute,
    embeddingsProvider,
    embeddingsLocalModel,
    embeddingsOpenaiModel,
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
  return 'Missing';
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

type ToggleSwitchProps = {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onToggle: () => void;
};

function ToggleSwitch({ checked, disabled = false, label, onToggle }: ToggleSwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={onToggle}
      className={`relative inline-flex h-6 w-10 shrink-0 items-center rounded-full p-[2px] transition-colors ${
        checked ? 'bg-emerald-500/75' : 'bg-zinc-700'
      } disabled:cursor-not-allowed disabled:opacity-40`}
    >
      <span
        className={`block h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
          checked ? 'translate-x-4' : 'translate-x-0'
        }`}
      />
    </button>
  );
}

type ScopeBadgeProps = {
  children: string;
};

function ScopeBadge({ children }: ScopeBadgeProps) {
  return (
    <span className="rounded-full border border-zinc-700 px-2.5 py-1 text-[11px] uppercase tracking-[0.18em] text-zinc-400">
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
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-medium text-zinc-100">{title}</div>
          <p className="mt-1 text-xs text-zinc-400">{description}</p>
        </div>
        {scope ? <ScopeBadge>{scope}</ScopeBadge> : null}
      </div>
      {children}
    </section>
  );
}

export function Settings({
  isOpen,
  onClose,
  onSaved,
}: SettingsProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>('assistant');
  const [providers, setProviders] = useState<ProviderSettings[]>(DEFAULT_PROVIDER_SETTINGS);
  const [ttsBackend, setTtsBackend] = useState<TtsBackendSettings>(DEFAULT_TTS_BACKEND);
  const [providerRuntime, setProviderRuntime] = useState<ProviderRuntimeByModel>(DEFAULT_PROVIDER_RUNTIME);
  const [providerRuntimeLoading, setProviderRuntimeLoading] = useState(false);
  const [providerRuntimeError, setProviderRuntimeError] = useState<string | null>(null);
  const [tone, setTone] = useState('balanced');
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [showAssistantAdvanced, setShowAssistantAdvanced] = useState(false);
  const [longPasteToFileEnabled, setLongPasteToFileEnabled] = useState(DEFAULT_LONG_PASTE_TO_FILE_ENABLED);
  const [longPasteThresholdChars, setLongPasteThresholdChars] = useState(DEFAULT_LONG_PASTE_THRESHOLD_CHARS);
  const [memoryAutoSaveDialogue, setMemoryAutoSaveDialogue] = useState(DEFAULT_MEMORY_AUTO_SAVE_DIALOGUE);
  const [memoryInboxMaxItems, setMemoryInboxMaxItems] = useState(DEFAULT_MEMORY_INBOX_MAX_ITEMS);
  const [memoryInboxTtlDays, setMemoryInboxTtlDays] = useState(DEFAULT_MEMORY_INBOX_TTL_DAYS);
  const [memoryInboxWritesPerMinute, setMemoryInboxWritesPerMinute] = useState(DEFAULT_MEMORY_INBOX_WRITES_PER_MINUTE);
  const [embeddingsProvider, setEmbeddingsProvider] = useState<EmbeddingsProvider>(DEFAULT_EMBEDDINGS_PROVIDER);
  const [embeddingsLocalModel, setEmbeddingsLocalModel] = useState(DEFAULT_EMBEDDINGS_LOCAL_MODEL);
  const [embeddingsOpenaiModel, setEmbeddingsOpenaiModel] = useState(DEFAULT_EMBEDDINGS_OPENAI_MODEL);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [exportingChats, setExportingChats] = useState(false);
  const [importingChats, setImportingChats] = useState(false);
  const [importMode, setImportMode] = useState<ImportMode>('merge');
  const [importPreview, setImportPreview] = useState<ImportPreview | null>(null);
  const [importPayloadText, setImportPayloadText] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const selectedStyle = useMemo(
    () => RESPONSE_STYLE_OPTIONS.find((option) => option.value === tone) || RESPONSE_STYLE_OPTIONS[0],
    [tone],
  );
  const presetIsCustom = !THRESHOLD_PRESETS.includes(longPasteThresholdChars);

  const applyParsedSettings = (parsed: ParsedSettings): void => {
    setProviders(parsed.providers);
    setTtsBackend(parsed.ttsBackend);
    setTone(parsed.tone);
    setSystemPrompt(parsed.systemPrompt);
    setLongPasteToFileEnabled(parsed.longPasteToFileEnabled);
    setLongPasteThresholdChars(parsed.longPasteThresholdChars);
    setMemoryAutoSaveDialogue(parsed.memoryAutoSaveDialogue);
    setMemoryInboxMaxItems(parsed.memoryInboxMaxItems);
    setMemoryInboxTtlDays(parsed.memoryInboxTtlDays);
    setMemoryInboxWritesPerMinute(parsed.memoryInboxWritesPerMinute);
    setEmbeddingsProvider(parsed.embeddingsProvider);
    setEmbeddingsLocalModel(parsed.embeddingsLocalModel);
    setEmbeddingsOpenaiModel(parsed.embeddingsOpenaiModel);
  };

  const loadSettings = async (): Promise<void> => {
    setLoading(true);
    setStatus(null);
    setProviderRuntimeLoading(true);
    setProviderRuntimeError(null);
    try {
      const [settingsResponse, providerRuntimeResponse] = await Promise.all([
        fetch('/ui/api/settings'),
        fetch('/ui/api/models'),
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

  const handleSave = async () => {
    if (saving) {
      return;
    }
    setSaving(true);
    setStatus(null);
    try {
      const payload = {
        personalization: {
          tone: tone.trim() || 'balanced',
          system_prompt: systemPrompt,
        },
        composer: {
          long_paste_to_file_enabled: longPasteToFileEnabled,
          long_paste_threshold_chars: Math.max(1000, Math.min(80000, longPasteThresholdChars)),
        },
        memory: {
          auto_save_dialogue: memoryAutoSaveDialogue,
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
            initial={{ opacity: 0, scale: 0.97, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 20 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="flex max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl shadow-black/60">
              <div className="flex items-center justify-between border-b border-zinc-800 p-6">
                <div>
                  <div className="mb-1 text-xs tracking-[0.18em] text-zinc-500">GLOBAL SETTINGS</div>
                  <h2 className="text-xl font-semibold text-zinc-100">Preferences and Diagnostics</h2>
                  <p className="mt-1 text-sm text-zinc-400">
                    Persistent settings for the assistant, composer behavior, memory, and data tools.
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      void loadSettings();
                    }}
                    className="rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-800"
                  >
                    Refresh
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      void handleSave();
                    }}
                    disabled={saving || loading}
                    className="rounded-lg border border-zinc-600 bg-zinc-800 px-4 py-2 text-sm font-medium text-zinc-100 transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {saving ? 'Saving...' : 'Save changes'}
                  </button>
                  <button
                    type="button"
                    onClick={onClose}
                    className="rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-800"
                  >
                    Close
                  </button>
                </div>
              </div>

              <div className="flex min-h-0 flex-1 overflow-hidden">
                <div className="w-56 border-r border-zinc-800 p-4">
                  <div className="space-y-1">
                    {TAB_ITEMS.map((tab) => (
                      <button
                        key={tab.id}
                        type="button"
                        onClick={() => setActiveTab(tab.id)}
                        className={`w-full rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                          activeTab === tab.id
                            ? 'bg-zinc-200 text-zinc-900'
                            : 'text-zinc-300 hover:bg-zinc-900'
                        }`}
                      >
                        {tab.title}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="min-h-0 flex-1 overflow-y-auto p-6" data-scrollbar="auto">
                  {status ? <div className="mb-4 text-sm text-zinc-300">{status}</div> : null}
                  {loading ? <div className="text-sm text-zinc-400">Loading settings...</div> : null}

                  {!loading && activeTab === 'assistant' ? (
                    <div className="space-y-6">
                      <SectionCard
                        title="Response style"
                        description="Choose the default tone for answers in the UI."
                        scope="Global"
                      >
                        <div className="grid gap-3 md:grid-cols-2">
                          {RESPONSE_STYLE_OPTIONS.map((option) => (
                            <button
                              key={option.value}
                              type="button"
                              onClick={() => setTone(option.value)}
                              className={`rounded-xl border p-4 text-left transition-colors ${
                                tone === option.value
                                  ? 'border-zinc-500 bg-zinc-800 text-zinc-100'
                                  : 'border-zinc-800 bg-zinc-950 text-zinc-300 hover:bg-zinc-900'
                              }`}
                            >
                              <div className="text-sm font-medium">{option.label}</div>
                              <div className="mt-2 text-xs text-zinc-400">{option.description}</div>
                            </button>
                          ))}
                        </div>
                        <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-3 text-xs text-zinc-400">
                          Active style: <span className="text-zinc-200">{selectedStyle.label}</span> • {selectedStyle.description}
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

                  {!loading && activeTab === 'composer' ? (
                    <div className="space-y-6">
                      <SectionCard
                        title="Long paste handling"
                        description="Decide when large pasted text should become a file attachment in the message composer."
                        scope="Global"
                      >
                        <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                          <div className="flex items-center justify-between gap-3">
                            <div>
                              <div className="text-sm font-medium text-zinc-100">Convert long paste to file</div>
                              <p className="mt-1 text-xs text-zinc-400">
                                Applies to the message composer before sending.
                              </p>
                            </div>
                            <ToggleSwitch
                              checked={longPasteToFileEnabled}
                              label="Convert long paste to file"
                              onToggle={() => setLongPasteToFileEnabled((current) => !current)}
                            />
                          </div>
                        </div>

                        <div className="space-y-3">
                          <div className="text-sm font-medium text-zinc-200">
                            Treat pasted text as file when longer than...
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {THRESHOLD_PRESETS.map((preset) => (
                              <button
                                key={preset}
                                type="button"
                                onClick={() => setLongPasteThresholdChars(preset)}
                                className={`rounded-lg border px-3 py-2 text-sm transition-colors ${
                                  longPasteThresholdChars === preset
                                    ? 'border-zinc-500 bg-zinc-800 text-zinc-100'
                                    : 'border-zinc-700 bg-zinc-950 text-zinc-400 hover:bg-zinc-900'
                                }`}
                              >
                                {formatThresholdPreset(preset)}
                              </button>
                            ))}
                            <button
                              type="button"
                              className={`rounded-lg border px-3 py-2 text-sm transition-colors ${
                                presetIsCustom
                                  ? 'border-zinc-500 bg-zinc-800 text-zinc-100'
                                  : 'border-zinc-700 bg-zinc-950 text-zinc-400'
                              }`}
                            >
                              Custom
                            </button>
                          </div>
                          <label className="block">
                            <span className="mb-1 block text-xs font-medium text-zinc-300">Character threshold</span>
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
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
                            />
                          </label>
                        </div>
                      </SectionCard>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'memory' ? (
                    <div className="space-y-6">
                      <SectionCard
                        title="Memory behavior"
                        description="Controls how dialogue and memory inbox settings behave at runtime."
                        scope="Global"
                      >
                        <div className="grid gap-4 md:grid-cols-2">
                          <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                            <div className="flex items-center justify-between gap-3">
                              <div>
                                <div className="text-sm font-medium text-zinc-100">Auto-save dialogue</div>
                                <p className="mt-1 text-xs text-zinc-400">
                                  Store dialogue in memory automatically instead of keeping it opt-in only.
                                </p>
                              </div>
                              <ToggleSwitch
                                checked={memoryAutoSaveDialogue}
                                label="Toggle auto-save dialogue"
                                onToggle={() => setMemoryAutoSaveDialogue((current) => !current)}
                              />
                            </div>
                          </div>

                          <label className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                            <span className="mb-2 block text-sm font-medium text-zinc-200">Inbox size</span>
                            <span className="mb-2 block text-xs text-zinc-400">Maximum items kept in the inbox.</span>
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
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                            />
                          </label>

                          <label className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                            <span className="mb-2 block text-sm font-medium text-zinc-200">Retention window</span>
                            <span className="mb-2 block text-xs text-zinc-400">How many days inbox entries stay available.</span>
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
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                            />
                          </label>

                          <label className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                            <span className="mb-2 block text-sm font-medium text-zinc-200">Write rate limit</span>
                            <span className="mb-2 block text-xs text-zinc-400">Maximum memory writes allowed per minute.</span>
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
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                            />
                          </label>
                        </div>
                      </SectionCard>

                      <SectionCard
                        title="Advanced indexing"
                        description="Embedding provider and model used for indexing and semantic retrieval."
                        scope="Advanced"
                      >
                        <div className="grid gap-2 md:grid-cols-2">
                          {(['local', 'openai'] as EmbeddingsProvider[]).map((provider) => (
                            <button
                              key={provider}
                              type="button"
                              onClick={() => setEmbeddingsProvider(provider)}
                              className={`rounded-xl border p-4 text-left transition-colors ${
                                embeddingsProvider === provider
                                  ? 'border-zinc-500 bg-zinc-800 text-zinc-100'
                                  : 'border-zinc-800 bg-zinc-950 text-zinc-300 hover:bg-zinc-900'
                              }`}
                            >
                              <div className="text-sm font-medium">
                                {provider === 'local' ? 'Local embeddings' : 'OpenAI embeddings'}
                              </div>
                              <div className="mt-2 text-xs text-zinc-400">
                                {provider === 'local'
                                  ? 'Uses a local sentence-transformer model.'
                                  : 'Uses OPENAI_API_KEY from the environment.'}
                              </div>
                            </button>
                          ))}
                        </div>

                        {embeddingsProvider === 'local' ? (
                          <label className="mt-4 block">
                            <span className="mb-2 block text-sm font-medium text-zinc-300">Local embedding model</span>
                            <input
                              type="text"
                              value={embeddingsLocalModel}
                              onChange={(event) => setEmbeddingsLocalModel(event.target.value)}
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                            />
                          </label>
                        ) : (
                          <label className="mt-4 block">
                            <span className="mb-2 block text-sm font-medium text-zinc-300">OpenAI embedding model</span>
                            <input
                              type="text"
                              value={embeddingsOpenaiModel}
                              onChange={(event) => setEmbeddingsOpenaiModel(event.target.value)}
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                            />
                          </label>
                        )}
                      </SectionCard>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'data' ? (
                    <div className="space-y-6">
                      <SectionCard
                        title="Export chats"
                        description="Download the current chat database as JSON."
                        scope="Global"
                      >
                        <div className="flex items-center justify-between gap-4 rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                          <div className="text-sm text-zinc-300">
                            Export includes session history and message data available to this UI principal.
                          </div>
                          <button
                            type="button"
                            onClick={() => {
                              void handleExportChats();
                            }}
                            disabled={exportingChats}
                            className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm text-zinc-100 transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
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
                          <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300">
                            <div className="font-medium text-zinc-100">{importPreview.fileName}</div>
                            <div className="mt-2 text-xs text-zinc-400">
                              {importPreview.sessionsCount} session{importPreview.sessionsCount === 1 ? '' : 's'} •{' '}
                              {importPreview.messagesCount} message{importPreview.messagesCount === 1 ? '' : 's'}
                            </div>
                          </div>
                        ) : null}
                      </SectionCard>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'diagnostics' ? (
                    <div className="space-y-6">
                      <SectionCard
                        title="Read-only diagnostics"
                        description="Connection and environment status for providers and audio backends."
                        scope="Read only"
                      >
                        <div className="flex items-center justify-between gap-3 rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                          <div className="text-sm text-zinc-300">
                            These values come from environment variables and runtime checks. They cannot be edited in the UI.
                          </div>
                          <button
                            type="button"
                            onClick={() => {
                              void loadSettings();
                            }}
                            className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm text-zinc-100 transition-colors hover:bg-zinc-800"
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
                        <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                          <div className="mb-3 flex items-center justify-between">
                            <h3 className="font-medium text-zinc-100">OpenAI TTS backend</h3>
                            <div className="flex items-center gap-2">
                              <span
                                className={`rounded-md px-2 py-1 text-xs font-medium ${
                                  ttsBackend.backend_ready
                                    ? 'bg-emerald-500/20 text-emerald-300'
                                    : 'bg-amber-500/20 text-amber-300'
                                }`}
                              >
                                {ttsBackend.backend_ready ? 'backend ready' : 'backend not ready'}
                              </span>
                              <span
                                className={`rounded-md px-2 py-1 text-xs font-medium ${
                                  ttsBackend.api_key_set
                                    ? 'bg-emerald-500/20 text-emerald-300'
                                    : 'bg-amber-500/20 text-amber-300'
                                }`}
                              >
                                {ttsBackend.api_key_set ? 'key set' : 'key missing'}
                              </span>
                            </div>
                          </div>
                          <div className="space-y-2 text-xs text-zinc-400">
                            <div>
                              Provider: <span className="text-zinc-300">{PROVIDER_LABELS[ttsBackend.provider]}</span>
                            </div>
                            <div>
                              Env: <span className="font-mono text-zinc-300">{ttsBackend.api_key_env}</span>
                            </div>
                            <div className="break-all">Endpoint: {ttsBackend.endpoint}</div>
                            <div>
                              Model: <span className="font-mono text-zinc-300">{ttsBackend.model}</span>
                            </div>
                            <div>
                              Voice: <span className="font-mono text-zinc-300">{ttsBackend.voice}</span>
                            </div>
                            <div>
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
                        <div className="space-y-4">
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
                            const runtimeClass = providerRuntimeLoading
                              ? 'bg-zinc-800 text-zinc-300'
                              : runtime?.error || providerRuntimeError
                                ? 'bg-rose-500/20 text-rose-300'
                                : runtime && runtime.modelsCount > 0
                                  ? 'bg-emerald-500/20 text-emerald-300'
                                  : 'bg-amber-500/20 text-amber-300';
                            const runtimeDetail = runtime?.error ?? providerRuntimeError;

                            return (
                              <div key={provider.provider} className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                                <div className="mb-3 flex items-center justify-between">
                                  <h3 className="font-medium text-zinc-100">{PROVIDER_LABELS[provider.provider]}</h3>
                                  <div className="flex items-center gap-2">
                                    <span
                                      className={`rounded-md px-2 py-1 text-xs font-medium ${
                                        provider.api_key_set
                                          ? 'bg-emerald-500/20 text-emerald-300'
                                          : 'bg-amber-500/20 text-amber-300'
                                      }`}
                                    >
                                      {provider.api_key_set ? 'key set' : 'key missing'}
                                    </span>
                                    <span className="rounded-md bg-zinc-800 px-2 py-1 text-xs text-zinc-400">
                                      {sourceLabel(provider.api_key_source)}
                                    </span>
                                    <span
                                      className={`rounded-md px-2 py-1 text-xs font-medium ${
                                        provider.api_key_valid === true
                                          ? 'bg-emerald-500/20 text-emerald-300'
                                          : provider.api_key_valid === false
                                            ? 'bg-rose-500/20 text-rose-300'
                                            : 'bg-zinc-800 text-zinc-300'
                                      }`}
                                    >
                                      {provider.api_key_valid === true
                                        ? 'key valid'
                                        : provider.api_key_valid === false
                                          ? 'key invalid'
                                          : 'key unchecked'}
                                    </span>
                                    {isModelProvider(provider.provider) ? (
                                      <span className={`rounded-md px-2 py-1 text-xs font-medium ${runtimeClass}`}>
                                        {runtimeLabel}
                                      </span>
                                    ) : null}
                                  </div>
                                </div>
                                <div className="space-y-2 text-xs text-zinc-400">
                                  <div>
                                    Env: <span className="font-mono text-zinc-300">{provider.api_key_env}</span>
                                  </div>
                                  <div className="break-all">Endpoint: {provider.endpoint}</div>
                                  {provider.last_checked_at ? <div>Last check: {provider.last_checked_at}</div> : null}
                                  {provider.last_check_error ? (
                                    <div className="rounded-md border border-rose-700/40 bg-rose-900/20 px-2 py-1.5 text-rose-200">
                                      Validation: {provider.last_check_error}
                                    </div>
                                  ) : null}
                                  {isModelProvider(provider.provider) && runtimeDetail ? (
                                    <div className="rounded-md border border-rose-700/40 bg-rose-900/20 px-2 py-1.5 text-rose-200">
                                      Runtime: {runtimeDetail}
                                    </div>
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
