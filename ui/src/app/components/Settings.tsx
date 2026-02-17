import { Download, Upload } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useState } from 'react';

interface SettingsProps {
  isOpen: boolean;
  onClose: () => void;
  onSaved?: () => void;
  sessionId?: string | null;
  sessionHeader?: string;
}

type SettingsTab = 'api' | 'personalization' | 'memory' | 'tools' | 'import';
type ApiKeyProvider = 'xai' | 'openrouter' | 'local' | 'openai';
type ModelProvider = 'xai' | 'openrouter' | 'local';
type ApiKeySource = 'settings' | 'env' | 'missing';
type ToolKey = 'fs' | 'shell' | 'web' | 'project' | 'img' | 'tts' | 'stt' | 'safe_mode';
type PolicyProfile = 'sandbox' | 'index' | 'yolo';
type EmbeddingsProvider = 'local' | 'openai';

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

type ProviderRuntimeByModel = Record<ModelProvider, ProviderRuntimeState | null>;

type ParsedSettings = {
  providers: ProviderSettings[];
  apiKeys: Record<ApiKeyProvider, string>;
  toolsState: Record<ToolKey, boolean>;
  policyProfile: PolicyProfile;
  yoloArmed: boolean;
  yoloArmedAt: string | null;
  tone: string;
  systemPrompt: string;
  longPasteToFileEnabled: boolean;
  longPasteThresholdChars: number;
  embeddingsProvider: EmbeddingsProvider;
  embeddingsLocalModel: string;
  embeddingsOpenaiModel: string;
};

type MemoryConflict = {
  atom_id: string;
  stable_key: string;
  claim_type: string;
  summary_text: string;
  confidence: number;
  support_count: number;
  contradict_count: number;
  status: string;
  last_seen_at: string;
};

const DEFAULT_SYSTEM_PROMPT =
  'You are SlavikAI, a helpful AI assistant with MWV architecture.';

const API_KEY_PROVIDERS: ApiKeyProvider[] = ['xai', 'openrouter', 'local', 'openai'];
const MODEL_PROVIDERS: ModelProvider[] = ['xai', 'openrouter', 'local'];
const TOOL_TOGGLE_KEYS: ToolKey[] = ['web', 'fs', 'project', 'shell', 'tts', 'stt', 'img', 'safe_mode'];
const SAFE_MODE_BLOCKED_TOOLS = new Set<ToolKey>(['web', 'shell', 'project', 'tts', 'stt']);
const TOOL_LABELS: Record<ToolKey, string> = {
  fs: 'Filesystem',
  shell: 'Shell',
  web: 'Web',
  project: 'Project',
  img: 'Images',
  tts: 'Text to speech',
  stt: 'Speech to text',
  safe_mode: 'Safe mode',
};

const PROVIDER_LABELS: Record<ApiKeyProvider, string> = {
  xai: 'xAI',
  openrouter: 'OpenRouter',
  local: 'Local',
  openai: 'OpenAI',
};

const DEFAULT_API_KEYS: Record<ApiKeyProvider, string> = {
  xai: '',
  openrouter: '',
  local: '',
  openai: '',
};
const DEFAULT_LONG_PASTE_TO_FILE_ENABLED = true;
const DEFAULT_LONG_PASTE_THRESHOLD_CHARS = 12000;
const DEFAULT_POLICY_PROFILE: PolicyProfile = 'sandbox';
const DEFAULT_EMBEDDINGS_PROVIDER: EmbeddingsProvider = 'local';
const DEFAULT_EMBEDDINGS_LOCAL_MODEL = 'all-MiniLM-L6-v2';
const DEFAULT_EMBEDDINGS_OPENAI_MODEL = 'text-embedding-3-small';

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
];
const DEFAULT_TOOLS_STATE: Record<ToolKey, boolean> = {
  fs: true,
  shell: false,
  web: false,
  project: true,
  img: false,
  tts: false,
  stt: false,
  safe_mode: true,
};

const isApiKeyProvider = (value: unknown): value is ApiKeyProvider =>
  value === 'xai' || value === 'openrouter' || value === 'local' || value === 'openai';

const isApiKeySource = (value: unknown): value is ApiKeySource =>
  value === 'settings' || value === 'env' || value === 'missing';

const isModelProvider = (value: unknown): value is ModelProvider =>
  value === 'xai' || value === 'openrouter' || value === 'local';

const isToolKey = (value: unknown): value is ToolKey =>
  value === 'fs'
  || value === 'shell'
  || value === 'web'
  || value === 'project'
  || value === 'img'
  || value === 'tts'
  || value === 'stt'
  || value === 'safe_mode';

const isPolicyProfile = (value: unknown): value is PolicyProfile =>
  value === 'sandbox' || value === 'index' || value === 'yolo';

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
    apiKeys: DEFAULT_API_KEYS,
    toolsState: DEFAULT_TOOLS_STATE,
    policyProfile: DEFAULT_POLICY_PROFILE,
    yoloArmed: false,
    yoloArmedAt: null,
    tone: 'balanced',
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
    longPasteToFileEnabled: DEFAULT_LONG_PASTE_TO_FILE_ENABLED,
    longPasteThresholdChars: DEFAULT_LONG_PASTE_THRESHOLD_CHARS,
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

  const personalization = (settings as { personalization?: unknown }).personalization;
  let tone = defaults.tone;
  let systemPrompt = defaults.systemPrompt;
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
  const composer = (settings as { composer?: unknown }).composer;
  let longPasteToFileEnabled = defaults.longPasteToFileEnabled;
  let longPasteThresholdChars = defaults.longPasteThresholdChars;
  if (composer && typeof composer === 'object') {
    const enabledRaw = (composer as { long_paste_to_file_enabled?: unknown }).long_paste_to_file_enabled;
    const thresholdRaw = (composer as { long_paste_threshold_chars?: unknown }).long_paste_threshold_chars;
    if (typeof enabledRaw === 'boolean') {
      longPasteToFileEnabled = enabledRaw;
    }
    if (
      typeof thresholdRaw === 'number'
      && Number.isFinite(thresholdRaw)
      && thresholdRaw > 0
    ) {
      longPasteThresholdChars = Math.floor(thresholdRaw);
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
      const apiKeyEnv = (item as { api_key_env?: unknown }).api_key_env;
      const endpoint = (item as { endpoint?: unknown }).endpoint;
      const apiKeySet = (item as { api_key_set?: unknown }).api_key_set;
      const sourceRaw = (item as { api_key_source?: unknown }).api_key_source;
      const apiKeyValid = (item as { api_key_valid?: unknown }).api_key_valid;
      const lastCheckError = (item as { last_check_error?: unknown }).last_check_error;
      const lastCheckedAt = (item as { last_checked_at?: unknown }).last_checked_at;
      const current = providersMap.get(providerRaw);
      providersMap.set(providerRaw, {
        provider: providerRaw,
        api_key_env: typeof apiKeyEnv === 'string' && apiKeyEnv.trim() ? apiKeyEnv : current?.api_key_env || '',
        endpoint: typeof endpoint === 'string' && endpoint.trim() ? endpoint : current?.endpoint || '',
        api_key_set: typeof apiKeySet === 'boolean' ? apiKeySet : current?.api_key_set || false,
        api_key_source: isApiKeySource(sourceRaw)
          ? sourceRaw
          : current?.api_key_source || 'missing',
        api_key_valid:
          typeof apiKeyValid === 'boolean'
            ? apiKeyValid
            : current?.api_key_valid ?? null,
        last_check_error:
          typeof lastCheckError === 'string'
            ? lastCheckError
            : current?.last_check_error ?? null,
        last_checked_at:
          typeof lastCheckedAt === 'string'
            ? lastCheckedAt
            : current?.last_checked_at ?? null,
      });
    }
  }

  const providers = API_KEY_PROVIDERS.map(
    (provider) =>
      providersMap.get(provider) ||
      DEFAULT_PROVIDER_SETTINGS.find((item) => item.provider === provider) ||
      DEFAULT_PROVIDER_SETTINGS[0],
  );
  const parsedApiKeys: Record<ApiKeyProvider, string> = {
    xai: '',
    openrouter: '',
    local: '',
    openai: '',
  };
  const tools = (settings as { tools?: unknown }).tools;
  const policy = (settings as { policy?: unknown }).policy;
  const memory = (settings as { memory?: unknown }).memory;
  let policyProfile: PolicyProfile = defaults.policyProfile;
  let yoloArmed = defaults.yoloArmed;
  let yoloArmedAt = defaults.yoloArmedAt;
  let embeddingsProvider = defaults.embeddingsProvider;
  let embeddingsLocalModel = defaults.embeddingsLocalModel;
  let embeddingsOpenaiModel = defaults.embeddingsOpenaiModel;
  let toolsState: Record<ToolKey, boolean> = { ...defaults.toolsState };
  if (tools && typeof tools === 'object') {
    const stateRaw = (tools as { state?: unknown }).state;
    if (stateRaw && typeof stateRaw === 'object') {
      const nextToolsState: Record<ToolKey, boolean> = { ...defaults.toolsState };
      for (const [key, value] of Object.entries(stateRaw as Record<string, unknown>)) {
        if (!isToolKey(key) || typeof value !== 'boolean') {
          continue;
        }
        nextToolsState[key] = value;
      }
      toolsState = nextToolsState;
    }
  }
  if (policy && typeof policy === 'object') {
    const profileRaw = (policy as { profile?: unknown }).profile;
    const yoloArmedRaw = (policy as { yolo_armed?: unknown }).yolo_armed;
    const yoloArmedAtRaw = (policy as { yolo_armed_at?: unknown }).yolo_armed_at;
    if (isPolicyProfile(profileRaw)) {
      policyProfile = profileRaw;
    }
    if (typeof yoloArmedRaw === 'boolean') {
      yoloArmed = yoloArmedRaw;
    }
    if (typeof yoloArmedAtRaw === 'string' && yoloArmedAtRaw.trim()) {
      yoloArmedAt = yoloArmedAtRaw.trim();
    }
  }

  if (memory && typeof memory === 'object') {
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
    } else {
      // legacy fallback for older payloads
      const legacyRaw = (memory as { embeddings_model?: unknown }).embeddings_model;
      if (typeof legacyRaw === 'string' && legacyRaw.trim()) {
        embeddingsProvider = 'local';
        embeddingsLocalModel = legacyRaw.trim();
      }
    }
  }

  return {
    providers,
    apiKeys: parsedApiKeys,
    toolsState,
    policyProfile,
    yoloArmed,
    yoloArmedAt,
    tone,
    systemPrompt,
    longPasteToFileEnabled,
    longPasteThresholdChars,
    embeddingsProvider,
    embeddingsLocalModel,
    embeddingsOpenaiModel,
  };
};

const DEFAULT_PROVIDER_RUNTIME: ProviderRuntimeByModel = {
  xai: null,
  openrouter: null,
  local: null,
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
      typeof errorRaw === 'string' && errorRaw.trim().length > 0
        ? errorRaw.trim()
        : null;
    result[providerRaw] = { modelsCount, error };
  }
  return result;
};

const parseMemoryConflictsPayload = (payload: unknown): MemoryConflict[] => {
  if (!payload || typeof payload !== 'object') {
    return [];
  }
  const conflictsRaw = (payload as { conflicts?: unknown }).conflicts;
  if (!Array.isArray(conflictsRaw)) {
    return [];
  }
  const conflicts: MemoryConflict[] = [];
  for (const item of conflictsRaw) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const row = item as Record<string, unknown>;
    const stableKey = row.stable_key;
    if (typeof stableKey !== 'string' || !stableKey.trim()) {
      continue;
    }
    conflicts.push({
      atom_id: typeof row.atom_id === 'string' ? row.atom_id : stableKey,
      stable_key: stableKey,
      claim_type: typeof row.claim_type === 'string' ? row.claim_type : 'unknown',
      summary_text: typeof row.summary_text === 'string' ? row.summary_text : stableKey,
      confidence: typeof row.confidence === 'number' ? row.confidence : 0,
      support_count: typeof row.support_count === 'number' ? row.support_count : 0,
      contradict_count: typeof row.contradict_count === 'number' ? row.contradict_count : 0,
      status: typeof row.status === 'string' ? row.status : 'conflict',
      last_seen_at: typeof row.last_seen_at === 'string' ? row.last_seen_at : '',
    });
  }
  return conflicts;
};

const providerPlaceholder = (provider: ProviderSettings): string => {
  if (provider.api_key_source === 'settings') {
    return `Stored in settings. Paste new ${provider.provider} key or leave empty and Save to clear.`;
  }
  if (provider.api_key_source === 'env') {
    return `Using ${provider.api_key_env}. Paste key here to override in UI.`;
  }
  return `Paste ${provider.provider} API key`;
};

const sourceLabel = (source: ApiKeySource): string => {
  if (source === 'settings') {
    return 'settings';
  }
  if (source === 'env') {
    return 'env';
  }
  return 'missing';
};

const isToolEffectivelyEnabled = (tool: ToolKey, state: Record<ToolKey, boolean>): boolean => {
  if (tool === 'safe_mode') {
    return state.safe_mode;
  }
  if (!state[tool]) {
    return false;
  }
  if (state.safe_mode && SAFE_MODE_BLOCKED_TOOLS.has(tool)) {
    return false;
  }
  return true;
};

const toolStatusLabel = (tool: ToolKey, state: Record<ToolKey, boolean>): string => {
  if (tool === 'safe_mode') {
    return state.safe_mode ? 'enabled' : 'disabled';
  }
  if (!state[tool]) {
    return 'disabled in settings';
  }
  if (state.safe_mode && SAFE_MODE_BLOCKED_TOOLS.has(tool)) {
    return 'blocked by safe mode';
  }
  return 'enabled';
};

const policyProfileLabel = (profile: PolicyProfile): string => {
  if (profile === 'index') {
    return 'Index';
  }
  if (profile === 'yolo') {
    return 'YOLO (Danger)';
  }
  return 'Sandbox';
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

export function Settings({
  isOpen,
  onClose,
  onSaved,
  sessionId = null,
  sessionHeader = 'X-Slavik-Session',
}: SettingsProps) {
  const EMPTY_PROVIDER_DIRTY: Record<ApiKeyProvider, boolean> = {
    xai: false,
    openrouter: false,
    local: false,
    openai: false,
  };
  const [activeTab, setActiveTab] = useState<SettingsTab>('api');
  const [selectedProvider, setSelectedProvider] = useState<ModelProvider>('local');
  const [apiKeys, setApiKeys] = useState<Record<ApiKeyProvider, string>>(DEFAULT_API_KEYS);
  const [providerDirty, setProviderDirty] = useState<Record<ApiKeyProvider, boolean>>(
    EMPTY_PROVIDER_DIRTY,
  );
  const [toolsState, setToolsState] = useState<Record<ToolKey, boolean>>(DEFAULT_TOOLS_STATE);
  const [policyProfile, setPolicyProfile] = useState<PolicyProfile>(DEFAULT_POLICY_PROFILE);
  const [yoloArmed, setYoloArmed] = useState(false);
  const [yoloArmedAt, setYoloArmedAt] = useState<string | null>(null);
  const [yoloConfirmText, setYoloConfirmText] = useState('');
  const [yoloSecondConfirm, setYoloSecondConfirm] = useState(false);
  const [providers, setProviders] = useState<ProviderSettings[]>(DEFAULT_PROVIDER_SETTINGS);
  const [providerRuntime, setProviderRuntime] = useState<ProviderRuntimeByModel>(
    DEFAULT_PROVIDER_RUNTIME,
  );
  const [providerRuntimeLoading, setProviderRuntimeLoading] = useState(false);
  const [providerRuntimeError, setProviderRuntimeError] = useState<string | null>(null);
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [tone, setTone] = useState('balanced');
  const [longPasteToFileEnabled, setLongPasteToFileEnabled] = useState(
    DEFAULT_LONG_PASTE_TO_FILE_ENABLED,
  );
  const [longPasteThresholdChars, setLongPasteThresholdChars] = useState(
    DEFAULT_LONG_PASTE_THRESHOLD_CHARS,
  );
  const [embeddingsProvider, setEmbeddingsProvider] = useState<EmbeddingsProvider>(
    DEFAULT_EMBEDDINGS_PROVIDER,
  );
  const [embeddingsLocalModel, setEmbeddingsLocalModel] = useState(DEFAULT_EMBEDDINGS_LOCAL_MODEL);
  const [embeddingsOpenaiModel, setEmbeddingsOpenaiModel] = useState(
    DEFAULT_EMBEDDINGS_OPENAI_MODEL,
  );
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savingTools, setSavingTools] = useState(false);
  const [savingPolicy, setSavingPolicy] = useState(false);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryResolvingKey, setMemoryResolvingKey] = useState<string | null>(null);
  const [memoryConflicts, setMemoryConflicts] = useState<MemoryConflict[]>([]);
  const [status, setStatus] = useState<string | null>(null);

  const requestHeaders = sessionId ? { [sessionHeader]: sessionId } : {};

  const applyParsedSettings = (parsed: ParsedSettings): void => {
    setProviders(parsed.providers);
    setApiKeys(parsed.apiKeys);
    setProviderDirty(EMPTY_PROVIDER_DIRTY);
    setToolsState(parsed.toolsState);
    setPolicyProfile(parsed.policyProfile);
    setYoloArmed(parsed.yoloArmed);
    setYoloArmedAt(parsed.yoloArmedAt);
    setSystemPrompt(parsed.systemPrompt);
    setTone(parsed.tone);
    setLongPasteToFileEnabled(parsed.longPasteToFileEnabled);
    setLongPasteThresholdChars(parsed.longPasteThresholdChars);
    setEmbeddingsProvider(parsed.embeddingsProvider);
    setEmbeddingsLocalModel(parsed.embeddingsLocalModel);
    setEmbeddingsOpenaiModel(parsed.embeddingsOpenaiModel);
  };

  const loadSettings = async () => {
    setLoading(true);
    setStatus(null);
    setProviderRuntimeLoading(true);
    setProviderRuntimeError(null);
    try {
      const requests: Promise<Response>[] = [
        fetch('/ui/api/settings', { headers: requestHeaders }),
        fetch('/ui/api/models', { headers: requestHeaders }),
      ];
      if (sessionId) {
        requests.push(fetch('/ui/api/workspace/root', { headers: requestHeaders }));
      }
      const [settingsResponse, providerRuntimeResponse, sessionRootResponse] = await Promise.all(requests);
      const settingsPayload: unknown = await settingsResponse.json();
      if (!settingsResponse.ok) {
        throw new Error(extractErrorMessage(settingsPayload, 'Failed to load settings.'));
      }
      const parsed = parseSettingsPayload(settingsPayload);
      applyParsedSettings(parsed);
      if (sessionId) {
        if (!sessionRootResponse) {
          throw new Error('Failed to load session policy.');
        }
        const rootPayload: unknown = await sessionRootResponse.json();
        if (!sessionRootResponse.ok) {
          throw new Error(extractErrorMessage(rootPayload, 'Failed to load session policy.'));
        }
        const policyRaw = (rootPayload as { policy?: unknown }).policy;
        if (policyRaw && typeof policyRaw === 'object') {
          const profileRaw = (policyRaw as { profile?: unknown }).profile;
          const yoloArmedRaw = (policyRaw as { yolo_armed?: unknown }).yolo_armed;
          const yoloArmedAtRaw = (policyRaw as { yolo_armed_at?: unknown }).yolo_armed_at;
          if (isPolicyProfile(profileRaw)) {
            setPolicyProfile(profileRaw);
          }
          setYoloArmed(yoloArmedRaw === true);
          setYoloArmedAt(
            typeof yoloArmedAtRaw === 'string' && yoloArmedAtRaw.trim() ? yoloArmedAtRaw.trim() : null,
          );
        }
      }
      const providerRuntimePayload: unknown = await providerRuntimeResponse.json();
      if (!providerRuntimeResponse.ok) {
        throw new Error(extractErrorMessage(providerRuntimePayload, 'Failed to load provider runtime status.'));
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

  const loadMemoryConflicts = async () => {
    setMemoryLoading(true);
    try {
      const response = await fetch('/ui/api/memory/conflicts?limit=100', {
        headers: requestHeaders,
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to load memory conflicts.'));
      }
      setMemoryConflicts(parseMemoryConflictsPayload(payload));
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to load memory conflicts.';
      setStatus(message);
    } finally {
      setMemoryLoading(false);
    }
  };

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    void loadSettings();
  }, [isOpen, sessionId]);

  useEffect(() => {
    if (!isOpen || activeTab !== 'memory') {
      return;
    }
    void loadMemoryConflicts();
  }, [activeTab, isOpen]);

  const handleSave = async () => {
    if (saving || savingPolicy) {
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
        const key = apiKeys[provider].trim();
        if (key) {
          providersPayload[provider] = { api_key: key };
        } else {
          providersPayload[provider] = null;
        }
        hasProviderChanges = true;
      }

      const payload: Record<string, unknown> = {
        personalization: {
          tone: tone.trim() || 'balanced',
          system_prompt: systemPrompt,
        },
        composer: {
          long_paste_to_file_enabled: longPasteToFileEnabled,
          long_paste_threshold_chars: Math.max(1000, Math.min(80000, longPasteThresholdChars)),
        },
        memory: {
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
          ...requestHeaders,
        },
        body: JSON.stringify(payload),
      });
      const body: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(body, 'Failed to save settings.'));
      }
      const parsed = parseSettingsPayload(body);
      applyParsedSettings(parsed);
      setStatus('Saved');
      onSaved?.();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save settings.';
      setStatus(message);
    } finally {
      setSaving(false);
    }
  };

  const handleToolToggle = async (tool: ToolKey) => {
    if (loading || saving || savingTools || savingPolicy) {
      return;
    }
    const previousState = toolsState;
    const nextValue = !previousState[tool];
    const nextState: Record<ToolKey, boolean> = {
      ...previousState,
      [tool]: nextValue,
    };
    setToolsState(nextState);
    setSavingTools(true);
    setStatus(null);
    try {
      const response = await fetch('/ui/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...requestHeaders,
        },
        body: JSON.stringify({
          tools: {
            state: {
              [tool]: nextValue,
            },
          },
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to update tools settings.'));
      }
      const parsed = parseSettingsPayload(payload);
      applyParsedSettings(parsed);
      setStatus(`Tool "${TOOL_LABELS[tool]}" updated.`);
      onSaved?.();
    } catch (error) {
      setToolsState(previousState);
      const message =
        error instanceof Error ? error.message : 'Failed to update tools settings.';
      setStatus(message);
    } finally {
      setSavingTools(false);
    }
  };

  const handleResolveMemoryConflict = async (
    stableKey: string,
    action: 'activate' | 'deprecate',
  ) => {
    if (memoryResolvingKey) {
      return;
    }
    setMemoryResolvingKey(stableKey);
    setStatus(null);
    try {
      const response = await fetch('/ui/api/memory/conflicts/resolve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...requestHeaders,
        },
        body: JSON.stringify({
          stable_key: stableKey,
          action,
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to resolve memory conflict.'));
      }
      setStatus(`Conflict ${action === 'activate' ? 'activated' : 'deprecated'}: ${stableKey}`);
      await loadMemoryConflicts();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to resolve memory conflict.';
      setStatus(message);
    } finally {
      setMemoryResolvingKey(null);
    }
  };

  const handleApplyPolicy = async () => {
    if (loading || saving || savingTools || savingPolicy) {
      return;
    }
    if (!sessionId) {
      setStatus('Выберите активную сессию для применения session policy.');
      return;
    }
    const wantsYolo = policyProfile === 'yolo';
    if (wantsYolo && yoloConfirmText.trim().toUpperCase() !== 'YOLO') {
      setStatus('Для YOLO введите подтверждение: YOLO');
      return;
    }
    if (wantsYolo && !yoloSecondConfirm) {
      setStatus('Подтвердите, что понимаете риск YOLO режима.');
      return;
    }
    setSavingPolicy(true);
    setStatus(null);
    try {
      const response = await fetch('/ui/api/session/policy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...requestHeaders,
        },
        body: JSON.stringify({
          policy_profile: policyProfile,
          confirm_yolo: wantsYolo ? true : undefined,
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to apply policy profile.'));
      }
      const policy = (payload as { policy?: unknown }).policy;
      if (!policy || typeof policy !== 'object') {
        throw new Error('Invalid policy response.');
      }
      const profileRaw = (policy as { profile?: unknown }).profile;
      const yoloArmedRaw = (policy as { yolo_armed?: unknown }).yolo_armed;
      const yoloArmedAtRaw = (policy as { yolo_armed_at?: unknown }).yolo_armed_at;
      if (!isPolicyProfile(profileRaw)) {
        throw new Error('Invalid policy profile in response.');
      }
      setPolicyProfile(profileRaw);
      setYoloArmed(yoloArmedRaw === true);
      setYoloArmedAt(
        typeof yoloArmedAtRaw === 'string' && yoloArmedAtRaw.trim() ? yoloArmedAtRaw.trim() : null,
      );
      if (!wantsYolo) {
        setYoloConfirmText('');
        setYoloSecondConfirm(false);
      }
      setStatus(`Policy profile updated: ${policyProfileLabel(profileRaw)}.`);
      onSaved?.();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to apply policy profile.';
      setStatus(message);
    } finally {
      setSavingPolicy(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
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
            <div className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl shadow-black/60">
              <div className="flex items-center justify-between border-b border-zinc-800 p-6">
                <div>
                  <div className="mb-1 text-xs tracking-[0.18em] text-zinc-500">SETTINGS</div>
                  <h2 className="text-xl font-semibold text-zinc-100">Workspace controls</h2>
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
                    disabled={saving || loading || savingTools || savingPolicy}
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
                    {[
                      { id: 'api', title: 'API Keys / Providers' },
                      { id: 'personalization', title: 'Personalization' },
                      { id: 'memory', title: 'Memory' },
                      { id: 'tools', title: 'Tools' },
                      { id: 'import', title: 'Import / Export chats DB' },
                    ].map((tab) => (
                      <button
                        key={tab.id}
                        type="button"
                        onClick={() => setActiveTab(tab.id as SettingsTab)}
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

                  {!loading && activeTab === 'api' ? (
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
                        <div key={provider.provider} className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
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
                            {provider.last_checked_at ? (
                              <div>Last check: {provider.last_checked_at}</div>
                            ) : null}
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
                          <input
                            type="password"
                            value={apiKeys[provider.provider]}
                            onChange={(event) => {
                              setApiKeys((prev) => ({ ...prev, [provider.provider]: event.target.value }));
                              setProviderDirty((prev) => ({ ...prev, [provider.provider]: true }));
                            }}
                            placeholder={providerPlaceholder(provider)}
                            className="mt-3 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
                          />
                        </div>
                        );
                      })}
                    </div>
                  ) : null}

                  {!loading && activeTab === 'personalization' ? (
                    <div className="space-y-6">
                      <div>
                        <label className="mb-2 block text-sm font-medium text-zinc-300">System Prompt</label>
                        <textarea
                          value={systemPrompt}
                          onChange={(event) => setSystemPrompt(event.target.value)}
                          rows={6}
                          className="w-full resize-none rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
                          placeholder="Enter system prompt..."
                        />
                      </div>

                      <div>
                        <label className="mb-2 block text-sm font-medium text-zinc-300">Tone</label>
                        <select
                          value={tone}
                          onChange={(event) => setTone(event.target.value)}
                          className="w-full rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                        >
                          <option value="balanced">Balanced</option>
                          <option value="professional">Professional</option>
                          <option value="casual">Casual</option>
                          <option value="technical">Technical</option>
                          <option value="friendly">Friendly</option>
                        </select>
                      </div>

                      <div className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium text-zinc-200">
                              Convert long paste to file
                            </div>
                            <p className="mt-1 text-xs text-zinc-400">
                              Large paste will be converted to a virtual file attachment in composer.
                            </p>
                          </div>
                          <ToggleSwitch
                            checked={longPasteToFileEnabled}
                            label="Convert long paste to file"
                            onToggle={() => setLongPasteToFileEnabled((prev) => !prev)}
                          />
                        </div>

                        <div>
                          <label className="mb-1 block text-xs font-medium text-zinc-300">
                            Long paste threshold (chars)
                          </label>
                          <input
                            type="number"
                            min={1000}
                            max={80000}
                            value={longPasteThresholdChars}
                            onChange={(event) => {
                              const next = Number.parseInt(event.target.value, 10);
                              if (Number.isNaN(next)) {
                                return;
                              }
                              setLongPasteThresholdChars(next);
                            }}
                            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="mb-2 block text-sm font-medium text-zinc-300">Model Provider</label>
                        <div className="grid grid-cols-3 gap-2">
                          {MODEL_PROVIDERS.map((provider) => (
                            <button
                              key={provider}
                              type="button"
                              onClick={() => setSelectedProvider(provider)}
                              className={`rounded-lg border px-4 py-3 transition-colors ${
                                selectedProvider === provider
                                  ? 'border-zinc-500 bg-zinc-800 text-zinc-100'
                                  : 'border-zinc-700 bg-zinc-900 text-zinc-400 hover:bg-zinc-800'
                              }`}
                            >
                              {PROVIDER_LABELS[provider]}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'memory' ? (
                    <div className="space-y-4">
                      <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 space-y-3">
                        <div>
                          <h3 className="text-sm font-medium text-zinc-100">Embeddings provider</h3>
                          <p className="mt-1 text-xs text-zinc-400">
                            Для индексации и semantic search. Ключ берется из OpenAI provider settings.
                          </p>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          {(['local', 'openai'] as EmbeddingsProvider[]).map((provider) => (
                            <button
                              key={provider}
                              type="button"
                              onClick={() => setEmbeddingsProvider(provider)}
                              className={`rounded-lg border px-4 py-2 text-sm transition-colors ${
                                embeddingsProvider === provider
                                  ? 'border-zinc-500 bg-zinc-800 text-zinc-100'
                                  : 'border-zinc-700 bg-zinc-900 text-zinc-400 hover:bg-zinc-800'
                              }`}
                            >
                              {provider === 'local' ? 'Local (SentenceTransformer)' : 'OpenAI'}
                            </button>
                          ))}
                        </div>
                        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                          <label className="space-y-2">
                            <span className="block text-xs text-zinc-400">local_model</span>
                            <input
                              type="text"
                              value={embeddingsLocalModel}
                              onChange={(event) => setEmbeddingsLocalModel(event.target.value)}
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
                            />
                          </label>
                          <label className="space-y-2">
                            <span className="block text-xs text-zinc-400">openai_model</span>
                            <input
                              type="text"
                              value={embeddingsOpenaiModel}
                              onChange={(event) => setEmbeddingsOpenaiModel(event.target.value)}
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
                            />
                          </label>
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <p className="text-sm text-zinc-400">
                          Conflict atoms require manual resolve. Choose activate or deprecate.
                        </p>
                        <button
                          type="button"
                          onClick={() => {
                            void loadMemoryConflicts();
                          }}
                          className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-800"
                        >
                          Refresh conflicts
                        </button>
                      </div>
                      {memoryLoading ? (
                        <div className="text-sm text-zinc-400">Loading memory conflicts...</div>
                      ) : null}
                      {!memoryLoading && memoryConflicts.length === 0 ? (
                        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4 text-sm text-zinc-400">
                          No memory conflicts.
                        </div>
                      ) : null}
                      {!memoryLoading && memoryConflicts.length > 0 ? (
                        <div className="space-y-3">
                          {memoryConflicts.map((conflict) => (
                            <div
                              key={conflict.atom_id}
                              className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4"
                            >
                              <div className="text-sm font-medium text-zinc-100">
                                {conflict.stable_key}
                              </div>
                              <div className="mt-1 text-xs text-zinc-400">
                                type={conflict.claim_type} | confidence={conflict.confidence.toFixed(2)} | support=
                                {conflict.support_count} | contradict={conflict.contradict_count}
                              </div>
                              <div className="mt-2 text-xs text-zinc-300">
                                {conflict.summary_text}
                              </div>
                              <div className="mt-3 flex items-center gap-2">
                                <button
                                  type="button"
                                  disabled={memoryResolvingKey === conflict.stable_key}
                                  onClick={() => {
                                    void handleResolveMemoryConflict(conflict.stable_key, 'activate');
                                  }}
                                  className="rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-300 transition-colors hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-40"
                                >
                                  Activate
                                </button>
                                <button
                                  type="button"
                                  disabled={memoryResolvingKey === conflict.stable_key}
                                  onClick={() => {
                                    void handleResolveMemoryConflict(conflict.stable_key, 'deprecate');
                                  }}
                                  className="rounded-md border border-zinc-600 bg-zinc-800 px-3 py-1 text-xs text-zinc-300 transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
                                >
                                  Deprecate
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  {!loading && activeTab === 'tools' ? (
                    <div className="space-y-4">
                      <p className="text-sm text-zinc-400">
                        Tools are applied live to the active agent. Safe mode can still block risky tools.
                      </p>
                      <section className="space-y-3 border border-zinc-800 p-4">
                        <div>
                          <h3 className="text-sm font-medium text-zinc-100">Session policy profile</h3>
                          <p className="mt-1 text-xs text-zinc-400">
                            Определяет базовые ограничения доступа: Sandbox / Index / YOLO.
                          </p>
                        </div>
                        <div className="grid grid-cols-[1fr_auto] gap-3 items-center">
                          <select
                            value={policyProfile}
                            onChange={(event) => setPolicyProfile(event.target.value as PolicyProfile)}
                            className="rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none"
                          >
                            <option value="sandbox">Sandbox</option>
                            <option value="index">Index</option>
                            <option value="yolo">YOLO (Danger)</option>
                          </select>
                          <button
                            type="button"
                            onClick={() => {
                              void handleApplyPolicy();
                            }}
                            disabled={savingPolicy || saving || loading || savingTools || !sessionId}
                            className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {savingPolicy ? 'Applying...' : 'Apply policy'}
                          </button>
                        </div>
                        {policyProfile === 'yolo' ? (
                          <div className="space-y-3 border border-red-500/50 bg-red-950/20 p-3">
                            <div className="text-xs font-medium text-red-300">
                              YOLO Danger Zone: one-shot override, все действия логируются.
                            </div>
                            <label className="block text-xs text-red-200">
                              Введите <span className="font-mono">YOLO</span> для подтверждения
                            </label>
                            <input
                              type="text"
                              value={yoloConfirmText}
                              onChange={(event) => setYoloConfirmText(event.target.value)}
                              className="w-full rounded-md border border-red-500/40 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-red-400 focus:outline-none"
                              placeholder="YOLO"
                            />
                            <label className="flex items-center gap-2 text-xs text-red-200">
                              <input
                                type="checkbox"
                                checked={yoloSecondConfirm}
                                onChange={(event) => setYoloSecondConfirm(event.target.checked)}
                              />
                              Я понимаю риск выполнения в YOLO режиме.
                            </label>
                          </div>
                        ) : null}
                        <div className="text-xs text-zinc-400">
                          Текущий профиль: {policyProfileLabel(policyProfile)}
                          {yoloArmed ? ' | YOLO active' : ''}
                          {yoloArmedAt ? ` | armed at ${yoloArmedAt}` : ''}
                        </div>
                      </section>

                      <div className="divide-y divide-zinc-800 border border-zinc-800">
                        {TOOL_TOGGLE_KEYS.map((tool) => {
                          const checked = toolsState[tool];
                          const effectiveEnabled = isToolEffectivelyEnabled(tool, toolsState);
                          const statusLabel = toolStatusLabel(tool, toolsState);
                          return (
                            <div
                              key={tool}
                              className="flex items-center justify-between gap-4 px-4 py-3"
                              title={statusLabel}
                            >
                              <div>
                                <div className="text-sm font-medium text-zinc-100">{TOOL_LABELS[tool]}</div>
                                <div className="mt-1 text-xs text-zinc-400">
                                  Requested: {checked ? 'on' : 'off'} | Effective: {effectiveEnabled ? 'on' : 'off'} ({statusLabel})
                                </div>
                              </div>
                              <ToggleSwitch
                                checked={checked}
                                label={`Toggle ${TOOL_LABELS[tool]}`}
                                disabled={savingTools || saving || loading || savingPolicy}
                                onToggle={() => {
                                  void handleToolToggle(tool);
                                }}
                              />
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'import' ? (
                    <div className="space-y-4">
                      <button
                        type="button"
                        className="flex w-full items-center justify-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-3 text-zinc-300 transition-colors hover:bg-zinc-800"
                      >
                        <Upload className="h-4 w-4" />
                        Import Chats Database
                      </button>
                      <button
                        type="button"
                        className="flex w-full items-center justify-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-3 text-zinc-300 transition-colors hover:bg-zinc-800"
                      >
                        <Download className="h-4 w-4" />
                        Export Chats Database
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
