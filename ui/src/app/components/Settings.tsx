import { Download, Upload } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useState } from 'react';

interface SettingsProps {
  isOpen: boolean;
  onClose: () => void;
  onSaved?: () => void;
}

type SettingsTab = 'api' | 'personalization' | 'memory' | 'tools' | 'import';
type ApiKeyProvider = 'xai' | 'openrouter' | 'local' | 'openai';
type ModelProvider = 'xai' | 'openrouter' | 'local';
type ApiKeySource = 'settings' | 'env' | 'missing';

type ProviderSettings = {
  provider: ApiKeyProvider;
  api_key_env: string;
  api_key_set: boolean;
  api_key_source: ApiKeySource;
  api_key_value: string;
  endpoint: string;
};

type ParsedSettings = {
  providers: ProviderSettings[];
  apiKeys: Record<ApiKeyProvider, string>;
  tone: string;
  systemPrompt: string;
  longPasteToFileEnabled: boolean;
  longPasteThresholdChars: number;
};

const DEFAULT_SYSTEM_PROMPT =
  'You are SlavikAI, a helpful AI assistant with MWV architecture.';

const API_KEY_PROVIDERS: ApiKeyProvider[] = ['xai', 'openrouter', 'local', 'openai'];
const MODEL_PROVIDERS: ModelProvider[] = ['xai', 'openrouter', 'local'];

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

const DEFAULT_PROVIDER_SETTINGS: ProviderSettings[] = [
  {
    provider: 'xai',
    api_key_env: 'XAI_API_KEY',
    api_key_set: false,
    api_key_source: 'missing',
    api_key_value: '',
    endpoint: 'https://api.x.ai/v1/models',
  },
  {
    provider: 'openrouter',
    api_key_env: 'OPENROUTER_API_KEY',
    api_key_set: false,
    api_key_source: 'missing',
    api_key_value: '',
    endpoint: 'https://openrouter.ai/api/v1/models',
  },
  {
    provider: 'local',
    api_key_env: 'LOCAL_LLM_API_KEY',
    api_key_set: false,
    api_key_source: 'missing',
    api_key_value: '',
    endpoint: 'http://localhost:11434/v1/models',
  },
  {
    provider: 'openai',
    api_key_env: 'OPENAI_API_KEY',
    api_key_set: false,
    api_key_source: 'missing',
    api_key_value: '',
    endpoint: 'https://api.openai.com/v1/audio/transcriptions',
  },
];

const isApiKeyProvider = (value: unknown): value is ApiKeyProvider =>
  value === 'xai' || value === 'openrouter' || value === 'local' || value === 'openai';

const isApiKeySource = (value: unknown): value is ApiKeySource =>
  value === 'settings' || value === 'env' || value === 'missing';

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
    tone: 'balanced',
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
    longPasteToFileEnabled: DEFAULT_LONG_PASTE_TO_FILE_ENABLED,
    longPasteThresholdChars: DEFAULT_LONG_PASTE_THRESHOLD_CHARS,
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
      const apiKeyValueRaw = (item as { api_key_value?: unknown }).api_key_value;
      const current = providersMap.get(providerRaw);
      providersMap.set(providerRaw, {
        provider: providerRaw,
        api_key_env: typeof apiKeyEnv === 'string' && apiKeyEnv.trim() ? apiKeyEnv : current?.api_key_env || '',
        endpoint: typeof endpoint === 'string' && endpoint.trim() ? endpoint : current?.endpoint || '',
        api_key_set: typeof apiKeySet === 'boolean' ? apiKeySet : current?.api_key_set || false,
        api_key_source: isApiKeySource(sourceRaw)
          ? sourceRaw
          : current?.api_key_source || 'missing',
        api_key_value:
          typeof apiKeyValueRaw === 'string'
            ? apiKeyValueRaw
            : current?.api_key_value || '',
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
  for (const item of providers) {
    parsedApiKeys[item.provider] = item.api_key_value;
  }

  return {
    providers,
    apiKeys: parsedApiKeys,
    tone,
    systemPrompt,
    longPasteToFileEnabled,
    longPasteThresholdChars,
  };
};

const providerPlaceholder = (provider: ProviderSettings): string => {
  if (provider.api_key_source === 'settings') {
    return `Stored in settings. Paste new ${provider.provider} key to replace.`;
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

export function Settings({ isOpen, onClose, onSaved }: SettingsProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>('api');
  const [selectedProvider, setSelectedProvider] = useState<ModelProvider>('local');
  const [apiKeys, setApiKeys] = useState<Record<ApiKeyProvider, string>>(DEFAULT_API_KEYS);
  const [providers, setProviders] = useState<ProviderSettings[]>(DEFAULT_PROVIDER_SETTINGS);
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [tone, setTone] = useState('balanced');
  const [longPasteToFileEnabled, setLongPasteToFileEnabled] = useState(
    DEFAULT_LONG_PASTE_TO_FILE_ENABLED,
  );
  const [longPasteThresholdChars, setLongPasteThresholdChars] = useState(
    DEFAULT_LONG_PASTE_THRESHOLD_CHARS,
  );
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const loadSettings = async () => {
    setLoading(true);
    setStatus(null);
    try {
      const response = await fetch('/ui/api/settings');
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to load settings.'));
      }
      const parsed = parseSettingsPayload(payload);
      setProviders(parsed.providers);
      setApiKeys(parsed.apiKeys);
      setSystemPrompt(parsed.systemPrompt);
      setTone(parsed.tone);
      setLongPasteToFileEnabled(parsed.longPasteToFileEnabled);
      setLongPasteThresholdChars(parsed.longPasteThresholdChars);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load settings.';
      setStatus(message);
    } finally {
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
      const providersPayload: Record<string, { api_key: string }> = {};
      for (const provider of API_KEY_PROVIDERS) {
        const key = apiKeys[provider].trim();
        if (key) {
          providersPayload[provider] = { api_key: key };
        }
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
      };
      if (Object.keys(providersPayload).length > 0) {
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
      const parsed = parseSettingsPayload(body);
      setProviders(parsed.providers);
      setApiKeys(parsed.apiKeys);
      setSystemPrompt(parsed.systemPrompt);
      setTone(parsed.tone);
      setLongPasteToFileEnabled(parsed.longPasteToFileEnabled);
      setLongPasteThresholdChars(parsed.longPasteThresholdChars);
      setStatus('Saved');
      onSaved?.();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save settings.';
      setStatus(message);
    } finally {
      setSaving(false);
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
                      {providers.map((provider) => (
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
                            </div>
                          </div>
                          <div className="space-y-2 text-xs text-zinc-400">
                            <div>
                              Env: <span className="font-mono text-zinc-300">{provider.api_key_env}</span>
                            </div>
                            <div className="break-all">Endpoint: {provider.endpoint}</div>
                          </div>
                          <input
                            type="password"
                            value={apiKeys[provider.provider]}
                            onChange={(event) =>
                              setApiKeys((prev) => ({ ...prev, [provider.provider]: event.target.value }))
                            }
                            placeholder={providerPlaceholder(provider)}
                            className="mt-3 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
                          />
                        </div>
                      ))}
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
                          <button
                            type="button"
                            role="switch"
                            aria-checked={longPasteToFileEnabled}
                            onClick={() => setLongPasteToFileEnabled((prev) => !prev)}
                            className={`relative h-6 w-11 rounded-full transition-colors ${
                              longPasteToFileEnabled ? 'bg-emerald-500/70' : 'bg-zinc-700'
                            }`}
                          >
                            <span
                              className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
                                longPasteToFileEnabled ? 'translate-x-5' : 'translate-x-0.5'
                              }`}
                            />
                          </button>
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
                      <p className="text-sm text-zinc-400">
                        Memory controls are managed in backend settings API and will be expanded in this panel.
                      </p>
                    </div>
                  ) : null}

                  {!loading && activeTab === 'tools' ? (
                    <div className="space-y-4">
                      <p className="text-sm text-zinc-400">
                        Tools controls are available from backend settings API and will be wired here next.
                      </p>
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
