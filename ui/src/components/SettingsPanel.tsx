import { useEffect, useMemo, useState } from "react";

import type { AppSettingsView, ProviderModels } from "../types";

type SettingsTab = "providers" | "personalization" | "memory" | "tools" | "chatdb";

type SettingsPanelProps = {
  open: boolean;
  settings: AppSettingsView | null;
  providerModels: ProviderModels[];
  loading: boolean;
  saving: boolean;
  importBusy: boolean;
  exportBusy: boolean;
  onClose: () => void;
  onRefresh: () => void;
  onSavePersonalization: (payload: { tone: string; system_prompt: string }) => Promise<void>;
  onSaveMemory: (payload: {
    auto_save_dialogue: boolean;
    inbox_max_items: number;
    inbox_ttl_days: number;
    inbox_writes_per_minute: number;
    embeddings_model: string;
  }) => Promise<void>;
  onSaveTools: (payload: { state: Record<string, boolean> }) => Promise<void>;
  onExportChats: () => Promise<void>;
  onImportChats: (file: File, mode: "replace" | "merge") => Promise<void>;
};

const tabs: Array<{ id: SettingsTab; title: string }> = [
  { id: "providers", title: "API Keys / Providers" },
  { id: "personalization", title: "Personalization" },
  { id: "memory", title: "Memory" },
  { id: "tools", title: "Tools" },
  { id: "chatdb", title: "Import / Export chats DB" },
];

export default function SettingsPanel({
  open,
  settings,
  providerModels,
  loading,
  saving,
  importBusy,
  exportBusy,
  onClose,
  onRefresh,
  onSavePersonalization,
  onSaveMemory,
  onSaveTools,
  onExportChats,
  onImportChats,
}: SettingsPanelProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>("providers");
  const [tone, setTone] = useState("balanced");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [autoSaveDialogue, setAutoSaveDialogue] = useState(false);
  const [inboxMaxItems, setInboxMaxItems] = useState(200);
  const [inboxTtlDays, setInboxTtlDays] = useState(30);
  const [inboxWritesPerMinute, setInboxWritesPerMinute] = useState(6);
  const [embeddingsModel, setEmbeddingsModel] = useState("all-MiniLM-L6-v2");
  const [toolsState, setToolsState] = useState<Record<string, boolean>>({});
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importMode, setImportMode] = useState<"replace" | "merge">("replace");

  useEffect(() => {
    if (!settings) {
      return;
    }
    setTone(settings.personalization.tone);
    setSystemPrompt(settings.personalization.system_prompt);
    setAutoSaveDialogue(settings.memory.auto_save_dialogue);
    setInboxMaxItems(settings.memory.inbox_max_items);
    setInboxTtlDays(settings.memory.inbox_ttl_days);
    setInboxWritesPerMinute(settings.memory.inbox_writes_per_minute);
    setEmbeddingsModel(settings.memory.embeddings_model);
    setToolsState(settings.tools.state);
  }, [settings]);

  const providerByName = useMemo(() => {
    const map: Record<string, ProviderModels> = {};
    for (const item of providerModels) {
      map[item.provider] = item;
    }
    return map;
  }, [providerModels]);

  if (!open) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="flex max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-3xl border border-neutral-700 bg-neutral-950 shadow-2xl shadow-black/50">
        <div className="flex items-center justify-between border-b border-neutral-800/80 px-5 py-4">
          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-neutral-500">Settings</div>
            <h2 className="text-lg font-semibold text-neutral-100">Workspace controls</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onRefresh}
              className="rounded-full border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-800"
            >
              Refresh
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-full border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-800"
            >
              Close
            </button>
          </div>
        </div>

        <div className="grid min-h-0 flex-1 grid-cols-1 gap-0 lg:grid-cols-[240px_minmax(0,1fr)]">
          <aside className="border-r border-neutral-800/80 bg-neutral-950/60 p-3">
            <div className="flex flex-col gap-2">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`rounded-xl border px-3 py-2 text-left text-sm transition ${
                    activeTab === tab.id
                      ? "border-neutral-200 bg-neutral-200 text-neutral-900"
                      : "border-neutral-800/80 bg-neutral-900/50 text-neutral-300 hover:border-neutral-700 hover:bg-neutral-900"
                  }`}
                >
                  {tab.title}
                </button>
              ))}
            </div>
          </aside>

          <section className="min-h-0 overflow-y-auto p-5">
            {loading || !settings ? (
              <div className="rounded-2xl border border-dashed border-neutral-700 px-4 py-6 text-sm text-neutral-400">
                Loading settings...
              </div>
            ) : null}

            {!loading && settings && activeTab === "providers" ? (
              <div className="space-y-3">
                {settings.providers.map((provider) => {
                  const modelsInfo = providerByName[provider.provider];
                  const modelsCount = modelsInfo ? modelsInfo.models.length : 0;
                  return (
                    <div
                      key={provider.provider}
                      className="rounded-2xl border border-neutral-800/80 bg-neutral-900/60 p-4"
                    >
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-semibold text-neutral-100">{provider.provider}</div>
                        <span
                          className={`rounded-full px-2 py-0.5 text-xs ${
                            provider.api_key_set
                              ? "bg-emerald-500/20 text-emerald-300"
                              : "bg-amber-500/20 text-amber-300"
                          }`}
                        >
                          {provider.api_key_set ? "key set" : "key missing"}
                        </span>
                      </div>
                      <div className="mt-2 space-y-1 text-xs text-neutral-400">
                        <div>Env: {provider.api_key_env}</div>
                        <div className="break-all">Endpoint: {provider.endpoint}</div>
                        <div>Available models: {modelsCount}</div>
                        {modelsInfo?.error ? <div className="text-rose-300">{modelsInfo.error}</div> : null}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : null}

            {!loading && settings && activeTab === "personalization" ? (
              <div className="space-y-4 rounded-2xl border border-neutral-800/80 bg-neutral-900/60 p-4">
                <label className="flex flex-col gap-1 text-sm text-neutral-300">
                  Tone
                  <input
                    value={tone}
                    onChange={(event) => setTone(event.target.value)}
                    className="rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100"
                  />
                </label>
                <label className="flex flex-col gap-1 text-sm text-neutral-300">
                  System prompt
                  <textarea
                    value={systemPrompt}
                    onChange={(event) => setSystemPrompt(event.target.value)}
                    rows={6}
                    className="rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100"
                  />
                </label>
                <button
                  type="button"
                  disabled={saving}
                  onClick={() => {
                    void onSavePersonalization({
                      tone: tone.trim() || "balanced",
                      system_prompt: systemPrompt,
                    });
                  }}
                  className="rounded-xl border border-neutral-700 bg-neutral-200 px-4 py-2 text-sm font-semibold text-neutral-950 disabled:opacity-50"
                >
                  {saving ? "Saving..." : "Save personalization"}
                </button>
              </div>
            ) : null}

            {!loading && settings && activeTab === "memory" ? (
              <div className="space-y-4 rounded-2xl border border-neutral-800/80 bg-neutral-900/60 p-4">
                <label className="flex items-center gap-2 text-sm text-neutral-200">
                  <input
                    type="checkbox"
                    checked={autoSaveDialogue}
                    onChange={(event) => setAutoSaveDialogue(event.target.checked)}
                    className="h-4 w-4 rounded border-neutral-700 bg-neutral-950"
                  />
                  Auto save dialogue
                </label>
                <label className="flex flex-col gap-1 text-sm text-neutral-300">
                  Embeddings model
                  <input
                    value={embeddingsModel}
                    onChange={(event) => setEmbeddingsModel(event.target.value)}
                    className="rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100"
                  />
                </label>
                <div className="grid gap-3 sm:grid-cols-3">
                  <label className="flex flex-col gap-1 text-sm text-neutral-300">
                    Inbox max items
                    <input
                      type="number"
                      min={1}
                      value={inboxMaxItems}
                      onChange={(event) => setInboxMaxItems(Number(event.target.value) || 1)}
                      className="rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-sm text-neutral-300">
                    Inbox TTL days
                    <input
                      type="number"
                      min={1}
                      value={inboxTtlDays}
                      onChange={(event) => setInboxTtlDays(Number(event.target.value) || 1)}
                      className="rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-sm text-neutral-300">
                    Writes / minute
                    <input
                      type="number"
                      min={1}
                      value={inboxWritesPerMinute}
                      onChange={(event) => setInboxWritesPerMinute(Number(event.target.value) || 1)}
                      className="rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100"
                    />
                  </label>
                </div>
                <button
                  type="button"
                  disabled={saving}
                  onClick={() => {
                    void onSaveMemory({
                      auto_save_dialogue: autoSaveDialogue,
                      inbox_max_items: Math.max(1, inboxMaxItems),
                      inbox_ttl_days: Math.max(1, inboxTtlDays),
                      inbox_writes_per_minute: Math.max(1, inboxWritesPerMinute),
                      embeddings_model: embeddingsModel.trim() || "all-MiniLM-L6-v2",
                    });
                  }}
                  className="rounded-xl border border-neutral-700 bg-neutral-200 px-4 py-2 text-sm font-semibold text-neutral-950 disabled:opacity-50"
                >
                  {saving ? "Saving..." : "Save memory settings"}
                </button>
              </div>
            ) : null}

            {!loading && settings && activeTab === "tools" ? (
              <div className="space-y-4 rounded-2xl border border-neutral-800/80 bg-neutral-900/60 p-4">
                <div className="text-xs text-neutral-400">
                  Changes are persisted to config and applied on next restart.
                </div>
                <div className="grid gap-2 sm:grid-cols-2">
                  {Object.entries(toolsState)
                    .sort((left, right) => left[0].localeCompare(right[0]))
                    .map(([toolName, enabled]) => (
                      <label
                        key={toolName}
                        className="flex items-center justify-between rounded-xl border border-neutral-800 bg-neutral-950/60 px-3 py-2 text-sm"
                      >
                        <span className="text-neutral-200">{toolName}</span>
                        <input
                          type="checkbox"
                          checked={enabled}
                          onChange={(event) => {
                            setToolsState((prev) => ({
                              ...prev,
                              [toolName]: event.target.checked,
                            }));
                          }}
                          className="h-4 w-4 rounded border-neutral-700 bg-neutral-950"
                        />
                      </label>
                    ))}
                </div>
                <div className="rounded-xl border border-neutral-800 bg-neutral-950/40 p-3 text-xs text-neutral-400">
                  Registry: {Object.keys(settings.tools.registry).join(", ") || "no tools"}
                </div>
                <button
                  type="button"
                  disabled={saving}
                  onClick={() => {
                    void onSaveTools({ state: toolsState });
                  }}
                  className="rounded-xl border border-neutral-700 bg-neutral-200 px-4 py-2 text-sm font-semibold text-neutral-950 disabled:opacity-50"
                >
                  {saving ? "Saving..." : "Save tools settings"}
                </button>
              </div>
            ) : null}

            {!loading && settings && activeTab === "chatdb" ? (
              <div className="space-y-4 rounded-2xl border border-neutral-800/80 bg-neutral-900/60 p-4">
                <div className="rounded-xl border border-neutral-800 bg-neutral-950/50 p-3 text-xs text-neutral-400">
                  Export downloads all chats as JSON. Import supports replace or merge.
                </div>
                <button
                  type="button"
                  disabled={exportBusy}
                  onClick={() => {
                    void onExportChats();
                  }}
                  className="rounded-xl border border-neutral-700 bg-neutral-200 px-4 py-2 text-sm font-semibold text-neutral-950 disabled:opacity-50"
                >
                  {exportBusy ? "Exporting..." : "Export chats DB"}
                </button>
                <div className="space-y-2 rounded-xl border border-neutral-800 bg-neutral-950/50 p-3">
                  <label className="flex flex-col gap-1 text-sm text-neutral-300">
                    Import file (.json)
                    <input
                      type="file"
                      accept="application/json"
                      onChange={(event) => {
                        const nextFile = event.target.files?.[0] ?? null;
                        setImportFile(nextFile);
                      }}
                      className="rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2 text-xs text-neutral-300"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-sm text-neutral-300">
                    Import mode
                    <select
                      value={importMode}
                      onChange={(event) => {
                        const nextMode = event.target.value === "merge" ? "merge" : "replace";
                        setImportMode(nextMode);
                      }}
                      className="rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100"
                    >
                      <option value="replace">replace</option>
                      <option value="merge">merge</option>
                    </select>
                  </label>
                  <button
                    type="button"
                    disabled={!importFile || importBusy}
                    onClick={() => {
                      if (!importFile) {
                        return;
                      }
                      void onImportChats(importFile, importMode);
                    }}
                    className="rounded-xl border border-neutral-700 bg-neutral-200 px-4 py-2 text-sm font-semibold text-neutral-950 disabled:opacity-50"
                  >
                    {importBusy ? "Importing..." : "Import chats DB"}
                  </button>
                </div>
              </div>
            ) : null}
          </section>
        </div>
      </div>
    </div>
  );
}
