import { useEffect, useMemo, useState } from "react";

import ChatView from "./components/ChatView";
import DecisionPanel from "./components/DecisionPanel";
import type {
  DecisionOptionView,
  DecisionPacketView,
  Message,
  ProviderModels,
  SelectedModel,
  UIEvent,
} from "./types";

const MAX_EVENTS = 120;
const SESSION_STORAGE_KEY = "slavik.ui.session_id";
type ProjectCommand = "find" | "index";

const isMessage = (value: unknown): value is Message => {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as { role?: string; content?: string };
  return (
    typeof candidate.role === "string" &&
    (candidate.role === "user" ||
      candidate.role === "assistant" ||
      candidate.role === "system") &&
    typeof candidate.content === "string"
  );
};

const isDecisionOption = (value: unknown): value is DecisionOptionView => {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as {
    id?: string;
    title?: string;
    action?: string;
    risk?: string;
  };
  return (
    typeof candidate.id === "string" &&
    typeof candidate.title === "string" &&
    typeof candidate.action === "string" &&
    typeof candidate.risk === "string"
  );
};

const parseDecision = (value: unknown): DecisionPacketView | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const candidate = value as {
    id?: unknown;
    reason?: unknown;
    summary?: unknown;
    options?: unknown;
    default_option_id?: unknown;
  };
  if (
    typeof candidate.id !== "string" ||
    typeof candidate.reason !== "string" ||
    typeof candidate.summary !== "string"
  ) {
    return null;
  }
  const options = Array.isArray(candidate.options)
    ? candidate.options.filter(isDecisionOption)
    : [];
  if (options.length === 0) {
    return null;
  }
  const defaultOption =
    typeof candidate.default_option_id === "string" ? candidate.default_option_id : null;
  return {
    id: candidate.id,
    reason: candidate.reason,
    summary: candidate.summary,
    options,
    default_option_id: defaultOption,
  };
};

const parseMessages = (value: unknown): Message[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(isMessage);
};

const parseSelectedModel = (value: unknown): SelectedModel | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const candidate = value as { provider?: unknown; model?: unknown };
  if (typeof candidate.provider !== "string" || typeof candidate.model !== "string") {
    return null;
  }
  return { provider: candidate.provider, model: candidate.model };
};

const parseProviderModels = (value: unknown): ProviderModels[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const items: ProviderModels[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const candidate = item as { provider?: unknown; models?: unknown; error?: unknown };
    if (typeof candidate.provider !== "string" || !Array.isArray(candidate.models)) {
      continue;
    }
    const models = candidate.models.filter((entry): entry is string => typeof entry === "string");
    const error =
      typeof candidate.error === "string" || candidate.error === null ? candidate.error : null;
    items.push({ provider: candidate.provider, models, error });
  }
  return items;
};

const toEventPreview = (payload: unknown): string => {
  try {
    const raw = JSON.stringify(payload ?? {}, null, 0);
    return raw.length > 160 ? `${raw.slice(0, 160)}...` : raw;
  } catch {
    return "{unserializable}";
  }
};

const statusDotClass = (status: string): string => {
  if (status === "busy") {
    return "bg-amber-400";
  }
  if (status === "error") {
    return "bg-rose-500";
  }
  if (status === "loading") {
    return "bg-neutral-500";
  }
  return "bg-emerald-500";
};

const isRecoverableHttpStatus = (status: number): boolean => status >= 400 && status < 500;

const readStoredSessionId = (): string | null => {
  try {
    const raw = window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (typeof raw !== "string") {
      return null;
    }
    const trimmed = raw.trim();
    return trimmed || null;
  } catch {
    return null;
  }
};

const persistSessionId = (sessionId: string | null): void => {
  try {
    if (!sessionId) {
      window.localStorage.removeItem(SESSION_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  } catch {
    return;
  }
};

export default function App() {
  const [statusOk, setStatusOk] = useState<boolean | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<UIEvent[]>([]);
  const [decision, setDecision] = useState<DecisionPacketView | null>(null);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [providerModels, setProviderModels] = useState<ProviderModels[]>([]);
  const [selectedModel, setSelectedModel] = useState<SelectedModel | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [savingModel, setSavingModel] = useState(false);
  const [projectCommand, setProjectCommand] = useState<ProjectCommand>("find");
  const [projectArgs, setProjectArgs] = useState("");
  const [projectBusy, setProjectBusy] = useState(false);

  const appendSystemMessage = (content: string) => {
    const normalized = content.trim();
    if (!normalized) {
      return;
    }
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "system" && last.content === normalized) {
        return prev;
      }
      return [...prev, { role: "system", content: normalized }];
    });
  };

  const statusLabel = useMemo(() => {
    if (sending) {
      return "busy";
    }
    if (statusOk === null) {
      return "loading";
    }
    return statusOk ? "ok" : "error";
  }, [sending, statusOk]);

  useEffect(() => {
    let active = true;
    const hydrateSession = async (currentSessionId: string) => {
      try {
        const resp = await fetch(`/ui/api/sessions/${encodeURIComponent(currentSessionId)}`, {
          headers: {
            "X-Slavik-Session": currentSessionId,
          },
        });
        if (!resp.ok || !active) {
          return;
        }
        const payload = (await resp.json()) as {
          session?: {
            messages?: unknown;
            decision?: unknown;
            selected_model?: unknown;
          };
        };
        if (!payload.session || !active) {
          return;
        }
        const parsedMessages = parseMessages(payload.session.messages);
        setMessages(parsedMessages);
        setDecision(parseDecision(payload.session.decision));
        const parsedModel = parseSelectedModel(payload.session.selected_model);
        setSelectedModel(parsedModel);
        if (parsedModel) {
          setSelectedProvider(parsedModel.provider);
          setSelectedModelId(parsedModel.model);
        }
      } catch {
        return;
      }
    };
    const loadStatus = async () => {
      try {
        const storedSession = readStoredSessionId();
        const headers: Record<string, string> = {};
        if (storedSession) {
          headers["X-Slavik-Session"] = storedSession;
        }
        const resp = await fetch(
          "/ui/api/status",
          Object.keys(headers).length > 0 ? { headers } : undefined,
        );
        if (!resp.ok) {
          throw new Error(`Status ${resp.status}`);
        }
        const payload = (await resp.json()) as {
          ok?: boolean;
          session_id?: string;
          decision?: unknown;
          selected_model?: unknown;
        };
        const headerSession = resp.headers.get("X-Slavik-Session");
        const nextSession = headerSession || payload.session_id || null;
        if (active) {
          setStatusOk(Boolean(payload.ok));
          if (nextSession) {
            setSessionId(nextSession);
            persistSessionId(nextSession);
            void hydrateSession(nextSession);
          }
          setDecision(parseDecision(payload.decision));
          const parsedModel = parseSelectedModel(payload.selected_model);
          setSelectedModel(parsedModel);
          if (parsedModel) {
            setSelectedProvider(parsedModel.provider);
            setSelectedModelId(parsedModel.model);
          }
        }
      } catch {
        if (active) {
          setStatusOk(false);
        }
      }
    };
    loadStatus();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    persistSessionId(sessionId);
  }, [sessionId]);

  useEffect(() => {
    let active = true;
    const loadModels = async () => {
      try {
        const resp = await fetch("/ui/api/models");
        if (!resp.ok) {
          throw new Error(`Status ${resp.status}`);
        }
        const payload = (await resp.json()) as { providers?: unknown };
        if (!active) {
          return;
        }
        const parsed = parseProviderModels(payload.providers);
        setProviderModels(parsed);
        if (parsed.length === 0) {
          return;
        }
        if (!selectedProvider || !parsed.some((item) => item.provider === selectedProvider)) {
          const first = parsed[0];
          setSelectedProvider(first.provider);
          if (!selectedModelId && first.models.length > 0) {
            setSelectedModelId(first.models[0]);
          }
        }
      } catch {
        if (active) {
          appendSystemMessage("Не удалось загрузить список моделей.");
        }
      }
    };
    loadModels();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    const url = `/ui/api/events/stream?session_id=${encodeURIComponent(sessionId)}`;
    const source = new EventSource(url, { withCredentials: false });

    source.onmessage = (evt) => {
      try {
        const parsed = JSON.parse(evt.data) as UIEvent;
        if (!parsed || typeof parsed !== "object") {
          return;
        }
        setEvents((prev) => {
          const next = [...prev, parsed];
          if (next.length > MAX_EVENTS) {
            next.splice(0, next.length - MAX_EVENTS);
          }
          return next;
        });
        if (parsed.type === "message.append" && parsed.payload) {
          const payload = parsed.payload as { message?: unknown };
          if (payload.message && isMessage(payload.message)) {
            const incoming = payload.message;
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.role === incoming.role && last.content === incoming.content) {
                return prev;
              }
              return [...prev, incoming];
            });
          }
        }
        if (parsed.type === "decision.packet" && parsed.payload) {
          const payload = parsed.payload as { decision?: unknown };
          const nextDecision = parseDecision(payload.decision);
          if (nextDecision) {
            setDecision(nextDecision);
          }
        }
        if (parsed.type === "session.model" && parsed.payload) {
          const payload = parsed.payload as { selected_model?: unknown };
          const nextModel = parseSelectedModel(payload.selected_model);
          if (nextModel) {
            setSelectedModel(nextModel);
            setSelectedProvider(nextModel.provider);
            setSelectedModelId(nextModel.model);
          }
        }
        if (parsed.type === "status" && parsed.payload) {
          const payload = parsed.payload as { ok?: boolean };
          setStatusOk(Boolean(payload.ok));
        }
      } catch {
        return;
      }
    };

    source.onerror = () => {
      setStatusOk(false);
    };

    return () => {
      source.close();
    };
  }, [sessionId]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || sending) {
      return;
    }
    setSending(true);
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (sessionId) {
        headers["X-Slavik-Session"] = sessionId;
      }
      const resp = await fetch("/ui/api/chat/send", {
        method: "POST",
        headers,
        body: JSON.stringify({ content: trimmed }),
      });
      if (!resp.ok) {
        try {
          const errorPayload = (await resp.json()) as {
            error?: { code?: string; message?: string };
          };
          const errorCode = errorPayload.error?.code;
          const errorMessage = errorPayload.error?.message;
          if (errorCode === "model_not_selected") {
            appendSystemMessage(errorMessage || "Сначала выбери модель.");
          } else if (typeof errorMessage === "string" && errorMessage.trim()) {
            appendSystemMessage(errorMessage);
          }
        } catch {
          appendSystemMessage("Ошибка запроса. Проверь выбор модели и повтори.");
        }
        setStatusOk(isRecoverableHttpStatus(resp.status));
        return;
      }
      const payload = (await resp.json()) as {
        session_id?: string;
        messages?: unknown;
        decision?: unknown;
        selected_model?: unknown;
      };
      const headerSession = resp.headers.get("X-Slavik-Session");
      const nextSession = headerSession || payload.session_id || null;
      if (nextSession) {
        setSessionId(nextSession);
      }
      const parsedMessages = parseMessages(payload.messages);
      if (parsedMessages.length > 0) {
        setMessages(parsedMessages);
      }
      const nextDecision = parseDecision(payload.decision);
      if (nextDecision) {
        setDecision(nextDecision);
      }
      const nextModel = parseSelectedModel(payload.selected_model);
      if (nextModel) {
        setSelectedModel(nextModel);
      }
      setInput("");
      setStatusOk(true);
    } catch {
      appendSystemMessage("Сервис недоступен. Проверь сеть и повтори.");
      setStatusOk(false);
    } finally {
      setSending(false);
    }
  };

  const handleSetModel = async () => {
    if (!sessionId || !selectedProvider || !selectedModelId || savingModel) {
      return;
    }
    setSavingModel(true);
    try {
      const resp = await fetch("/ui/api/session-model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Slavik-Session": sessionId,
        },
        body: JSON.stringify({ provider: selectedProvider, model: selectedModelId }),
      });
      if (!resp.ok) {
        const payload = (await resp.json()) as {
          error?: { message?: string; details?: { suggestion?: string } };
        };
        const message = payload.error?.message || "Не удалось выбрать модель.";
        const suggestion = payload.error?.details?.suggestion;
        appendSystemMessage(suggestion ? `${message} Подсказка: ${suggestion}` : message);
        return;
      }
      const payload = (await resp.json()) as { selected_model?: unknown };
      const nextModel = parseSelectedModel(payload.selected_model);
      if (nextModel) {
        setSelectedModel(nextModel);
      }
    } catch {
      appendSystemMessage("Ошибка установки модели.");
    } finally {
      setSavingModel(false);
    }
  };

  const handleNewChat = async () => {
    try {
      const resp = await fetch("/ui/api/sessions", { method: "POST" });
      if (!resp.ok) {
        throw new Error(`Status ${resp.status}`);
      }
      const payload = (await resp.json()) as { session?: { session_id?: string } };
      const nextSession = payload.session?.session_id ?? null;
      if (nextSession) {
        setSessionId(nextSession);
      }
      setMessages([]);
      setEvents([]);
      setDecision(null);
      setStatusOk(true);
    } catch {
      appendSystemMessage("Не удалось создать новую сессию.");
    }
  };

  const handleProjectRun = async () => {
    if (!sessionId || projectBusy) {
      appendSystemMessage("Сначала создай/выбери сессию и модель.");
      return;
    }
    setProjectBusy(true);
    try {
      const resp = await fetch("/ui/api/tools/project", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Slavik-Session": sessionId,
        },
        body: JSON.stringify({ command: projectCommand, args: projectArgs }),
      });
      if (!resp.ok) {
        try {
          const payload = (await resp.json()) as {
            error?: { message?: string };
          };
          const message = payload.error?.message;
          appendSystemMessage(message || "Ошибка вызова project инструмента.");
        } catch {
          appendSystemMessage("Ошибка вызова project инструмента.");
        }
        setStatusOk(isRecoverableHttpStatus(resp.status));
        return;
      }
      const payload = (await resp.json()) as {
        session_id?: string;
        messages?: unknown;
        decision?: unknown;
        selected_model?: unknown;
      };
      const headerSession = resp.headers.get("X-Slavik-Session");
      const nextSession = headerSession || payload.session_id || null;
      if (nextSession) {
        setSessionId(nextSession);
      }
      const parsedMessages = parseMessages(payload.messages);
      if (parsedMessages.length > 0) {
        setMessages(parsedMessages);
      }
      const nextDecision = parseDecision(payload.decision);
      if (nextDecision) {
        setDecision(nextDecision);
      }
      const nextModel = parseSelectedModel(payload.selected_model);
      if (nextModel) {
        setSelectedModel(nextModel);
      }
      setStatusOk(true);
    } catch {
      appendSystemMessage("Сервис недоступен. Проверь сеть и повтори.");
      setStatusOk(false);
    } finally {
      setProjectBusy(false);
    }
  };

  const modelsForSelectedProvider = useMemo(() => {
    const found = providerModels.find((item) => item.provider === selectedProvider);
    return found?.models ?? [];
  }, [providerModels, selectedProvider]);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="flex min-h-screen">
        <aside className="hidden w-64 flex-col border-r border-neutral-800/80 bg-neutral-950/60 p-4 md:flex">
          <div className="flex items-center gap-2 text-sm text-neutral-400">
            <span className="h-2 w-2 rounded-full bg-neutral-200" />
            Slavik UI
          </div>
          <button
            type="button"
            onClick={handleNewChat}
            className="mt-4 rounded-2xl bg-neutral-200 px-4 py-2 text-sm font-semibold text-neutral-950 shadow-lg shadow-black/30 hover:bg-neutral-100"
          >
            New chat
          </button>
          <div className="mt-6 text-xs uppercase tracking-[0.3em] text-neutral-500">
            Conversations
          </div>
          <div className="mt-3 space-y-2 text-sm text-neutral-400">
            <div className="rounded-xl border border-dashed border-neutral-800/80 px-3 py-2">
              No chats yet
            </div>
            <div className="rounded-xl border border-neutral-800/60 bg-neutral-900/50 px-3 py-2 opacity-60">
              Placeholder thread
            </div>
          </div>
          <div className="mt-auto pt-6 text-xs text-neutral-500">v0 skeleton</div>
        </aside>

        <main className="flex min-w-0 flex-1 flex-col">
          <header className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800/80 bg-neutral-950/80 px-5 py-4">
            <div>
              <div className="text-xs uppercase tracking-[0.3em] text-neutral-500">Workspace</div>
              <div className="text-lg font-semibold text-neutral-100">Conversation</div>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <div className="flex items-center gap-2 rounded-full border border-neutral-800/80 bg-neutral-900/60 px-3 py-1.5">
                <span className={`h-2.5 w-2.5 rounded-full ${statusDotClass(statusLabel)}`} />
                <span className="font-medium">Status: {statusLabel}</span>
              </div>
              <div className="flex items-center gap-2 rounded-full border border-neutral-800/80 bg-neutral-900/60 px-2 py-1">
                <select
                  value={selectedProvider}
                  onChange={(event) => {
                    const nextProvider = event.target.value;
                    setSelectedProvider(nextProvider);
                    const nextModels =
                      providerModels.find((item) => item.provider === nextProvider)?.models ?? [];
                    setSelectedModelId(nextModels[0] ?? "");
                  }}
                  className="rounded-full border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200"
                >
                  <option value="">provider</option>
                  {providerModels.map((item) => (
                    <option key={item.provider} value={item.provider}>
                      {item.provider}
                    </option>
                  ))}
                </select>
                <select
                  value={selectedModelId}
                  onChange={(event) => setSelectedModelId(event.target.value)}
                  className="max-w-[220px] rounded-full border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200"
                >
                  <option value="">model</option>
                  {modelsForSelectedProvider.map((modelId) => (
                    <option key={modelId} value={modelId}>
                      {modelId}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={handleSetModel}
                  disabled={!sessionId || !selectedProvider || !selectedModelId || savingModel}
                  className="rounded-full border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200 disabled:opacity-50"
                >
                  {savingModel ? "..." : "Set"}
                </button>
              </div>
              <button
                type="button"
                className="rounded-full border border-neutral-800/80 bg-neutral-900/60 px-3 py-1.5 text-neutral-300"
              >
                Settings
              </button>
            </div>
          </header>

          <div className="flex min-h-0 flex-1 flex-col gap-6 p-5 lg:flex-row">
            <section className="flex min-w-0 flex-1 flex-col gap-4">
              <div className="flex items-center justify-between rounded-2xl border border-neutral-800/80 bg-neutral-900/60 px-4 py-3 text-sm">
                <div className="text-neutral-200">Current conversation</div>
                <div className="text-xs text-neutral-400">
                  Session: {sessionId ?? "pending"}
                  {selectedModel ? ` · ${selectedModel.provider}/${selectedModel.model}` : ""}
                </div>
              </div>
              <ChatView
                messages={messages}
                input={input}
                sending={sending}
                onInputChange={setInput}
                onSend={handleSend}
              />
            </section>

            <aside className="flex w-full flex-col gap-4 lg:w-[360px]">
              <DecisionPanel
                decision={decision}
                projectCommand={projectCommand}
                projectArgs={projectArgs}
                projectBusy={projectBusy}
                onProjectCommandChange={setProjectCommand}
                onProjectArgsChange={setProjectArgs}
                onProjectRun={handleProjectRun}
              />

              <div className="flex flex-1 flex-col gap-4 rounded-3xl border border-neutral-800/80 bg-neutral-900/60 p-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-neutral-100">Event log</h2>
                  <span className="text-xs text-neutral-500">{events.length} items</span>
                </div>
                <div className="flex min-h-[30vh] flex-1 flex-col gap-3 overflow-y-auto rounded-3xl border border-neutral-800/80 bg-neutral-950/40 p-3 font-mono text-xs">
                  {events.length === 0 ? (
                    <div className="rounded-2xl border border-dashed border-neutral-800/70 bg-neutral-900/60 px-4 py-6 text-neutral-400">
                      Awaiting SSE events.
                    </div>
                  ) : (
                    events.map((evt) => (
                      <div
                        key={evt.id}
                        className="rounded-2xl border border-neutral-800/80 bg-neutral-900/80 px-3 py-2"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-neutral-200">{evt.type}</span>
                          <span className="text-[10px] text-neutral-500">{evt.ts}</span>
                        </div>
                        <div className="mt-1 text-neutral-400">{toEventPreview(evt.payload)}</div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </aside>
          </div>
        </main>
      </div>
    </div>
  );
}
