import { useEffect, useMemo, useState } from "react";

import ChatView from "./components/ChatView";
import DecisionPanel from "./components/DecisionPanel";
import type {
  DecisionOptionView,
  DecisionPacketView,
  Message,
  UIEvent,
} from "./types";

const MAX_EVENTS = 120;

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

export default function App() {
  const [statusOk, setStatusOk] = useState<boolean | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<UIEvent[]>([]);
  const [decision, setDecision] = useState<DecisionPacketView | null>(null);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

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
    const loadStatus = async () => {
      try {
        const resp = await fetch("/ui/api/status");
        if (!resp.ok) {
          throw new Error(`Status ${resp.status}`);
        }
        const payload = (await resp.json()) as {
          ok?: boolean;
          session_id?: string;
          decision?: unknown;
        };
        const headerSession = resp.headers.get("X-Slavik-Session");
        const nextSession = headerSession || payload.session_id || null;
        if (active) {
          setStatusOk(Boolean(payload.ok));
          if (nextSession) {
            setSessionId(nextSession);
          }
          setDecision(parseDecision(payload.decision));
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
        throw new Error(`Status ${resp.status}`);
      }
      const payload = (await resp.json()) as {
        session_id?: string;
        messages?: unknown;
        decision?: unknown;
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
      setInput("");
      setStatusOk(true);
    } catch {
      setStatusOk(false);
    } finally {
      setSending(false);
    }
  };

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
              <button
                type="button"
                className="rounded-full border border-neutral-800/80 bg-neutral-900/60 px-3 py-1.5 text-neutral-300"
              >
                Model: Select
              </button>
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
                <div className="text-xs text-neutral-400">Session: {sessionId ?? "pending"}</div>
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
              <DecisionPanel decision={decision} />

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
