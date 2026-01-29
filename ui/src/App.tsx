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
    return "bg-slate-500";
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
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6">
        <header className="flex flex-col gap-3 rounded-3xl border border-slate-800/80 bg-slate-900/70 px-5 py-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Slavik UI Workbench</p>
            <h1 className="text-2xl font-semibold text-slate-100">Slavik UI</h1>
          </div>
          <div className="flex flex-col gap-2 text-sm">
            <div className="flex items-center gap-2">
              <span className={`h-2.5 w-2.5 rounded-full ${statusDotClass(statusLabel)}`} />
              <span className="font-medium">Status: {statusLabel}</span>
            </div>
            <div className="text-xs text-slate-400">Session: {sessionId ?? "pending"}</div>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[2.2fr_1fr]">
          <ChatView
            messages={messages}
            input={input}
            sending={sending}
            onInputChange={setInput}
            onSend={handleSend}
          />

          <div className="flex flex-col gap-4">
            <DecisionPanel decision={decision} />

            <aside className="flex flex-1 flex-col gap-4 rounded-3xl border border-slate-800/80 bg-slate-900/60 p-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-slate-100">Event log</h2>
                <span className="text-xs text-slate-500">{events.length} items</span>
              </div>
              <div className="flex min-h-[30vh] flex-1 flex-col gap-3 overflow-y-auto rounded-3xl border border-slate-800/80 bg-slate-950/40 p-3 font-mono text-xs">
                {events.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-slate-800/70 bg-slate-900/60 px-4 py-6 text-slate-400">
                    Awaiting SSE events.
                  </div>
                ) : (
                  events.map((evt) => (
                    <div
                      key={evt.id}
                      className="rounded-2xl border border-slate-800/80 bg-slate-900/80 px-3 py-2"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-indigo-300">{evt.type}</span>
                        <span className="text-[10px] text-slate-500">{evt.ts}</span>
                      </div>
                      <div className="mt-1 text-slate-400">{toEventPreview(evt.payload)}</div>
                    </div>
                  ))
                )}
              </div>
            </aside>
          </div>
        </div>
      </div>
    </div>
  );
}
