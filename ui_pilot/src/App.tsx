import { useEffect, useMemo, useState } from "react";

const MAX_EVENTS = 120;

type Message = {
  role: "user" | "assistant";
  content: string;
};

type PilotEvent = {
  id: string;
  type: string;
  ts: string;
  payload: unknown;
};

const isMessage = (value: unknown): value is Message => {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as { role?: string; content?: string };
  return (
    typeof candidate.role === "string" &&
    (candidate.role === "user" || candidate.role === "assistant") &&
    typeof candidate.content === "string"
  );
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
    return "bg-slate-400";
  }
  return "bg-emerald-500";
};

export default function App() {
  const [statusOk, setStatusOk] = useState<boolean | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<PilotEvent[]>([]);
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
        const resp = await fetch("/pilot/api/status");
        if (!resp.ok) {
          throw new Error(`Status ${resp.status}`);
        }
        const payload = (await resp.json()) as {
          ok?: boolean;
          session_id?: string;
        };
        const headerSession = resp.headers.get("X-Slavik-Session");
        const nextSession = headerSession || payload.session_id || null;
        if (active) {
          setStatusOk(Boolean(payload.ok));
          if (nextSession) {
            setSessionId(nextSession);
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
    if (!sessionId) {
      return;
    }
    const url = `/pilot/api/events/stream?session_id=${encodeURIComponent(sessionId)}`;
    const source = new EventSource(url, { withCredentials: false });

    source.onmessage = (evt) => {
      try {
        const parsed = JSON.parse(evt.data) as PilotEvent;
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
      const resp = await fetch("/pilot/api/chat/send", {
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
      setInput("");
      setStatusOk(true);
    } catch {
      setStatusOk(false);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <header className="panel panel-strong fade-in flex flex-col gap-3 px-5 py-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-[color:var(--muted)]">
              Slavik Pilot Workbench
            </p>
            <h1 className="text-2xl font-semibold">UI-pilot /pilot</h1>
          </div>
          <div className="flex flex-col gap-2 text-sm">
            <div className="flex items-center gap-2">
              <span className={`h-2.5 w-2.5 rounded-full ${statusDotClass(statusLabel)}`} />
              <span className="font-medium">Status: {statusLabel}</span>
            </div>
            <div className="text-xs text-[color:var(--muted)]">
              Session: {sessionId ?? "pending"}
            </div>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[2.1fr_1fr]">
          <section className="panel fade-in fade-in-delay-1 flex flex-col gap-4 p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Messages</h2>
              <span className="text-xs text-[color:var(--muted)]">
                {messages.length} total
              </span>
            </div>
            <div className="flex h-[56vh] flex-col gap-3 overflow-y-auto pr-2">
              {messages.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-[color:var(--line)] bg-white/60 px-4 py-6 text-sm text-[color:var(--muted)]">
                  No messages yet. Send a prompt to populate the session.
                </div>
              ) : (
                messages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    className={`flex ${
                      message.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-4 py-3 shadow-sm ${
                        message.role === "user"
                          ? "bg-[color:var(--accent)] text-white"
                          : "bg-white/90 text-[color:var(--ink)]"
                      }`}
                    >
                      <div className="text-[11px] uppercase tracking-[0.18em] opacity-80">
                        {message.role}
                      </div>
                      <div className="mt-1 whitespace-pre-wrap text-sm leading-relaxed">
                        {message.content}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
            <form
              className="flex flex-col gap-3 border-t border-[color:var(--line)] pt-4 md:flex-row"
              onSubmit={(event) => {
                event.preventDefault();
                void handleSend();
              }}
            >
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Send a message..."
                className="flex-1 rounded-2xl border border-[color:var(--line)] bg-white/80 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-[color:var(--accent)]"
              />
              <button
                type="submit"
                disabled={sending || !input.trim()}
                className="rounded-2xl bg-[color:var(--accent)] px-6 py-3 text-sm font-semibold text-white transition-opacity disabled:opacity-50"
              >
                {sending ? "Sending..." : "Send"}
              </button>
            </form>
          </section>

          <aside className="panel fade-in fade-in-delay-2 flex flex-col gap-4 p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Event log</h2>
              <span className="text-xs text-[color:var(--muted)]">{events.length} items</span>
            </div>
            <div className="flex h-[56vh] flex-col gap-3 overflow-y-auto pr-2 font-mono text-xs">
              {events.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-[color:var(--line)] bg-white/60 px-4 py-6 text-[color:var(--muted)]">
                  Awaiting SSE events.
                </div>
              ) : (
                events.map((evt) => (
                  <div
                    key={evt.id}
                    className="rounded-2xl border border-[color:var(--line)] bg-white/80 px-3 py-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-[color:var(--accent-strong)]">
                        {evt.type}
                      </span>
                      <span className="text-[10px] text-[color:var(--muted)]">
                        {evt.ts}
                      </span>
                    </div>
                    <div className="mt-1 text-[color:var(--muted)]">
                      {toEventPreview(evt.payload)}
                    </div>
                  </div>
                ))
              )}
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
