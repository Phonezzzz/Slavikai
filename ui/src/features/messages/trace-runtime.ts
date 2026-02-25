export type ToolCallView = Record<string, unknown>;
export type TraceEventView = Record<string, unknown>;

export type TraceDiagnostics = {
  traceId: string;
  toolCalls: ToolCallView[];
  events: TraceEventView[];
  fetchedAt: number;
};

const parseRecordArray = (value: unknown): Record<string, unknown>[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is Record<string, unknown> => {
    return !!item && typeof item === 'object' && !Array.isArray(item);
  });
};

const fetchJson = async (url: string): Promise<unknown> => {
  const response = await fetch(url);
  const payload: unknown = await response.json();
  if (!response.ok) {
    const fallback = `Request failed: ${response.status}`;
    if (payload && typeof payload === 'object') {
      const maybeError = payload as { error?: { message?: unknown } };
      const message = maybeError.error?.message;
      if (typeof message === 'string' && message.trim()) {
        throw new Error(message.trim());
      }
    }
    throw new Error(fallback);
  }
  return payload;
};

export const fetchTraceDiagnostics = async (traceId: string): Promise<TraceDiagnostics> => {
  const normalizedTrace = traceId.trim();
  if (!normalizedTrace) {
    throw new Error('trace_id is required.');
  }

  const [toolCallsPayload, tracePayload] = await Promise.all([
    fetchJson(`/slavik/tool-calls/${encodeURIComponent(normalizedTrace)}`),
    fetchJson(`/slavik/trace/${encodeURIComponent(normalizedTrace)}`),
  ]);

  const toolCalls = parseRecordArray(
    (toolCallsPayload as { tool_calls?: unknown }).tool_calls,
  );
  const events = parseRecordArray((tracePayload as { events?: unknown }).events);

  return {
    traceId: normalizedTrace,
    toolCalls,
    events,
    fetchedAt: Date.now(),
  };
};
