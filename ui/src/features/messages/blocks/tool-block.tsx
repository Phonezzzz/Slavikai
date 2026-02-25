import { useEffect, useMemo, useState } from 'react';
import { ChevronDown, Wrench } from 'lucide-react';

import { fetchTraceDiagnostics, type TraceDiagnostics } from '../trace-runtime';
import { DetailsPanel } from './details-panel';

type ToolBlockProps = {
  traceId: string;
  summary: string;
  open: boolean;
  onToggle: () => void;
  report: Record<string, unknown> | null;
};

const readString = (value: unknown): string | null => {
  if (typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
};

const getToolCallTitle = (call: Record<string, unknown>, index: number): string => {
  const name =
    readString(call.tool_name)
    ?? readString(call.tool)
    ?? readString(call.name)
    ?? `tool #${index + 1}`;
  const status = readString(call.status);
  return status ? `${name} (${status})` : name;
};

const readPayload = (call: Record<string, unknown>): unknown => {
  if (call.args !== undefined) {
    return call.args;
  }
  if (call.arguments !== undefined) {
    return call.arguments;
  }
  if (call.input !== undefined) {
    return call.input;
  }
  return null;
};

export function ToolBlock({ traceId, summary, open, onToggle, report }: ToolBlockProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [diagnostics, setDiagnostics] = useState<TraceDiagnostics | null>(null);

  useEffect(() => {
    if (!open || diagnostics || loading) {
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    void fetchTraceDiagnostics(traceId)
      .then((payload) => {
        if (!cancelled) {
          setDiagnostics(payload);
        }
      })
      .catch((reason) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : 'Failed to load trace details.');
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [diagnostics, loading, open, traceId]);

  const detailBody = useMemo(() => {
    const toolCalls = diagnostics?.toolCalls ?? [];
    const events = diagnostics?.events ?? [];
    if (loading) {
      return <div className="message-details-empty">Loading trace details...</div>;
    }
    if (error) {
      return <div className="message-details-error">{error}</div>;
    }
    if (toolCalls.length === 0 && events.length === 0 && !report) {
      return <div className="message-details-empty">No details.</div>;
    }

    return (
      <div className="message-details-stack">
        {toolCalls.map((call, index) => {
          const title = getToolCallTitle(call, index);
          const args = readPayload(call);
          const command = call.command;
          const stdout = call.stdout;
          const stderr = call.stderr;
          const errorValue = call.error;
          const result = call.result;

          return (
            <div key={`${traceId}-${index}`} className="message-diagnostic-entry">
              <div className="message-diagnostic-entry-title">{title}</div>
              {command !== undefined ? (
                <pre className="message-diagnostic-pre">{JSON.stringify(command, null, 2)}</pre>
              ) : null}
              {args !== null ? (
                <pre className="message-diagnostic-pre">{JSON.stringify(args, null, 2)}</pre>
              ) : null}
              {stdout !== undefined ? (
                <pre className="message-diagnostic-pre">{JSON.stringify(stdout, null, 2)}</pre>
              ) : null}
              {stderr !== undefined ? (
                <pre className="message-diagnostic-pre">{JSON.stringify(stderr, null, 2)}</pre>
              ) : null}
              {errorValue !== undefined ? (
                <pre className="message-diagnostic-pre">{JSON.stringify(errorValue, null, 2)}</pre>
              ) : null}
              {result !== undefined ? (
                <pre className="message-diagnostic-pre">{JSON.stringify(result, null, 2)}</pre>
              ) : null}
            </div>
          );
        })}

        {report ? (
          <div className="message-diagnostic-entry">
            <div className="message-diagnostic-entry-title">MWV report</div>
            <pre className="message-diagnostic-pre">{JSON.stringify(report, null, 2)}</pre>
          </div>
        ) : null}

        {events.length > 0 ? (
          <div className="message-diagnostic-entry">
            <div className="message-diagnostic-entry-title">Trace events</div>
            <pre className="message-diagnostic-pre">{JSON.stringify(events, null, 2)}</pre>
          </div>
        ) : null}
      </div>
    );
  }, [diagnostics?.events, diagnostics?.toolCalls, error, loading, report, traceId]);

  return (
    <div className="message-diagnostic message-diagnostic--tool">
      <div className="message-diagnostic-summary">
        <div className="message-diagnostic-left">
          <Wrench className="h-3.5 w-3.5" />
          <span>Tool run</span>
          <span className="message-diagnostic-divider" />
          <span className="message-diagnostic-text">{summary}</span>
        </div>
        <button
          type="button"
          onClick={onToggle}
          className="message-diagnostic-toggle"
          aria-expanded={open}
          aria-label={open ? 'Hide tool details' : 'Show tool details'}
        >
          <span>Details</span>
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>
      <DetailsPanel open={open}>{detailBody}</DetailsPanel>
    </div>
  );
}
