import { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ShieldCheck } from 'lucide-react';

import { fetchTraceDiagnostics, type TraceDiagnostics } from '../trace-runtime';
import { DetailsPanel } from './details-panel';

type VerifierBlockProps = {
  summary: string;
  verifier: Record<string, unknown> | null;
  traceId: string | null;
  open: boolean;
  onToggle: () => void;
};

export function VerifierBlock({
  summary,
  verifier,
  traceId,
  open,
  onToggle,
}: VerifierBlockProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [diagnostics, setDiagnostics] = useState<TraceDiagnostics | null>(null);

  useEffect(() => {
    if (!open || !traceId || diagnostics || loading) {
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
          setError(reason instanceof Error ? reason.message : 'Failed to load verifier details.');
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

  const detailsBody = useMemo(() => {
    if (loading) {
      return <div className="message-details-empty">Loading verifier details...</div>;
    }
    if (error) {
      return <div className="message-details-error">{error}</div>;
    }
    if (!verifier && !(diagnostics?.events?.length)) {
      return <div className="message-details-empty">No details.</div>;
    }

    return (
      <div className="message-details-stack">
        {verifier ? (
          <div className="message-diagnostic-entry">
            <div className="message-diagnostic-entry-title">Verifier payload</div>
            <pre className="message-diagnostic-pre">{JSON.stringify(verifier, null, 2)}</pre>
          </div>
        ) : null}

        {diagnostics?.events && diagnostics.events.length > 0 ? (
          <div className="message-diagnostic-entry">
            <div className="message-diagnostic-entry-title">Trace events</div>
            <pre className="message-diagnostic-pre">{JSON.stringify(diagnostics.events, null, 2)}</pre>
          </div>
        ) : null}
      </div>
    );
  }, [diagnostics?.events, error, loading, verifier]);

  return (
    <div className="message-diagnostic message-diagnostic--verifier">
      <div className="message-diagnostic-summary">
        <div className="message-diagnostic-left">
          <ShieldCheck className="h-3.5 w-3.5" />
          <span>Verifier</span>
          <span className="message-diagnostic-divider" />
          <span className="message-diagnostic-text">{summary}</span>
        </div>
        <button
          type="button"
          onClick={onToggle}
          className="message-diagnostic-toggle"
          aria-expanded={open}
          aria-label={open ? 'Hide verifier details' : 'Show verifier details'}
        >
          <span>Details</span>
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>
      <DetailsPanel open={open}>{detailsBody}</DetailsPanel>
    </div>
  );
}
