import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, Copy } from 'lucide-react';

import type { MessageRenderContext } from '../types';

type CodeBlockProps = {
  context: MessageRenderContext;
  language: string | null;
  code: string;
  isFinal: boolean;
};

export function CodeBlock({ context, language, code, isFinal }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const [highlightedHtml, setHighlightedHtml] = useState<string | null>(null);
  const [stableMinHeight, setStableMinHeight] = useState<number | null>(null);
  const plainRef = useRef<HTMLPreElement | null>(null);

  const displayCode = useMemo(() => code.replace(/\r\n/g, '\n'), [code]);

  useEffect(() => {
    if (!isFinal || highlightedHtml) {
      return;
    }
    const plainElement = plainRef.current;
    if (!plainElement) {
      return;
    }
    const measuredHeight = Math.ceil(plainElement.getBoundingClientRect().height);
    if (measuredHeight > 0) {
      setStableMinHeight(measuredHeight);
    }
  }, [highlightedHtml, isFinal]);

  useEffect(() => {
    let cancelled = false;
    if (!isFinal) {
      setHighlightedHtml(null);
      setStableMinHeight(null);
      return () => {
        cancelled = true;
      };
    }

    void import('../shiki-cache')
      .then((module) =>
        module.highlightCodeWithCache({
          context,
          code: displayCode,
          language,
        }),
      )
      .then((html) => {
        if (!cancelled) {
          setHighlightedHtml(html);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHighlightedHtml(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [context, displayCode, isFinal, language]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(displayCode);
      setCopied(true);
      window.setTimeout(() => {
        setCopied(false);
      }, 1000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="message-code-block group">
      <button
        type="button"
        onClick={() => {
          void handleCopy();
        }}
        className="message-code-copy"
        title="Copy code"
        aria-label="Copy code"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-emerald-300" /> : <Copy className="h-3.5 w-3.5" />}
      </button>

      <div
        className="message-code-content"
        style={stableMinHeight !== null ? { minHeight: `${stableMinHeight}px` } : undefined}
      >
        {highlightedHtml ? (
          <div
            className="message-code-html"
            dangerouslySetInnerHTML={{ __html: highlightedHtml }}
          />
        ) : (
          <pre ref={plainRef} className="message-code-plain">
            <code>{displayCode}</code>
          </pre>
        )}
      </div>
    </div>
  );
}
