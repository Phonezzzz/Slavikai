import type { MessageRuntimeMeta } from '../../app/types';
import type {
  CodeMessageBlock,
  MessageBlock,
  MessageRenderContext,
  RenderableMessage,
  TextMessageBlock,
  VerifierMessageBlock,
} from './types';

const CODE_FENCE_PATTERN = /(?:^|\n|\\n)```([a-zA-Z0-9_-]{0,32})(?:\n|\\n)([\s\S]*?)```(?=\n|\\n|$)/g;

const normalizeEscapedMarkdown = (value: string): string => {
  if (!value.includes('\\n') || !value.includes('```')) {
    return value;
  }
  if (!/```[a-zA-Z0-9_-]*\\n/.test(value)) {
    return value;
  }
  return value.replace(/\\n/g, '\n');
};

const createTextBlock = (id: string, markdown: string): TextMessageBlock | null => {
  if (!markdown.trim()) {
    return null;
  }
  return {
    kind: 'text',
    id,
    markdown,
  };
};

const createCodeBlock = (
  id: string,
  language: string,
  code: string,
  isFinal: boolean,
): CodeMessageBlock => {
  const normalizedLanguage = language.trim().toLowerCase();
  return {
    kind: 'code',
    id,
    language: /^[a-z0-9_-]{1,32}$/.test(normalizedLanguage) ? normalizedLanguage : null,
    code,
    isFinal,
  };
};

const parseMarkdownBlocks = (markdown: string, isFinal: boolean): Array<TextMessageBlock | CodeMessageBlock> => {
  const source = normalizeEscapedMarkdown(markdown);
  const output: Array<TextMessageBlock | CodeMessageBlock> = [];
  const pattern = new RegExp(CODE_FENCE_PATTERN.source, 'g');
  let lastIndex = 0;
  let sequence = 0;
  let match: RegExpExecArray | null = pattern.exec(source);

  while (match) {
    const [fullMatch, languageRaw, codeRaw] = match;
    const prefixOffset = fullMatch.startsWith('\n') ? 1 : fullMatch.startsWith('\\n') ? 2 : 0;
    const blockStart = match.index + prefixOffset;
    const textBefore = source.slice(lastIndex, blockStart);
    const textBlock = createTextBlock(`text-${sequence}`, textBefore.trim());
    if (textBlock) {
      output.push(textBlock);
      sequence += 1;
    }

    const code = (codeRaw ?? '').replace(/\\n/g, '\n').replace(/\r\n/g, '\n').trimEnd();
    output.push(createCodeBlock(`code-${sequence}`, languageRaw ?? '', code, isFinal));
    sequence += 1;

    lastIndex = match.index + fullMatch.length;
    match = pattern.exec(source);
  }

  const tail = source.slice(lastIndex);
  const tailBlock = createTextBlock(`text-${sequence}`, tail.trim());
  if (tailBlock) {
    output.push(tailBlock);
  }

  if (output.length === 0) {
    output.push({
      kind: 'text',
      id: 'text-0',
      markdown: source,
    });
  }

  return output;
};

const getToolSummary = (meta: MessageRuntimeMeta | null): string => {
  const executionSummary = meta?.mwvReport?.execution_summary;
  if (typeof executionSummary === 'string' && executionSummary.trim()) {
    return executionSummary.trim();
  }
  const stopCode = meta?.mwvReport?.stop_reason_code;
  if (typeof stopCode === 'string' && stopCode.trim()) {
    return `Trace captured (${stopCode.trim()})`;
  }
  return 'Tool activity available for this response';
};

const getVerifierSummary = (verifier: Record<string, unknown> | null): string => {
  if (!verifier) {
    return 'Verifier details available';
  }
  const status = verifier.status;
  const duration = verifier.duration_ms;
  const statusText = typeof status === 'string' && status.trim() ? status.trim().toUpperCase() : 'UNKNOWN';
  const durationText = typeof duration === 'number' && Number.isFinite(duration)
    ? ` in ${Math.max(0, Math.round(duration))} ms`
    : '';
  return `Verifier ${statusText}${durationText}`;
};

const buildVerifierBlock = (meta: MessageRuntimeMeta | null): VerifierMessageBlock | null => {
  const rawVerifier = meta?.mwvReport?.verifier;
  if (!rawVerifier || typeof rawVerifier !== 'object' || Array.isArray(rawVerifier)) {
    return null;
  }
  const verifier = rawVerifier as Record<string, unknown>;
  return {
    kind: 'verifier',
    id: 'verifier-0',
    summary: getVerifierSummary(verifier),
    verifier,
    traceId: meta?.traceId ?? null,
  };
};

export const buildMessageBlocks = (
  context: MessageRenderContext,
  message: RenderableMessage,
): MessageBlock[] => {
  if (message.kind === 'decision') {
    return [
      {
        kind: 'decision',
        id: `decision-${message.id}`,
        decision: message.decision,
      },
    ];
  }

  const output: MessageBlock[] = [];
  const runtimeMeta = message.meta ?? message.message.runtimeMeta ?? null;
  const isFinal = runtimeMeta?.isFinal ?? !message.message.transient;
  const isAssistant = message.message.role === 'assistant';

  if (context === 'workspace' && isAssistant && runtimeMeta?.traceId) {
    output.push({
      kind: 'tool',
      id: 'tool-0',
      traceId: runtimeMeta.traceId,
      summary: getToolSummary(runtimeMeta),
      report: runtimeMeta.mwvReport,
    });
  }

  if (context === 'workspace' && isAssistant) {
    const verifierBlock = buildVerifierBlock(runtimeMeta);
    if (verifierBlock) {
      output.push(verifierBlock);
    }
  }

  output.push(...parseMarkdownBlocks(message.message.content, isFinal));
  return output;
};
