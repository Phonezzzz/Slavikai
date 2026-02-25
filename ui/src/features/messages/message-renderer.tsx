import { useMemo, useState } from 'react';
import { FileText } from 'lucide-react';

import { buildMessageBlocks } from './block-factory';
import { CodeBlock } from './blocks/code-block';
import { DecisionBlock } from './blocks/decision-block';
import { TextBlock } from './blocks/text-block';
import { ToolBlock } from './blocks/tool-block';
import { VerifierBlock } from './blocks/verifier-block';
import type { MessageBlock, MessageRendererProps } from './types';

const detailsKey = (messageId: string, blockId: string): string => `${messageId}:${blockId}`;

const isRenderableTextBlock = (block: MessageBlock): block is Extract<MessageBlock, { kind: 'text' }> => {
  return block.kind === 'text';
};

const isRenderableCodeBlock = (block: MessageBlock): block is Extract<MessageBlock, { kind: 'code' }> => {
  return block.kind === 'code';
};

const isRenderableToolBlock = (block: MessageBlock): block is Extract<MessageBlock, { kind: 'tool' }> => {
  return block.kind === 'tool';
};

const isRenderableVerifierBlock = (
  block: MessageBlock,
): block is Extract<MessageBlock, { kind: 'verifier' }> => {
  return block.kind === 'verifier';
};

export function MessageRenderer({
  context,
  message,
  decisionBusy = false,
  decisionError = null,
  onDecisionRespond,
}: MessageRendererProps) {
  const [detailsOpenByKey, setDetailsOpenByKey] = useState<Record<string, boolean>>({});

  const blocks = useMemo(() => buildMessageBlocks(context, message), [context, message]);

  if (message.kind === 'decision') {
    return (
      <DecisionBlock
        decision={message.decision}
        busy={decisionBusy}
        error={decisionError}
        onRespond={onDecisionRespond}
      />
    );
  }

  const isUser = message.message.role === 'user';
  const attachments = message.message.attachments ?? [];

  const toggleDetails = (blockId: string) => {
    const key = detailsKey(message.message.messageId, blockId);
    setDetailsOpenByKey((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  return (
    <div
      className={`message-renderer message-renderer--${context} ${
        isUser ? 'message-renderer--user' : 'message-renderer--assistant'
      }`}
    >
      <div className="message-card">
        {blocks.map((block) => {
          if (isRenderableTextBlock(block)) {
            return <TextBlock key={block.id} context={context} markdown={block.markdown} />;
          }
          if (isRenderableCodeBlock(block)) {
            return (
              <CodeBlock
                key={block.id}
                context={context}
                language={block.language}
                code={block.code}
                isFinal={block.isFinal}
              />
            );
          }
          if (isRenderableToolBlock(block)) {
            const key = detailsKey(message.message.messageId, block.id);
            return (
              <ToolBlock
                key={block.id}
                traceId={block.traceId}
                summary={block.summary}
                report={block.report}
                open={Boolean(detailsOpenByKey[key])}
                onToggle={() => toggleDetails(block.id)}
              />
            );
          }
          if (isRenderableVerifierBlock(block)) {
            const key = detailsKey(message.message.messageId, block.id);
            return (
              <VerifierBlock
                key={block.id}
                summary={block.summary}
                verifier={block.verifier}
                traceId={block.traceId}
                open={Boolean(detailsOpenByKey[key])}
                onToggle={() => toggleDetails(block.id)}
              />
            );
          }
          return null;
        })}

        {attachments.length > 0 ? (
          <div className="message-attachments-row">
            {attachments.map((attachment, index) => (
              <div
                key={`${attachment.name}-${index}`}
                className="message-attachment"
                title={`${attachment.name} (${attachment.mime})`}
              >
                <FileText className="h-3.5 w-3.5" />
                <span className="truncate">{attachment.name}</span>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
