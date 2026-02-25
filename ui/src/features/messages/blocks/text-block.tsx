import { MarkdownRenderer } from '../markdown-renderer';
import type { MessageRenderContext } from '../types';

type TextBlockProps = {
  context: MessageRenderContext;
  markdown: string;
};

export function TextBlock({ context, markdown }: TextBlockProps) {
  return <MarkdownRenderer context={context} markdown={markdown} />;
}
