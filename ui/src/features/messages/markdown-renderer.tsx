import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';
import remarkGfm from 'remark-gfm';

import type { MessageRenderContext } from './types';

type MarkdownRendererProps = {
  context: MessageRenderContext;
  markdown: string;
};

const isExternalLink = (href: string): boolean => /^https?:\/\//i.test(href);

export function MarkdownRenderer({ context, markdown }: MarkdownRendererProps) {
  return (
    <div className={`message-markdown message-markdown--${context}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize]}
        components={{
          a: ({ href, children, ...props }) => {
            const normalizedHref = typeof href === 'string' ? href : '';
            const external = isExternalLink(normalizedHref);
            return (
              <a
                {...props}
                href={normalizedHref || undefined}
                target={external ? '_blank' : undefined}
                rel={external ? 'noopener noreferrer' : undefined}
              >
                {children}
              </a>
            );
          },
          code: ({ className, children, ...props }) => {
            const languageClass = typeof className === 'string' ? className : '';
            const content = String(children ?? '');
            if (languageClass.includes('language-') && content.includes('\n')) {
              return (
                <code {...props} className="message-markdown-inline-code">
                  {content}
                </code>
              );
            }
            return (
              <code {...props} className="message-markdown-inline-code">
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="message-markdown-pre">{children}</pre>
          ),
          table: ({ children }) => (
            <div className="message-markdown-table-wrap">
              <table className="message-markdown-table">{children}</table>
            </div>
          ),
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
