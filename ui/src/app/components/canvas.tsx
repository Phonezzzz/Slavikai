import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import {
  Send,
  Copy,
  ChevronDown,
  Paperclip,
  Mic,
  Bot,
  User,
  Check,
} from "lucide-react";

// ====== Types ======

interface CodeBlock {
  language: string;
  code: string;
}

type MessageSection =
  | { type: "text"; content: string }
  | { type: "code"; codeBlock: CodeBlock };

export interface CanvasMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
}

const parseContentSections = (content: string): MessageSection[] => {
  if (!content.includes("```")) {
    return [{ type: "text", content }];
  }
  const parts = content.split("```");
  const sections: MessageSection[] = [];
  parts.forEach((part, index) => {
    if (index % 2 === 0) {
      if (part.trim()) {
        sections.push({ type: "text", content: part.trim() });
      }
      return;
    }
    const lines = part.split("\n");
    const language = lines[0]?.trim() || "text";
    const code = lines.slice(1).join("\n").trimEnd();
    sections.push({ type: "code", codeBlock: { language, code } });
  });
  return sections.length > 0 ? sections : [{ type: "text", content }];
};

interface CanvasProps {
  messages?: CanvasMessage[];
  pendingMessage?: CanvasMessage | null;
  sending?: boolean;
  onSendMessage?: (message: string) => void;
  className?: string;
  modelName?: string;
  onOpenSettings?: () => void;
  statusMessage?: string | null;
  modelOptions?: Array<{
    value: string;
    label: string;
    provider: string;
    model: string;
  }>;
  selectedModelValue?: string | null;
  onSelectModel?: (provider: string, model: string) => void;
  modelsLoading?: boolean;
  savingModel?: boolean;
}

// ====== Sub Components ======

function CodeBlockRenderer({ codeBlock }: { codeBlock: CodeBlock }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(codeBlock.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Simple syntax highlighting
  const highlightCode = (code: string, lang: string) => {
    const keywords =
      lang === "python"
        ? [
            "class",
            "def",
            "import",
            "from",
            "return",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "in",
            "not",
            "and",
            "or",
            "True",
            "False",
            "None",
            "print",
            "str",
            "int",
            "list",
            "dict",
          ]
        : [
            "const",
            "let",
            "var",
            "function",
            "return",
            "if",
            "else",
            "for",
            "while",
            "import",
            "from",
            "export",
            "default",
            "class",
            "new",
            "true",
            "false",
            "null",
            "undefined",
          ];

    return code.split("\n").map((line, i) => {
      let highlighted = line;

      // Comments
      const commentIdx = line.indexOf("#");
      const jsCommentIdx = line.indexOf("//");
      const cIdx = commentIdx >= 0 ? commentIdx : jsCommentIdx;

      if (cIdx >= 0) {
        const before = line.substring(0, cIdx);
        const comment = line.substring(cIdx);
        highlighted = before;
        return (
          <div key={i} className="flex">
            <span className="text-[#666] select-none mr-4 text-right w-6 inline-block">
              {i + 1}
            </span>
            <span>
              <HighlightLine text={highlighted} keywords={keywords} />
              <span className="text-[#6a7a5a]">{comment}</span>
            </span>
          </div>
        );
      }

      return (
        <div key={i} className="flex">
          <span className="text-[#666] select-none mr-4 text-right w-6 inline-block">
            {i + 1}
          </span>
          <HighlightLine text={line} keywords={keywords} />
        </div>
      );
    });
  };

  return (
    <div className="rounded-lg overflow-hidden bg-[#0d0d10] border border-[#2a2a2e] my-2">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#18181c] border-b border-[#2a2a2e]">
        <span className="text-[12px] text-[#888]">{codeBlock.language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-[12px] text-[#666] hover:text-[#ccc] transition-colors cursor-pointer"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-green-400" />
              <span className="text-green-400">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              Copy
            </>
          )}
        </button>
      </div>
      {/* Code */}
      <div className="p-4 overflow-x-auto">
        <pre className="text-[13px] leading-relaxed font-mono">
          {highlightCode(codeBlock.code, codeBlock.language)}
        </pre>
      </div>
    </div>
  );
}

function HighlightLine({
  text,
  keywords,
}: {
  text: string;
  keywords: string[];
}) {
  // Very simple keyword highlighting
  const parts = text.split(/(\s+|[.,:;()[\]{}=|"'])/);
  return (
    <span>
      {parts.map((part, i) => {
        if (keywords.includes(part)) {
          return (
            <span key={i} className="text-[#c792ea]">
              {part}
            </span>
          );
        }
        // Strings
        if (part.startsWith('"') || part.startsWith("'")) {
          return (
            <span key={i} className="text-[#c3e88d]">
              {part}
            </span>
          );
        }
        // Numbers
        if (/^\d+$/.test(part)) {
          return (
            <span key={i} className="text-[#f78c6c]">
              {part}
            </span>
          );
        }
        return (
          <span key={i} className="text-[#d4d4d8]">
            {part}
          </span>
        );
      })}
    </span>
  );
}

function MessageBubble({ message }: { message: CanvasMessage }) {
  const isUser = message.role === "user";
  const sections = useMemo(() => parseContentSections(message.content), [message.content]);

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center ${
          isUser
            ? "bg-[#6366f1]/20 border border-[#6366f1]/30"
            : "bg-[#2a2a30] border border-[#3a3a42]"
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-[#818cf8]" />
        ) : (
          <Bot className="w-4 h-4 text-[#888]" />
        )}
      </div>

      {/* Content */}
      <div className={`flex-1 max-w-[calc(100%-50px)] ${isUser ? "text-right" : ""}`}>
        {sections.map((section, idx) => {
          switch (section.type) {
            case "text":
              return (
                <p
                  key={idx}
                  className={`whitespace-pre-wrap text-[14px] leading-relaxed text-[#c8c8cc] my-1 ${
                    isUser ? "text-right" : ""
                  }`}
                >
                  {section.content}
                </p>
              );
            case "code":
              return section.codeBlock ? (
                <CodeBlockRenderer key={idx} codeBlock={section.codeBlock} />
              ) : null;
            default:
              return null;
          }
        })}
      </div>
    </div>
  );
}

// ====== Main Canvas Component ======

export function Canvas({
  messages = [],
  pendingMessage = null,
  sending = false,
  onSendMessage,
  className = "",
  modelName = "Model not selected",
  onOpenSettings,
  statusMessage = null,
  modelOptions = [],
  selectedModelValue = null,
  onSelectModel,
  modelsLoading = false,
  savingModel = false,
}: CanvasProps) {
  const [inputValue, setInputValue] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const displayMessages = useMemo(() => {
    if (pendingMessage) {
      return [...messages, pendingMessage];
    }
    return messages;
  }, [messages, pendingMessage]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [displayMessages]);

  const handleSend = () => {
    if (sending) {
      return;
    }
    if (inputValue.trim()) {
      onSendMessage?.(inputValue.trim());
      setInputValue("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div
      className={`flex flex-col h-full bg-transparent ${className}`}
    >
      {/* Model selector header */}
      <div className="flex items-center justify-center py-3 border-b border-[#1e1e22]">
        {modelOptions.length > 0 ? (
          <div className="relative">
            <select
              value={selectedModelValue ?? ""}
              onChange={(event) => {
                const next = modelOptions.find((option) => option.value === event.target.value);
                if (next) {
                  onSelectModel?.(next.provider, next.model);
                }
              }}
              disabled={modelsLoading || savingModel}
              className="appearance-none bg-[#1e1e22] text-[#aaa] text-[13px] px-3 py-1.5 rounded-lg border border-[#2a2a2e] hover:border-[#3a3a42] transition-colors cursor-pointer pr-8"
            >
              <option value="" disabled>
                Select model
              </option>
              {modelOptions.map((option) => (
                <option key={option.value} value={option.value} className="bg-[#141418] text-[#ddd]">
                  {option.label}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[#666]" />
          </div>
        ) : (
          <button
            onClick={onOpenSettings}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-[#1e1e22] transition-colors cursor-pointer"
          >
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-[13px] text-[#aaa]">{modelName}</span>
            <ChevronDown className="w-3.5 h-3.5 text-[#666]" />
          </button>
        )}
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto" data-scrollbar>
        <div className="max-w-3xl mx-auto px-6 py-6 space-y-8">
          {displayMessages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          {sending ? (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-[#2a2a30] border border-[#3a3a42]">
                <Bot className="w-4 h-4 text-[#888]" />
              </div>
              <div className="flex-1">
                <div className="text-[14px] text-[#999]">Thinking…</div>
              </div>
            </div>
          ) : null}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <div className="border-t border-[#1e1e22] px-4 py-3">
        <div className="max-w-3xl mx-auto">
          {statusMessage ? (
            <div className="mb-2 rounded-lg border border-[#2a2a2e] bg-[#1e1e22] px-3 py-2 text-[12px] text-[#c0c0c0]">
              {statusMessage}
            </div>
          ) : null}
          <div className="flex items-end gap-2 bg-[#1e1e22] rounded-xl border border-[#2a2a2e] focus-within:border-[#3a3a42] transition-colors px-4 py-3">
            {/* Attachment button */}
            <button className="text-[#555] hover:text-[#999] transition-colors pb-0.5 cursor-pointer">
              <Paperclip className="w-4.5 h-4.5" />
            </button>

            {/* Textarea */}
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message... (Shift+Enter for new line)"
              rows={1}
              className="flex-1 bg-transparent text-[14px] text-[#d4d4d8] placeholder-[#555] resize-none outline-none min-h-[24px] max-h-[120px]"
              style={{ lineHeight: "24px" }}
              disabled={sending}
            />

            {/* Mic button */}
            <button className="text-[#555] hover:text-[#999] transition-colors pb-0.5 cursor-pointer">
              <Mic className="w-4.5 h-4.5" />
            </button>

            {/* Send button */}
            <button
              onClick={handleSend}
              disabled={!inputValue.trim() || sending}
              className={`p-1.5 rounded-lg transition-all cursor-pointer ${
                inputValue.trim() && !sending
                  ? "bg-[#6366f1] hover:bg-[#5558e6] text-white"
                  : "bg-[#2a2a2e] text-[#555]"
              }`}
            >
              <Send className="w-4 h-4" />
            </button>
          </div>

          <p className="text-[11px] text-[#444] text-center mt-2">
            SlavikAI v1.0 - Python Agent
          </p>
        </div>
      </div>
    </div>
  );
}
