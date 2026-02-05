import { useEffect, useMemo, useRef, useState } from "react";

type ChatInputProps = {
  value: string;
  onChange: (value: string) => void;
  onSend: (contentOverride?: string) => void;
  sending: boolean;
};

type InputAttachment = {
  name: string;
  size: number;
  type: string;
  preview: string;
};

type SpeechRecognitionAlternativeLike = {
  transcript: string;
};

type SpeechRecognitionResultLike = {
  isFinal: boolean;
  length: number;
  [index: number]: SpeechRecognitionAlternativeLike;
};

type SpeechRecognitionEventLike = Event & {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
};

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: Event) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

type SpeechWindow = Window & {
  SpeechRecognition?: SpeechRecognitionCtor;
  webkitSpeechRecognition?: SpeechRecognitionCtor;
};

const quickTools: Array<{ label: string; snippet: string }> = [
  { label: "Find", snippet: "/project find " },
  { label: "Index", snippet: "/project index ." },
  { label: "Trace", snippet: "/trace" },
  { label: "Plan", snippet: "/plan " },
];

const MAX_ATTACHMENTS = 6;
const MAX_PREVIEW_CHARS = 4000;

const canReadAsText = (file: File): boolean => {
  if (file.type.startsWith("text/")) {
    return true;
  }
  return /\.(txt|md|json|yaml|yml|toml|csv|py|ts|tsx|js|jsx|css|html|xml)$/i.test(file.name);
};

const formatSize = (bytes: number): string => {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const serializeAttachments = (attachments: InputAttachment[]): string => {
  const lines: string[] = ["[Attached files]"];
  attachments.forEach((item, index) => {
    const header = `#${index + 1} ${item.name} (${item.type || "file"}, ${formatSize(item.size)})`;
    lines.push(header);
    if (item.preview.trim()) {
      lines.push(item.preview.trim());
    } else {
      lines.push("(empty preview)");
    }
    lines.push("---");
  });
  return lines.join("\n");
};

const readAttachment = async (file: File): Promise<InputAttachment> => {
  if (!canReadAsText(file)) {
    return {
      name: file.name,
      size: file.size,
      type: file.type || "binary",
      preview: "[binary file attached]",
    };
  }
  try {
    const text = await file.text();
    return {
      name: file.name,
      size: file.size,
      type: file.type || "text/plain",
      preview: text.slice(0, MAX_PREVIEW_CHARS),
    };
  } catch {
    return {
      name: file.name,
      size: file.size,
      type: file.type || "text/plain",
      preview: "[failed to read file preview]",
    };
  }
};

export default function ChatInput({ value, onChange, onSend, sending }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const latestValueRef = useRef(value);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const [attachments, setAttachments] = useState<InputAttachment[]>([]);
  const [recording, setRecording] = useState(false);

  useEffect(() => {
    latestValueRef.current = value;
    const element = textareaRef.current;
    if (!element) {
      return;
    }
    element.style.height = "";
    element.style.height = `${element.scrollHeight}px`;
  }, [value]);

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
      recognitionRef.current = null;
    };
  }, []);

  const speechCtor = useMemo(() => {
    const browserWindow = window as SpeechWindow;
    return browserWindow.SpeechRecognition ?? browserWindow.webkitSpeechRecognition ?? null;
  }, []);

  const sendDisabled = sending || (value.trim().length === 0 && attachments.length === 0);

  const handleAttachFiles = async (fileList: FileList | null) => {
    if (!fileList) {
      return;
    }
    const incoming = Array.from(fileList).slice(0, MAX_ATTACHMENTS);
    const parsed = await Promise.all(incoming.map((file) => readAttachment(file)));
    setAttachments((prev) => [...prev, ...parsed].slice(0, MAX_ATTACHMENTS));
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleToggleRecording = () => {
    if (!speechCtor) {
      return;
    }
    if (recording) {
      recognitionRef.current?.stop();
      setRecording(false);
      return;
    }
    const recognition = new speechCtor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "ru-RU";
    recognition.onresult = (event) => {
      const pieces: string[] = [];
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const alternative = result?.[0];
        if (alternative && typeof alternative.transcript === "string") {
          pieces.push(alternative.transcript);
        }
      }
      const transcript = pieces.join(" ").trim();
      if (!transcript) {
        return;
      }
      const current = latestValueRef.current;
      const spacer = current.trim().length > 0 ? " " : "";
      onChange(`${current}${spacer}${transcript}`.trimStart());
    };
    recognition.onerror = () => {
      setRecording(false);
    };
    recognition.onend = () => {
      setRecording(false);
      recognitionRef.current = null;
    };
    recognitionRef.current = recognition;
    recognition.start();
    setRecording(true);
  };

  const handleSend = () => {
    const trimmed = value.trim();
    const attachmentBlock = attachments.length > 0 ? serializeAttachments(attachments) : "";
    const composed = trimmed
      ? attachmentBlock
        ? `${trimmed}\n\n${attachmentBlock}`
        : trimmed
      : attachmentBlock;
    if (!composed.trim()) {
      return;
    }
    onSend(composed);
    setAttachments([]);
  };

  return (
    <div className="rounded-3xl border border-neutral-800/80 bg-neutral-950/60 p-3">
      <div className="mb-2 flex flex-wrap gap-2">
        {quickTools.map((tool) => (
          <button
            key={tool.label}
            type="button"
            onClick={() => {
              const base = latestValueRef.current;
              const spacer = base.trim().length > 0 ? "\n" : "";
              onChange(`${base}${spacer}${tool.snippet}`);
            }}
            className="rounded-full border border-neutral-800 bg-neutral-900 px-2.5 py-1 text-[11px] uppercase tracking-wide text-neutral-300 hover:bg-neutral-800"
          >
            {tool.label}
          </button>
        ))}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="rounded-full border border-neutral-800 bg-neutral-900 px-2.5 py-1 text-[11px] uppercase tracking-wide text-neutral-300 hover:bg-neutral-800"
        >
          Attach files
        </button>
        <button
          type="button"
          onClick={handleToggleRecording}
          disabled={!speechCtor}
          className={`rounded-full border px-2.5 py-1 text-[11px] uppercase tracking-wide ${
            !speechCtor
              ? "border-neutral-800 bg-neutral-900 text-neutral-500"
              : recording
                ? "border-rose-400/70 bg-rose-500/20 text-rose-200"
                : "border-neutral-800 bg-neutral-900 text-neutral-300 hover:bg-neutral-800"
          }`}
        >
          {recording ? "Stop rec" : "Rec / STT"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={(event) => {
            void handleAttachFiles(event.target.files);
          }}
          className="hidden"
        />
      </div>

      {attachments.length > 0 ? (
        <div className="mb-2 flex flex-wrap gap-2">
          {attachments.map((item, index) => (
            <button
              key={`${item.name}-${index}`}
              type="button"
              onClick={() => {
                setAttachments((prev) => prev.filter((_, currentIndex) => currentIndex !== index));
              }}
              className="rounded-full border border-neutral-700 bg-neutral-900 px-3 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
            >
              {item.name} · {formatSize(item.size)} · remove
            </button>
          ))}
        </div>
      ) : null}

      <div className="flex flex-col gap-3 md:flex-row md:items-end">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              handleSend();
            }
          }}
          rows={1}
          placeholder="Send a message"
          className="min-h-[44px] flex-1 resize-none rounded-2xl border border-neutral-800/80 bg-neutral-900/60 px-4 py-3 text-sm text-neutral-100 outline-none focus:ring-2 focus:ring-neutral-500/60"
        />
        <button
          type="button"
          disabled={sendDisabled}
          onClick={handleSend}
          className="flex h-11 items-center justify-center rounded-2xl bg-neutral-200 px-6 text-sm font-semibold text-neutral-950 shadow-lg shadow-black/30 transition hover:bg-neutral-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {sending ? "Sending..." : "Send"}
        </button>
      </div>
      <div className="mt-2 text-xs text-neutral-500">
        Enter — отправить, Shift+Enter — новая строка, Attach/Rec — расширенный input.
      </div>
    </div>
  );
}
