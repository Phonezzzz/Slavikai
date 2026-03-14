import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ClipboardEvent,
  type KeyboardEvent,
} from "react";
import {
  Send,
  Copy,
  Edit2,
  RefreshCcw,
  Volume2,
  Square,
  ThumbsUp,
  ThumbsDown,
  ChevronDown,
  Paperclip,
  Mic,
  Check,
  PanelRight,
  LoaderCircle,
  X,
  FileText,
} from "lucide-react";

import BrainLogo from "../../assets/brain.png";
import { MessageRenderer } from "../../features/messages";
import type { RenderableMessage } from "../../features/messages";
import type { DecisionRespondChoice, MessageRuntimeMeta, UiDecision } from "../types";
import { DecisionPanel } from "./decision-panel";
import {
  MAX_COMPOSER_ATTACHMENTS,
  buildPastedTextAttachment,
  createComposerAttachmentId,
  readComposerAttachmentFromFile,
} from "../../features/composer/attachment-utils";

// ====== Types ======

export interface CanvasMessage {
  id: string;
  messageId: string;
  role: "user" | "assistant";
  content: string;
  createdAt?: string;
  traceId?: string | null;
  parentUserMessageId?: string | null;
  attachments?: Array<{ name: string; mime: string; content: string }>;
  transient?: boolean;
  runtimeMeta?: MessageRuntimeMeta | null;
}

export type CanvasComposerAttachment = {
  name: string;
  mime: string;
  content: string;
};

export type CanvasSendPayload = {
  content: string;
  attachments?: CanvasComposerAttachment[];
};

interface CanvasProps {
  messages?: CanvasMessage[];
  pendingMessage?: CanvasMessage | null;
  streamingAssistantMessage?: CanvasMessage | null;
  showAssistantLoading?: boolean;
  sending?: boolean;
  onSendMessage?: (payload: CanvasSendPayload) => Promise<boolean> | boolean | void;
  onSendFeedback?: (interactionId: string, rating: "good" | "bad") => Promise<boolean>;
  className?: string;
  modelName?: string;
  onOpenSettings?: () => void;
  statusMessage?: string | null;
  modelOptions?: Array<{
    value: string;
    label: string;
    provider: string;
    model: string;
    disabled?: boolean;
  }>;
  selectedModelValue?: string | null;
  onSelectModel?: (provider: string, model: string) => void;
  modelsLoading?: boolean;
  savingModel?: boolean;
  forceCanvasNext?: boolean;
  onToggleForceCanvasNext?: () => void;
  longPasteToFileEnabled?: boolean;
  longPasteThresholdChars?: number;
  decision?: UiDecision | null;
  decisionBusy?: boolean;
  decisionError?: string | null;
  onDecisionRespond?: (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
  ) => Promise<void> | void;
}

// ====== Sub Components ======

function MessageBubble({ message }: { message: CanvasMessage }) {
  const renderable: RenderableMessage = {
    kind: "message",
    message,
    meta: message.runtimeMeta ?? null,
  };

  return (
    <div className={`flex gap-3 ${message.role === "user" ? "flex-row-reverse" : ""}`}>
      <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-[#2a2a30] border border-[#3a3a42]">
        {message.role === "assistant" ? (
          <img
            src={BrainLogo}
            alt="SlavikAI"
            className="w-4 h-4 object-contain"
          />
        ) : (
          <span className="text-[11px] font-semibold text-[#aeb1ff]">YOU</span>
        )}
      </div>
      <div className={`flex-1 min-w-0 ${message.role === "user" ? "text-right" : ""}`}>
        <MessageRenderer context="chat" message={renderable} />
      </div>
    </div>
  );
}

function MessageActions({
  message,
  copied,
  speaking,
  feedbackRating,
  feedbackBusy,
  sending,
  canRefresh,
  canFeedback,
  onCopy,
  onEdit,
  onRefresh,
  onListenToggle,
  onLike,
  onDislike,
}: {
  message: CanvasMessage;
  copied: boolean;
  speaking: boolean;
  feedbackRating: "good" | "bad" | null;
  feedbackBusy: boolean;
  sending: boolean;
  canRefresh: boolean;
  canFeedback: boolean;
  onCopy: () => void;
  onEdit: () => void;
  onRefresh: () => void;
  onListenToggle: () => void;
  onLike: () => void;
  onDislike: () => void;
}) {
  const isUser = message.role === "user";
  const baseClass =
    "rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40";

  return (
    <div className={`mt-2 flex items-center gap-1 ${isUser ? "justify-end" : "justify-start"}`}>
      <button
        type="button"
        onClick={onCopy}
        className={baseClass}
        title="Copy"
        aria-label="Copy"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
      </button>

      {isUser ? (
        <button
          type="button"
          onClick={onEdit}
          className={baseClass}
          title="Edit"
          aria-label="Edit"
        >
          <Edit2 className="h-3.5 w-3.5" />
        </button>
      ) : null}

      {!isUser ? (
        <>
          <button
            type="button"
            onClick={onRefresh}
            disabled={!canRefresh || sending}
            className={baseClass}
            title={canRefresh ? "Refresh" : "Refresh unavailable"}
            aria-label="Refresh"
          >
            <RefreshCcw className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={onListenToggle}
            className={baseClass}
            title={speaking ? "Stop listen" : "Listen"}
            aria-label={speaking ? "Stop listen" : "Listen"}
          >
            {speaking ? <Square className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
          </button>
          <button
            type="button"
            onClick={onLike}
            disabled={!canFeedback || feedbackBusy}
            className={`${baseClass} ${feedbackRating === "good" ? "text-emerald-300" : ""}`}
            title={canFeedback ? "Like" : "Like unavailable"}
            aria-label="Like"
          >
            <ThumbsUp className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={onDislike}
            disabled={!canFeedback || feedbackBusy}
            className={`${baseClass} ${feedbackRating === "bad" ? "text-rose-300" : ""}`}
            title={canFeedback ? "Dislike" : "Dislike unavailable"}
            aria-label="Dislike"
          >
            <ThumbsDown className="h-3.5 w-3.5" />
          </button>
        </>
      ) : null}
    </div>
  );
}

function LoadingBubble() {
  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-[#2a2a30] border border-[#3a3a42]">
        <img
          src={BrainLogo}
          alt="SlavikAI"
          className="w-4 h-4 object-contain"
        />
      </div>
      <div className="flex-1 max-w-[calc(100%-50px)]">
        <div className="inline-flex items-center gap-2 rounded-lg border border-[#1f1f24] bg-[#111115] px-3 py-2">
          <LoaderCircle className="h-4 w-4 animate-spin text-[#8a8a90]" />
          <span className="text-[13px] text-[#b8b8be]">Подключаюсь к модели...</span>
        </div>
      </div>
    </div>
  );
}

// ====== Main Canvas Component ======

export function Canvas({
  messages = [],
  pendingMessage = null,
  streamingAssistantMessage = null,
  showAssistantLoading = false,
  sending = false,
  onSendMessage,
  onSendFeedback,
  className = "",
  modelName = "Model not selected",
  onOpenSettings,
  statusMessage = null,
  modelOptions = [],
  selectedModelValue = null,
  onSelectModel,
  modelsLoading = false,
  savingModel = false,
  forceCanvasNext = false,
  onToggleForceCanvasNext,
  longPasteToFileEnabled = true,
  longPasteThresholdChars = 12000,
  decision = null,
  decisionBusy = false,
  decisionError = null,
  onDecisionRespond,
}: CanvasProps) {
  const [inputValue, setInputValue] = useState("");
  const [composerAttachments, setComposerAttachments] = useState<
    Array<CanvasComposerAttachment & { id: string }>
  >([]);
  const [pasteUndo, setPasteUndo] = useState<{
    attachmentId: string;
    originalText: string;
  } | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);
  const [feedbackByMessageId, setFeedbackByMessageId] = useState<Record<string, "good" | "bad">>(
    {},
  );
  const [feedbackBusyMessageId, setFeedbackBusyMessageId] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [sttError, setSttError] = useState<string | null>(null);
  const [ttsNotice, setTtsNotice] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const caretSelectionRef = useRef<{ start: number; end: number }>({ start: 0, end: 0 });
  const displayMessages = useMemo(() => {
    const items = [...messages];
    if (pendingMessage) {
      items.push(pendingMessage);
    }
    if (streamingAssistantMessage) {
      items.push(streamingAssistantMessage);
    }
    return items;
  }, [messages, pendingMessage, streamingAssistantMessage]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [displayMessages]);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = "";
        audioRef.current = null;
      }
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
        audioUrlRef.current = null;
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    const lineHeightPx = 24;
    const maxHeight = lineHeightPx * 5;
    textarea.style.height = "auto";
    const nextHeight = Math.min(Math.max(textarea.scrollHeight, lineHeightPx), maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [inputValue]);

  const canUseMediaRecorder = useMemo(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return (
      typeof window.MediaRecorder !== "undefined"
      && !!window.navigator?.mediaDevices
      && typeof window.navigator.mediaDevices.getUserMedia === "function"
    );
  }, []);

  const effectiveLongPasteThreshold = useMemo(() => {
    const normalized = Number.isFinite(longPasteThresholdChars)
      ? Math.floor(longPasteThresholdChars)
      : 12000;
    return Math.max(1000, Math.min(80000, normalized));
  }, [longPasteThresholdChars]);
  const composerBlocked = decision?.status === "pending" && decision.blocking;

  const toComposerAttachments = (
    items: Array<{ name: string; mime: string; content: string }> | undefined,
  ): Array<CanvasComposerAttachment & { id: string }> => {
    if (!items || items.length === 0) {
      return [];
    }
    return items.map((item, index) => ({
      id: `composer-${Date.now()}-${index}`,
      name: item.name,
      mime: item.mime,
      content: item.content,
    }));
  };

  const buildMessageTextForCopy = (message: CanvasMessage): string => {
    const attachments = message.attachments ?? [];
    if (attachments.length === 0) {
      return message.content;
    }
    const lines: string[] = [];
    if (message.content.trim()) {
      lines.push(message.content.trim());
      lines.push("");
    }
    lines.push("[attachments]");
    attachments.forEach((attachment, index) => {
      lines.push(`#${index + 1} ${attachment.name} (${attachment.mime})`);
      lines.push(attachment.content);
      lines.push("---");
    });
    return lines.join("\n");
  };

  const insertTextIntoComposer = (rawText: string) => {
    const text = rawText.trim();
    if (!text) {
      return;
    }
    const textarea = textareaRef.current;
    const isActive =
      typeof document !== "undefined"
      && textarea !== null
      && document.activeElement === textarea;
    let nextCaret = 0;
    setInputValue((prev) => {
      const startBase = isActive
        ? textarea?.selectionStart ?? prev.length
        : prev.length;
      const endBase = isActive
        ? textarea?.selectionEnd ?? prev.length
        : prev.length;
      const start = Math.max(0, Math.min(startBase, prev.length));
      const end = Math.max(start, Math.min(endBase, prev.length));
      const prefix = prev.slice(0, start);
      const suffix = prev.slice(end);
      const spacerBefore = prefix.length > 0 && !/\s$/.test(prefix) ? " " : "";
      const spacerAfter = suffix.length > 0 && !/^\s/.test(suffix) ? " " : "";
      const insertion = `${spacerBefore}${text}${spacerAfter}`;
      nextCaret = prefix.length + insertion.length;
      return `${prefix}${insertion}${suffix}`;
    });
    if (isActive && textarea) {
      window.requestAnimationFrame(() => {
        textarea.focus();
        textarea.setSelectionRange(nextCaret, nextCaret);
        caretSelectionRef.current = { start: nextCaret, end: nextCaret };
      });
    }
  };

  const extractErrorMessage = (payload: unknown, fallback: string): string => {
    if (!payload || typeof payload !== "object") {
      return fallback;
    }
    const body = payload as { error?: { message?: unknown } };
    if (body.error && typeof body.error.message === "string" && body.error.message.trim()) {
      return body.error.message;
    }
    return fallback;
  };

  const stopPlayback = () => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    setSpeakingMessageId(null);
  };

  const pushComposerAttachments = (attachments: CanvasComposerAttachment[]): boolean => {
    if (attachments.length === 0) {
      return true;
    }
    let appended = false;
    let truncated = false;
    setComposerAttachments((prev) => {
      const remaining = MAX_COMPOSER_ATTACHMENTS - prev.length;
      if (remaining <= 0) {
        truncated = true;
        return prev;
      }
      const nextItems = attachments.slice(0, remaining).map((attachment) => ({
        id: createComposerAttachmentId("canvas-attachment"),
        ...attachment,
      }));
      truncated = attachments.length > remaining;
      appended = nextItems.length > 0;
      return [...prev, ...nextItems];
    });
    if (truncated) {
      setSttError("Достигнут лимит вложений в одном сообщении.");
    }
    return appended;
  };

  const appendFilesToComposer = async (files: File[]) => {
    if (files.length === 0) {
      return;
    }
    setSttError(null);
    try {
      const attachments = await Promise.all(files.map((file) => readComposerAttachmentFromFile(file)));
      pushComposerAttachments(attachments);
    } catch (error) {
      setSttError(error instanceof Error ? error.message : "Не удалось подготовить вложение.");
    }
  };

  const handleAttachFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    await appendFilesToComposer(files);
  };

  const handleSend = async () => {
    if (sending || isTranscribing) {
      return;
    }
    const trimmed = inputValue.trim();
    const attachmentsPayload: CanvasComposerAttachment[] = composerAttachments.map((item) => ({
      name: item.name,
      mime: item.mime,
      content: item.content,
    }));
    if (!trimmed && attachmentsPayload.length === 0) {
      return;
    }
    const previousInputValue = inputValue;
    const previousAttachments = composerAttachments;
    const previousPasteUndo = pasteUndo;
    setInputValue("");
    setComposerAttachments([]);
    setPasteUndo(null);
    setSttError(null);
    setTtsNotice(null);

    const sent = await onSendMessage?.({
      content: trimmed,
      attachments: attachmentsPayload.length > 0 ? attachmentsPayload : undefined,
    });
    if (sent === false) {
      setInputValue((current) => (current.trim().length === 0 ? previousInputValue : current));
      setComposerAttachments((current) => (current.length === 0 ? previousAttachments : current));
      setPasteUndo((current) => current ?? previousPasteUndo);
      return;
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  const handleCopyMessage = async (message: CanvasMessage) => {
    try {
      await navigator.clipboard.writeText(buildMessageTextForCopy(message));
      setCopiedMessageId(message.messageId);
      window.setTimeout(() => {
        setCopiedMessageId((prev) => (prev === message.messageId ? null : prev));
      }, 1200);
    } catch {
      setCopiedMessageId(null);
    }
  };

  const handleEditMessage = (message: CanvasMessage) => {
    setInputValue(message.content);
    setComposerAttachments(toComposerAttachments(message.attachments));
    textareaRef.current?.focus();
  };

  const handleRefreshMessage = (message: CanvasMessage) => {
    if (!message.parentUserMessageId) {
      return;
    }
    const source = messages.find(
      (entry) =>
        entry.role === "user"
        && entry.messageId === message.parentUserMessageId
        && !entry.transient
        && (entry.content.trim().length > 0 || (entry.attachments?.length ?? 0) > 0),
    );
    if (!source) {
      return;
    }
    void onSendMessage?.({
      content: source.content.trim(),
      attachments: source.attachments ?? [],
    });
  };

  const playViaBrowserTts = (message: CanvasMessage): boolean => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      return false;
    }
    const utterance = new SpeechSynthesisUtterance(message.content);
    utterance.onend = () => setSpeakingMessageId((prev) => (prev === message.messageId ? null : prev));
    utterance.onerror = () => setSpeakingMessageId((prev) => (prev === message.messageId ? null : prev));
    setSpeakingMessageId(message.messageId);
    window.speechSynthesis.speak(utterance);
    return true;
  };

  const handleListenToggle = async (message: CanvasMessage) => {
    if (!message.content.trim()) {
      return;
    }
    if (speakingMessageId === message.messageId) {
      stopPlayback();
      return;
    }

    stopPlayback();
    setTtsNotice(null);
    setSttError(null);
    try {
      const response = await fetch("/ui/api/tts/speak", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: message.content }),
      });
      if (!response.ok) {
        let payload: unknown = null;
        try {
          payload = await response.json();
        } catch {
          payload = null;
        }
        throw new Error(extractErrorMessage(payload, "TTS request failed."));
      }
      const audioBlob = await response.blob();
      if (audioBlob.size === 0) {
        throw new Error("TTS returned empty audio.");
      }
      const objectUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(objectUrl);
      audioRef.current = audio;
      audioUrlRef.current = objectUrl;
      audio.onended = () => stopPlayback();
      audio.onerror = () => {
        stopPlayback();
        setSttError("Не удалось воспроизвести аудио от TTS сервиса.");
      };
      setSpeakingMessageId(message.messageId);
      await audio.play();
    } catch (error) {
      stopPlayback();
      if (playViaBrowserTts(message)) {
        setTtsNotice("Server TTS недоступен, использую browser speech synthesis.");
      } else {
        setSttError(error instanceof Error ? error.message : "TTS failed.");
      }
    }
  };

  const handleFeedback = async (message: CanvasMessage, rating: "good" | "bad") => {
    const interactionId = typeof message.traceId === "string" ? message.traceId.trim() : "";
    if (!interactionId || !onSendFeedback) {
      return;
    }
    const previous = feedbackByMessageId[message.messageId] ?? null;
    if (previous === rating) {
      return;
    }
    setFeedbackBusyMessageId(message.messageId);
    setFeedbackByMessageId((prev) => ({ ...prev, [message.messageId]: rating }));
    const ok = await onSendFeedback(interactionId, rating);
    if (!ok) {
      setFeedbackByMessageId((prev) => {
        const next = { ...prev };
        if (previous === null) {
          delete next[message.messageId];
        } else {
          next[message.messageId] = previous;
        }
        return next;
      });
    }
    setFeedbackBusyMessageId((prev) => (prev === message.messageId ? null : prev));
  };

  const transcribeAudio = async (blob: Blob) => {
    setIsTranscribing(true);
    setSttError(null);
    try {
      const extension = blob.type.includes("ogg") ? "ogg" : "webm";
      const file = new File([blob], `recording.${extension}`, { type: blob.type || "audio/webm" });
      const body = new FormData();
      body.append("audio", file);
      body.append("language", "ru");
      const response = await fetch("/ui/api/stt/transcribe", {
        method: "POST",
        body,
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, "STT request failed."));
      }
      const text = (payload as { text?: unknown }).text;
      if (typeof text !== "string" || !text.trim()) {
        throw new Error("STT returned empty text.");
      }
      insertTextIntoComposer(text);
    } catch (error) {
      setSttError(error instanceof Error ? error.message : "STT failed.");
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleToggleRecording = async () => {
    if (!canUseMediaRecorder || sending || isTranscribing) {
      return;
    }
    if (isRecording) {
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.stop();
      }
      setIsRecording(false);
      return;
    }

    try {
      const stream = await window.navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeCandidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/ogg",
      ];
      const selectedMime = mimeCandidates.find((candidate) => {
        if (typeof window.MediaRecorder.isTypeSupported !== "function") {
          return false;
        }
        return window.MediaRecorder.isTypeSupported(candidate);
      });
      const recorder = selectedMime
        ? new window.MediaRecorder(stream, { mimeType: selectedMime })
        : new window.MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onerror = () => {
        setSttError("Не удалось записать аудио.");
        setIsRecording(false);
      };
      recorder.onstop = () => {
        setIsRecording(false);
        const chunks = audioChunksRef.current;
        audioChunksRef.current = [];
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
        }
        mediaRecorderRef.current = null;
        if (chunks.length === 0) {
          return;
        }
        const audioBlob = new Blob(chunks, {
          type: recorder.mimeType || "audio/webm",
        });
        void transcribeAudio(audioBlob);
      };
      recorder.start();
      setSttError(null);
      setIsRecording(true);
    } catch {
      setSttError("Микрофон недоступен.");
      setIsRecording(false);
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
    }
  };

  const handlePaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const imageFiles = Array.from(event.clipboardData.items ?? [])
      .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter((file): file is File => file instanceof File);
    if (imageFiles.length > 0) {
      event.preventDefault();
      void appendFilesToComposer(imageFiles);
      return;
    }
    const text = event.clipboardData.getData("text/plain");
    if (!longPasteToFileEnabled || !text || text.length <= effectiveLongPasteThreshold) {
      return;
    }
    if (composerAttachments.length >= MAX_COMPOSER_ATTACHMENTS) {
      setSttError("Достигнут лимит вложений в одном сообщении.");
      return;
    }
    event.preventDefault();
    const attachmentId = createComposerAttachmentId("paste");
    const attachment = buildPastedTextAttachment(text);
    setComposerAttachments((prev) => [...prev, { id: attachmentId, ...attachment }]);
    setPasteUndo({ attachmentId, originalText: text });
  };

  const handleUndoPaste = () => {
    if (!pasteUndo) {
      return;
    }
    const targetId = pasteUndo.attachmentId;
    const text = pasteUndo.originalText;
    setComposerAttachments((prev) => prev.filter((item) => item.id !== targetId));
    setPasteUndo(null);
    insertTextIntoComposer(text);
  };

  const handleRemoveComposerAttachment = (attachmentId: string) => {
    setComposerAttachments((prev) => prev.filter((item) => item.id !== attachmentId));
    setPasteUndo((prev) => (prev?.attachmentId === attachmentId ? null : prev));
  };

  return (
    <div
      className={`flex flex-col h-full bg-transparent ${className}`}
    >
      {/* Model selector header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#141418]">
        {modelOptions.length > 0 ? (
          <div className="relative">
            <select
              value={selectedModelValue ?? ""}
              onChange={(event) => {
                const next = modelOptions.find((option) => option.value === event.target.value);
                if (next && !next.disabled && next.model.trim()) {
                  onSelectModel?.(next.provider, next.model);
                }
              }}
              disabled={modelsLoading || savingModel}
              className="appearance-none bg-[#141418] text-[#aaa] text-[13px] px-3 py-1.5 rounded-lg border border-[#1f1f24] hover:border-[#2a2a30] transition-colors cursor-pointer pr-8"
            >
              <option value="" disabled>
                Select model
              </option>
              {modelOptions.map((option) => (
                <option
                  key={option.value}
                  value={option.value}
                  disabled={option.disabled}
                  className="bg-[#0b0b0d] text-[#ddd]"
                >
                  {option.label}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[#666]" />
          </div>
        ) : (
          <button
            onClick={onOpenSettings}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-[#141418] transition-colors cursor-pointer"
          >
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-[13px] text-[#aaa]">{modelName}</span>
            <ChevronDown className="w-3.5 h-3.5 text-[#666]" />
          </button>
        )}
        <button
          onClick={onToggleForceCanvasNext}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-[12px] transition-colors cursor-pointer ${
            forceCanvasNext
              ? "border-[#2f6a49] bg-[#173124] text-[#9fe3b8]"
              : "border-[#1f1f24] bg-[#141418] text-[#8f8f95] hover:border-[#2a2a30] hover:text-[#c6c6cb]"
          }`}
          title="Принудительно открыть Canvas для следующего ответа"
        >
          <PanelRight className="h-3.5 w-3.5" />
          {forceCanvasNext ? "Canvas: next ON" : "Canvas next"}
        </button>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto" data-scrollbar="auto">
        <div className="max-w-3xl mx-auto px-6 py-6 space-y-8">
          {displayMessages.map((msg) => {
            const isSavedMessage = !msg.transient;
            const canFeedback =
              !!onSendFeedback
              && msg.role === "assistant"
              && typeof msg.traceId === "string"
              && msg.traceId.trim().length > 0;
            const feedbackRating = feedbackByMessageId[msg.messageId] ?? null;
            return (
              <div key={msg.id}>
                <MessageBubble message={msg} />
                {isSavedMessage ? (
                  <MessageActions
                    message={msg}
                    copied={copiedMessageId === msg.messageId}
                    speaking={speakingMessageId === msg.messageId}
                    feedbackRating={feedbackRating}
                    feedbackBusy={feedbackBusyMessageId === msg.messageId}
                    sending={sending}
                    canRefresh={msg.role === "assistant" && !!msg.parentUserMessageId}
                    canFeedback={canFeedback}
                    onCopy={() => {
                      void handleCopyMessage(msg);
                    }}
                    onEdit={() => handleEditMessage(msg)}
                    onRefresh={() => handleRefreshMessage(msg)}
                    onListenToggle={() => handleListenToggle(msg)}
                    onLike={() => {
                      void handleFeedback(msg, "good");
                    }}
                    onDislike={() => {
                      void handleFeedback(msg, "bad");
                    }}
                  />
                ) : null}
              </div>
            );
          })}
          {showAssistantLoading ? <LoadingBubble /> : null}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {decision && decision.status === "pending" ? (
        <DecisionPanel
          decision={decision}
          busy={decisionBusy}
          error={decisionError}
          onRespond={(choice, editedAction) => {
            if (!onDecisionRespond) {
              return;
            }
            void onDecisionRespond(choice, editedAction);
          }}
        />
      ) : null}

      {/* Input area */}
      <div className="border-t border-[#141418] px-4 py-3">
        <div className="max-w-3xl mx-auto">
          {statusMessage ? (
            <div className="mb-2 rounded-lg border border-[#1f1f24] bg-[#141418] px-3 py-2 text-[12px] text-[#c0c0c0]">
              {statusMessage}
            </div>
          ) : null}
          {ttsNotice ? (
            <div className="mb-2 rounded-lg border border-[#1f1f24] bg-[#141418] px-3 py-2 text-[12px] text-[#c0c0c0]">
              {ttsNotice}
            </div>
          ) : null}
          {sttError ? (
            <div className="mb-2 rounded-lg border border-rose-700/40 bg-rose-900/20 px-3 py-2 text-[12px] text-rose-200">
              {sttError}
            </div>
          ) : null}
          {pasteUndo ? (
            <div className="mb-2 flex items-center justify-between gap-2 rounded-lg border border-[#1f1f24] bg-[#141418] px-3 py-2 text-[12px] text-[#c0c0c0]">
              <span>Вставка длинного текста сохранена как файл.</span>
              <button
                type="button"
                onClick={handleUndoPaste}
                className="rounded border border-[#2a2a30] px-2 py-0.5 text-[11px] text-[#ddd] hover:bg-[#1f1f24]"
              >
                Undo
              </button>
            </div>
          ) : null}
          {composerAttachments.length > 0 ? (
            <div className="mb-2 flex flex-wrap gap-2">
              {composerAttachments.map((attachment) => (
                <div
                  key={attachment.id}
                  className="inline-flex items-center gap-2 rounded-md border border-[#2a2a30] bg-[#141418] px-2.5 py-1 text-[12px] text-[#c8c8cc]"
                >
                  <FileText className="h-3.5 w-3.5 text-[#8f8f95]" />
                  <span className="max-w-[260px] truncate">{attachment.name}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveComposerAttachment(attachment.id)}
                    className="rounded p-0.5 text-[#8f8f95] hover:bg-[#1f1f24] hover:text-[#d6d6db]"
                    title="Remove attachment"
                    aria-label="Remove attachment"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            multiple
            accept="image/*,.txt,.md,.markdown,.json,.yaml,.yml,.toml,.csv,.log,.py,.ts,.tsx,.js,.jsx,.css,.scss,.html,.xml,.sh,.bash,.zsh,.ini,.cfg,.conf,.sql,.env"
            onChange={(event) => {
              void handleAttachFiles(event);
            }}
          />
          <div className="flex items-end gap-2 bg-[#141418] rounded-xl border border-[#1f1f24] focus-within:border-[#2a2a30] transition-colors px-4 py-3">
            {/* Attachment button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={sending || isTranscribing || composerBlocked}
              className="text-[#555] hover:text-[#999] transition-colors pb-0.5 cursor-pointer disabled:cursor-not-allowed disabled:text-[#444]"
              title="Attach file"
            >
              <Paperclip className="w-4.5 h-4.5" />
            </button>

            {/* Textarea */}
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onSelect={() => {
                const textarea = textareaRef.current;
                if (!textarea) {
                  return;
                }
                caretSelectionRef.current = {
                  start: textarea.selectionStart ?? 0,
                  end: textarea.selectionEnd ?? 0,
                };
              }}
              onClick={() => {
                const textarea = textareaRef.current;
                if (!textarea) {
                  return;
                }
                caretSelectionRef.current = {
                  start: textarea.selectionStart ?? 0,
                  end: textarea.selectionEnd ?? 0,
                };
              }}
              onKeyUp={() => {
                const textarea = textareaRef.current;
                if (!textarea) {
                  return;
                }
                caretSelectionRef.current = {
                  start: textarea.selectionStart ?? 0,
                  end: textarea.selectionEnd ?? 0,
                };
              }}
              onPaste={handlePaste}
              onKeyDown={handleKeyDown}
              placeholder={
                composerBlocked
                  ? "Нужно решить запрос разрешения перед отправкой."
                  : "Type your message... (Shift+Enter for new line)"
              }
              rows={1}
              className="composer-textarea flex-1 bg-transparent text-[14px] text-[#d4d4d8] placeholder-[#555] resize-none outline-none min-h-[24px] max-h-[120px]"
              style={{ lineHeight: "24px" }}
              disabled={sending || isTranscribing || composerBlocked}
              data-scrollbar="always"
            />

            {/* Mic button */}
            <button
              type="button"
              onClick={() => {
                void handleToggleRecording();
              }}
              disabled={!canUseMediaRecorder || sending || isTranscribing || composerBlocked}
              className={`relative transition-colors pb-0.5 cursor-pointer ${
                !canUseMediaRecorder
                  ? "text-[#444]"
                  : isRecording
                    ? "text-rose-300"
                    : isTranscribing
                      ? "text-amber-300"
                      : "text-[#555] hover:text-[#999]"
              }`}
              title={
                !canUseMediaRecorder
                  ? "Microphone unavailable"
                : isRecording
                  ? "Stop recording"
                  : isTranscribing
                    ? "Transcribing..."
                    : "Start recording"
              }
            >
              {isTranscribing ? (
                <LoaderCircle className="w-4.5 h-4.5 animate-spin" />
              ) : (
                <span className="relative inline-flex items-center justify-center">
                  {isRecording ? <span className="stt-mic-recording" aria-hidden="true" /> : null}
                  <Mic className="relative z-10 w-4.5 h-4.5" />
                </span>
              )}
            </button>

            {/* Send button */}
            <button
              onClick={() => {
                void handleSend();
              }}
              disabled={
                (!inputValue.trim() && composerAttachments.length === 0)
                || sending
                || isTranscribing
                || composerBlocked
              }
              className={`p-1.5 rounded-lg transition-all cursor-pointer ${
                (inputValue.trim() || composerAttachments.length > 0)
                && !sending
                && !isTranscribing
                && !composerBlocked
                  ? "bg-[#6366f1] hover:bg-[#5558e6] text-white"
                  : "bg-[#1b1b20] text-[#555]"
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
