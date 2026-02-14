import { useState, useRef, useEffect } from "react";
import { Search, Edit3, FileText, X } from "lucide-react";

export interface ChatHistoryItem {
  id: string;
  title: string;
  date: string;
  messageCount: number;
  preview?: string;
}

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  chats?: ChatHistoryItem[];
  onSelectChat?: (id: string) => void;
  onNewChat?: () => void;
  onNewNote?: () => void;
}

const defaultChats: ChatHistoryItem[] = [];

function groupChatsByDate(chats: ChatHistoryItem[]) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const lastWeek = new Date(today);
  lastWeek.setDate(lastWeek.getDate() - 7);

  const groups: { label: string; chats: ChatHistoryItem[] }[] = [
    { label: "Today", chats: [] },
    { label: "Yesterday", chats: [] },
    { label: "Previous 7 days", chats: [] },
    { label: "Older", chats: [] },
  ];

  chats.forEach((chat) => {
    const dateStr = chat.date;
    if (
      dateStr.includes("Last") ||
      dateStr.includes("last")
    ) {
      groups[2].chats.push(chat);
    } else {
      try {
        const d = new Date(dateStr);
        if (d >= today) groups[0].chats.push(chat);
        else if (d >= yesterday) groups[1].chats.push(chat);
        else if (d >= lastWeek) groups[2].chats.push(chat);
        else groups[3].chats.push(chat);
      } catch {
        groups[3].chats.push(chat);
      }
    }
  });

  return groups.filter((g) => g.chats.length > 0);
}

export function SearchModal({
  isOpen,
  onClose,
  chats = defaultChats,
  onSelectChat,
  onNewChat,
  onNewNote,
}: SearchModalProps) {
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedId(null);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) {
      document.addEventListener("keydown", handleKeyDown);
      return () => document.removeEventListener("keydown", handleKeyDown);
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const filteredChats = query
    ? chats.filter((c) =>
        c.title.toLowerCase().includes(query.toLowerCase())
      )
    : chats;

  const groups = groupChatsByDate(filteredChats);
  const selectedChat = chats.find((c) => c.id === selectedId);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[10vh]"
      onClick={onClose}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />

      {/* Modal */}
      <div
        className="relative w-[780px] max-h-[70vh] bg-[#0f0f12] rounded-2xl border border-[#1f1f24] shadow-2xl shadow-black/50 overflow-hidden flex"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Left side - search & list */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Search input */}
          <div className="flex items-center gap-3 px-5 py-4 border-b border-[#1f1f24]">
            <Search className="w-4.5 h-4.5 text-[#666] flex-shrink-0" />
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search"
              className="flex-1 bg-transparent text-[14px] text-[#e0e0e0] placeholder-[#555] outline-none"
            />
            {query && (
              <button
                onClick={() => setQuery("")}
                className="text-[#555] hover:text-[#999] cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Actions */}
          {!query && (
            <div className="px-4 py-2 border-b border-[#1f1f24]">
              <p className="text-[11px] text-[#666] uppercase tracking-wider px-2 mb-1">
                Actions
              </p>
              <button
                onClick={() => {
                  onNewChat?.();
                  onClose();
                }}
                className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-[#ccc] hover:bg-[#1b1b20] transition-colors cursor-pointer"
              >
                <Edit3 className="w-4 h-4 text-[#888]" />
                Start a new conversation
              </button>
              <button
                onClick={() => {
                  onNewNote?.();
                  onClose();
                }}
                className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-[#ccc] hover:bg-[#1b1b20] transition-colors cursor-pointer"
              >
                <FileText className="w-4 h-4 text-[#888]" />
                Create a new note
              </button>
            </div>
          )}

          {/* Chat list */}
          <div className="flex-1 overflow-y-auto" data-scrollbar="auto">
            <div className="px-4 py-2">
              {groups.map((group) => (
                <div key={group.label} className="mb-3">
                  <p className="text-[11px] text-[#666] uppercase tracking-wider px-2 mb-1">
                    {group.label}
                  </p>
                  {group.chats.map((chat) => (
                    <button
                      key={chat.id}
                      onClick={() => {
                        setSelectedId(chat.id);
                        onSelectChat?.(chat.id);
                      }}
                      className={`flex items-center justify-between w-full px-3 py-2.5 rounded-lg text-left transition-colors cursor-pointer ${
                        selectedId === chat.id
                          ? "bg-[#1b1b20]"
                          : "hover:bg-[#141418]"
                      }`}
                    >
                      <span className="text-[13px] text-[#ccc] truncate flex-1 mr-3">
                        {chat.title}
                      </span>
                      <span className="text-[11px] text-[#555] flex-shrink-0 whitespace-nowrap">
                        {chat.date.includes(",")
                          ? chat.date.split(",")[0]
                          : chat.date}
                      </span>
                    </button>
                  ))}
                </div>
              ))}
              {filteredChats.length === 0 && (
                <div className="px-3 py-8 text-center">
                  <p className="text-[13px] text-[#555]">No results found</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right side - preview */}
        {selectedChat && (
          <div className="w-[340px] border-l border-[#1f1f24] flex flex-col">
            <div className="flex-1 flex items-center justify-center px-6">
              {selectedChat.preview ? (
                <div className="w-full">
                  <p className="text-[13px] text-[#b0b0b0] leading-relaxed whitespace-pre-wrap">
                    {selectedChat.preview}
                  </p>
                </div>
              ) : (
                <p className="text-[13px] text-[#555]">
                  Select a conversation to preview
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
