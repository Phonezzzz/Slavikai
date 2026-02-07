import { useState } from "react";
import {
  Search,
  StickyNote,
  LayoutGrid,
  FolderClosed,
  MessageSquare,
  Calendar,
  Clock,
  Settings,
  Trash2,
  MoreHorizontal,
  Plus,
  Bot,
} from "lucide-react";

export interface ChatItem {
  id: string;
  title: string;
  messageCount: number;
  date: string;
  group: "today" | "yesterday" | "older";
}

interface HistorySidebarProps {
  chats?: ChatItem[];
  activeChatId?: string | null;
  onNewChat?: () => void;
  onSelectChat?: (id: string) => void;
  onDeleteChat?: (id: string) => void;
  onOpenSearch?: () => void;
  onOpenNotes?: () => void;
  onOpenWorkspace?: () => void;
  onOpenSettings?: () => void;
  className?: string;
}

const defaultChats: ChatItem[] = [
  {
    id: "bf09b790",
    title: "New chat",
    messageCount: 0,
    date: "2/7/2026, 1:25:16 AM",
    group: "today",
  },
  {
    id: "6d8e750f",
    title: "napishio kusok koda v canvas",
    messageCount: 2,
    date: "2/6/2026, 10:14:27 PM",
    group: "yesterday",
  },
  {
    id: "46163d26",
    title: "napishi otrezok coda v canvas xochu potestit...",
    messageCount: 2,
    date: "2/6/2026, 8:30:34 PM",
    group: "yesterday",
  },
];

export function HistorySidebar({
  chats = defaultChats,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onOpenSearch,
  onOpenNotes,
  onOpenWorkspace,
  onOpenSettings,
  className = "",
}: HistorySidebarProps) {
  const [hoveredChat, setHoveredChat] = useState<string | null>(null);

  const todayChats = chats.filter((c) => c.group === "today");
  const yesterdayChats = chats.filter((c) => c.group === "yesterday");
  const olderChats = chats.filter((c) => c.group === "older");

  const formatDate = (dateStr: string) => {
    try {
      const d = new Date(dateStr);
      return d.toLocaleString("en-US", {
        month: "numeric",
        day: "numeric",
        year: "numeric",
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
      });
    } catch {
      return dateStr;
    }
  };

  const ChatGroup = ({
    label,
    icon,
    items,
  }: {
    label: string;
    icon: React.ReactNode;
    items: ChatItem[];
  }) => {
    if (items.length === 0) return null;
    return (
      <div className="mb-3">
        <div className="flex items-center gap-2 px-3 py-1.5">
          {icon}
          <span className="text-[11px] text-[#666] uppercase tracking-wider">
            {label}
          </span>
        </div>
        {items.map((chat) => (
          <div
            key={chat.id}
            className={`group flex items-center gap-2 mx-2 px-3 py-2.5 rounded-lg cursor-pointer transition-all ${
              activeChatId === chat.id
                ? "bg-[#2a2a30] border border-[#3a3a42]"
                : "hover:bg-[#1e1e24] border border-transparent"
            }`}
            onClick={() => onSelectChat?.(chat.id)}
            onMouseEnter={() => setHoveredChat(chat.id)}
            onMouseLeave={() => setHoveredChat(null)}
          >
            <div className="flex-1 min-w-0">
              <div className="text-[13px] text-[#d0d0d0] truncate">
                {chat.title}
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-[11px] text-[#666]">
                  {chat.messageCount} messages
                </span>
                <span className="text-[11px] text-[#444]">-</span>
                <span className="text-[11px] text-[#555]">
                  {formatDate(chat.date)}
                </span>
              </div>
            </div>

            {/* Action buttons on hover */}
            <div
              className={`flex items-center gap-1 flex-shrink-0 transition-opacity ${
                hoveredChat === chat.id ? "opacity-100" : "opacity-0"
              }`}
            >
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteChat?.(chat.id);
                }}
                className="p-1 rounded text-[#555] hover:text-red-400 hover:bg-red-400/10 transition-colors cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={(e) => e.stopPropagation()}
                className="p-1 rounded text-[#555] hover:text-[#aaa] hover:bg-[#333] transition-colors cursor-pointer"
              >
                <MoreHorizontal className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div
      className={`flex flex-col h-full w-[260px] bg-[#141418] border-r border-[#222226] ${className}`}
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-5 py-4">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#6366f1] to-[#8b5cf6] flex items-center justify-center">
          <Bot className="w-4 h-4 text-white" />
        </div>
        <span className="text-[15px] text-[#e0e0e0]">SlavikAI</span>
      </div>

      {/* New Chat button */}
      <div className="px-3 mb-3">
        <button
          onClick={onNewChat}
          className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl bg-[#222228] hover:bg-[#2a2a32] border border-[#2f2f35] hover:border-[#3a3a42] text-[13px] text-[#d0d0d0] transition-all cursor-pointer"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {/* Quick actions */}
      <div className="px-2 mb-2 space-y-0.5">
        <button
          onClick={onOpenSearch}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-[#999] hover:text-[#ccc] hover:bg-[#1e1e24] transition-all cursor-pointer"
        >
          <Search className="w-4 h-4" />
          Search
        </button>
        <button
          onClick={onOpenNotes}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-[#999] hover:text-[#ccc] hover:bg-[#1e1e24] transition-all cursor-pointer"
        >
          <StickyNote className="w-4 h-4" />
          Notes
        </button>
        <button
          onClick={onOpenWorkspace}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-[#999] hover:text-[#ccc] hover:bg-[#1e1e24] transition-all cursor-pointer"
        >
          <LayoutGrid className="w-4 h-4" />
          Workspace
        </button>
      </div>

      {/* Folders section */}
      <div className="px-3 mb-2">
        <div className="flex items-center gap-2 px-2 py-1.5">
          <FolderClosed className="w-3.5 h-3.5 text-[#555]" />
          <span className="text-[11px] text-[#555] uppercase tracking-wider">
            Folders
          </span>
        </div>
        <p className="px-3 py-2 text-[12px] text-[#444] italic">
          No folders yet
        </p>
      </div>

      {/* Divider */}
      <div className="mx-3 border-t border-[#222226] mb-2" />

      {/* Chats */}
      <div className="px-1 mb-1">
        <div className="flex items-center gap-2 px-4 py-1.5">
          <MessageSquare className="w-3.5 h-3.5 text-[#555]" />
          <span className="text-[11px] text-[#555] uppercase tracking-wider">
            Chats
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto" data-scrollbar>
        <ChatGroup
          label="Today"
          icon={<Calendar className="w-3 h-3 text-[#555]" />}
          items={todayChats}
        />
        <ChatGroup
          label="Yesterday"
          icon={<Clock className="w-3 h-3 text-[#555]" />}
          items={yesterdayChats}
        />
        <ChatGroup
          label="Older"
          icon={<Clock className="w-3 h-3 text-[#555]" />}
          items={olderChats}
        />
      </div>

      {/* Bottom section */}
      <div className="border-t border-[#222226] px-3 py-3 space-y-2">
        <button
          onClick={onOpenSettings}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-[#999] hover:text-[#ccc] hover:bg-[#1e1e24] transition-all cursor-pointer"
        >
          <Settings className="w-4 h-4" />
          Settings
        </button>
        <p className="text-[11px] text-[#444] text-center">
          SlavikAI v1.0
        </p>
      </div>
    </div>
  );
}
