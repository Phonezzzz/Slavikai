import { MessageSquarePlus, ChevronLeft, ChevronRight, Sparkles, Settings as SettingsIcon } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface SidebarProps {
  selectedConversation: string | null;
  onSelectConversation: (id: string) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onOpenSettings: () => void;
}

const conversations = [
  { id: 'conv-1', title: 'Current conversation', preview: 'Сделай мне нормальный UI в стиле manus.ai', time: '2m ago' },
  { id: 'conv-2', title: 'Architecture Review', preview: 'Can you help me review the MWV runtime?', time: '1h ago' },
  { id: 'conv-3', title: 'Project Setup', preview: 'Setting up the workspace tools', time: '3h ago' },
  { id: 'conv-4', title: 'Bug Fix Session', preview: 'Fixing the trace logging issue', time: 'Yesterday' },
  { id: 'conv-5', title: 'New Feature', preview: 'Adding image generation support', time: '2 days ago' },
];

export function Sidebar({ selectedConversation, onSelectConversation, collapsed, onToggleCollapse, onOpenSettings }: SidebarProps) {
  return (
    <motion.div
      initial={false}
      animate={{ width: collapsed ? '0px' : '280px' }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="relative border-r border-white/5 bg-zinc-900/50 backdrop-blur-xl"
    >
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex flex-col h-full w-[280px]"
          >
            {/* Header */}
            <div className="p-4 border-b border-white/5">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 rounded-lg bg-white/10 border border-white/20 flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <h1 className="text-lg font-semibold text-white">
                  SlavikAI
                </h1>
              </div>
              
              <button className="w-full px-4 py-2.5 rounded-lg bg-white hover:bg-white/90 text-black font-medium transition-all duration-200 flex items-center justify-center gap-2">
                <MessageSquarePlus className="w-4 h-4" />
                New Chat
              </button>
            </div>

            {/* Conversations List */}
            <div className="flex-1 overflow-y-auto p-2">
              <div className="text-xs font-medium text-white/40 px-3 py-2 mb-1">
                CONVERSATIONS
              </div>
              <div className="space-y-1">
                {conversations.map((conv) => (
                  <button
                    key={conv.id}
                    onClick={() => onSelectConversation(conv.id)}
                    className={`w-full text-left px-3 py-2.5 rounded-lg transition-all duration-200 group ${
                      selectedConversation === conv.id
                        ? 'bg-white/10 text-white shadow-lg'
                        : 'text-white/60 hover:bg-white/5 hover:text-white/90'
                    }`}
                  >
                    <div className="font-medium text-sm mb-0.5 truncate">
                      {conv.title}
                    </div>
                    <div className="text-xs text-white/40 truncate">
                      {conv.preview}
                    </div>
                    <div className="text-xs text-white/30 mt-1">
                      {conv.time}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-white/5 space-y-3">
              <button
                onClick={onOpenSettings}
                className="w-full px-4 py-2.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white/70 hover:text-white font-medium transition-all duration-200 flex items-center justify-center gap-2"
              >
                <SettingsIcon className="w-4 h-4" />
                Settings
              </button>
              <div className="text-xs text-white/40 text-center">
                SlavikAI v1.0 • Python Agent
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle Button */}
      <button
        onClick={onToggleCollapse}
        className="absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-xl border border-white/10 flex items-center justify-center transition-all duration-200 z-10"
      >
        {collapsed ? (
          <ChevronRight className="w-3.5 h-3.5 text-white/60" />
        ) : (
          <ChevronLeft className="w-3.5 h-3.5 text-white/60" />
        )}
      </button>
    </motion.div>
  );
}
