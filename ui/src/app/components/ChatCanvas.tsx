import { ChevronLeft, ChevronRight } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';

interface ChatCanvasProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export function ChatCanvas({ collapsed, onToggleCollapse }: ChatCanvasProps) {
  return (
    <motion.div
      initial={false}
      animate={{ width: collapsed ? '0px' : '480px' }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="relative bg-zinc-950"
    >
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex h-full w-[480px] flex-col"
          >
            <div className="p-4">
              <div className="text-xs font-semibold uppercase tracking-wide text-white/40">
                Chat Canvas
              </div>
              <div className="mt-2 text-sm text-white/80">
                Артефакты сессии появятся здесь.
              </div>
            </div>
            <div className="flex-1 overflow-y-auto px-4 pb-6">
              <div className="rounded-lg border border-white/10 bg-black/40 p-4 text-sm text-white/60">
                Пока пусто. Здесь будут результат работы агента, файлы и история загрузок.
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <button
        type="button"
        onClick={onToggleCollapse}
        className="absolute -left-3 top-1/2 z-10 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full border border-white/10 bg-white/10 backdrop-blur-xl transition-all duration-200 hover:bg-white/20"
      >
        {collapsed ? (
          <ChevronLeft className="h-3.5 w-3.5 text-white/60" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-white/60" />
        )}
      </button>
    </motion.div>
  );
}
