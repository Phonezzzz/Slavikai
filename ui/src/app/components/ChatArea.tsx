import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, User, Code2, Copy, Volume2, Brain, RotateCw, StopCircle, Edit2 } from 'lucide-react';
import { motion } from 'motion/react';

interface ChatAreaProps {
  conversationId: string | null;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

const mockMessages: Message[] = [
  {
    id: '1',
    role: 'assistant',
    content: 'Привет! Я SlavikAI — твой локальный агент с MWV архитектурой. Чем могу помочь?',
    timestamp: '13:14',
  },
  {
    id: '2',
    role: 'user',
    content: 'Сделай мне нормальный UI в стиле manus.ai',
    timestamp: '13:15',
  },
  {
    id: '3',
    role: 'assistant',
    content: 'Понял! Создаю современный интерфейс с градиентами, плавными анимациями и минималистичным дизайном. Используем темную тему с акцентами violet/fuchsia.',
    timestamp: '13:15',
  },
];

export function ChatArea({ conversationId }: ChatAreaProps) {
  const [messages, setMessages] = useState<Message[]>(mockMessages);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages([...messages, userMessage]);
    setInput('');
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Отличный вопрос! Сейчас работаю над этим...',
        timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, aiMessage]);
      setIsTyping(false);
    }, 1000);
  };

  return (
    <div className="flex-1 flex flex-col relative">
      {/* Subtle Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-white/[0.02] via-transparent to-white/[0.02] pointer-events-none" />

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-8 relative z-10">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((message, index) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-white/10 border border-white/20 flex items-center justify-center shrink-0">
                  <Sparkles className="w-4 h-4 text-white" />
                </div>
              )}

              <div className={`flex flex-col gap-2 max-w-[70%] ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div
                  className={`px-4 py-3 rounded-2xl ${
                    message.role === 'user'
                      ? 'bg-white text-black'
                      : 'bg-transparent text-white/90'
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                </div>
                
                {/* Message Actions */}
                <div className="flex items-center gap-1">
                  <span className="text-xs text-white/30 px-2">{message.timestamp}</span>
                  
                  {message.role === 'assistant' ? (
                    <div className="flex gap-1">
                      <button className="p-1 rounded hover:bg-white/5 transition-colors" title="Copy">
                        <Copy className="w-3 h-3 text-white/40 hover:text-white/70" />
                      </button>
                      <button className="p-1 rounded hover:bg-white/5 transition-colors" title="Listen">
                        <Volume2 className="w-3 h-3 text-white/40 hover:text-white/70" />
                      </button>
                      <button className="p-1 rounded hover:bg-white/5 transition-colors" title="Think deeper">
                        <Brain className="w-3 h-3 text-white/40 hover:text-white/70" />
                      </button>
                      <button className="p-1 rounded hover:bg-white/5 transition-colors" title="Refresh">
                        <RotateCw className="w-3 h-3 text-white/40 hover:text-white/70" />
                      </button>
                      <button className="p-1 rounded hover:bg-white/5 transition-colors" title="Stop">
                        <StopCircle className="w-3 h-3 text-white/40 hover:text-white/70" />
                      </button>
                    </div>
                  ) : (
                    <div className="flex gap-1">
                      <button className="p-1 rounded hover:bg-white/5 transition-colors" title="Copy">
                        <Copy className="w-3 h-3 text-white/40 hover:text-white/70" />
                      </button>
                      <button className="p-1 rounded hover:bg-white/5 transition-colors" title="Edit">
                        <Edit2 className="w-3 h-3 text-white/40 hover:text-white/70" />
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {message.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-white/10 border border-white/20 flex items-center justify-center shrink-0">
                  <User className="w-4 h-4 text-white/60" />
                </div>
              )}
            </motion.div>
          ))}

          {isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-4"
            >
              <div className="w-8 h-8 rounded-full bg-white/10 border border-white/20 flex items-center justify-center shrink-0">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div className="bg-transparent px-4 py-3">
                <div className="flex gap-1">
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
                    className="w-2 h-2 rounded-full bg-white/40"
                  />
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                    className="w-2 h-2 rounded-full bg-white/40"
                  />
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                    className="w-2 h-2 rounded-full bg-white/40"
                  />
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-white/5 bg-zinc-900/50 backdrop-blur-xl p-6 relative z-10">
        <div className="max-w-3xl mx-auto">
          <div className="relative">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Type your message... (Shift+Enter for new line)"
              className="w-full px-4 py-3 pr-12 bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl text-white placeholder-white/30 resize-none focus:outline-none focus:ring-2 focus:ring-white/20 focus:border-white/20 transition-all duration-200"
              rows={1}
              style={{ minHeight: '48px', maxHeight: '200px' }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg bg-white hover:bg-white/90 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-all duration-200"
            >
              <Send className="w-4 h-4 text-black" />
            </button>
          </div>

          {/* Quick Actions */}
          <div className="flex gap-2 mt-3">
            <button className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs text-white/60 hover:text-white/90 transition-all duration-200 flex items-center gap-1.5">
              <Code2 className="w-3 h-3" />
              /fs
            </button>
            <button className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs text-white/60 hover:text-white/90 transition-all duration-200 flex items-center gap-1.5">
              <Code2 className="w-3 h-3" />
              /web
            </button>
            <button className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs text-white/60 hover:text-white/90 transition-all duration-200 flex items-center gap-1.5">
              <Code2 className="w-3 h-3" />
              /project
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
