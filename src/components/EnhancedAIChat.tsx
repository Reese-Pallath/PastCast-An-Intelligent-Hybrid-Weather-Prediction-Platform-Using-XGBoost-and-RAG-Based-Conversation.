import React, { useState, useRef, useEffect } from 'react';
import {
  sendMessageFull,
  getSessionId,
  createSession,
  clearChat,
  getSessionContext,
} from '../services/messageClient';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

interface MemoryStatus {
  rag_used: boolean;
  lstm_active: boolean;
  message_count: number;
}

interface EnhancedAIChatProps {
  location?: string;
}

const EnhancedAIChat: React.FC<EnhancedAIChatProps> = ({
  location = 'Bengaluru, India',
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: `Hello! I'm your general-purpose AI Assistant. Ask me anything — weather questions, world facts, historical events, translations (Hindi, Marathi, Tamil, Telugu), explanations, and more. I remember our conversation context across messages. What would you like to know?`,
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [memoryStatus, setMemoryStatus] = useState<MemoryStatus | null>(null);
  const [contextSummary, setContextSummary] = useState('');
  const [showMemory, setShowMemory] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize session on mount
  useEffect(() => {
    const initSession = async () => {
      await getSessionId();
      const ctx = await getSessionContext();
      if (ctx && ctx.message_count > 0) {
        setContextSummary(ctx.context_summary);
      }
    };
    initSession();
  }, []);

  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      text: inputText,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    const query = inputText.trim();
    setInputText('');
    setIsLoading(true);

    try {
      const response = await sendMessageFull(query);

      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        text: response.reply,
        isUser: false,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiMsg]);

      // Update memory status
      if (response.memory) {
        setMemoryStatus(response.memory);
      }
    } catch (err) {
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        text: 'AI server error. Please try again.',
        isUser: false,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewSession = async () => {
    await clearChat();
    await createSession();
    setMessages([
      {
        id: Date.now().toString(),
        text: 'New session started! My memory has been reset. How can I help you?',
        isUser: false,
        timestamp: new Date(),
      },
    ]);
    setMemoryStatus(null);
    setContextSummary('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date: Date) =>
    date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

  return (
    <div className="relative">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-cyan-500/5 rounded-2xl blur-xl"></div>

      <div className="relative flex flex-col h-full bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 shadow-2xl">

        {/* Header */}
        <div className="p-6 border-b border-white/10 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-xl flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
            </div>

            <div>
              <h3 className="text-2xl font-bold text-white bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">
                AI Assistant
              </h3>
              <p className="text-white/70 text-sm font-medium">{location}</p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* Memory indicator */}
            {memoryStatus && (
              <button
                onClick={() => setShowMemory(!showMemory)}
                className="flex items-center space-x-2 bg-white/5 rounded-full px-3 py-1.5 border border-white/10 hover:bg-white/10 transition-colors cursor-pointer"
                title="Memory Status"
              >
                <div className={`w-2 h-2 rounded-full ${memoryStatus.lstm_active ? 'bg-purple-400 animate-pulse' : 'bg-gray-400'}`}></div>
                <span className="text-white/70 text-xs font-medium">
                  {memoryStatus.message_count} msgs
                </span>
                {memoryStatus.rag_used && (
                  <span className="text-cyan-400 text-xs">RAG</span>
                )}
              </button>
            )}

            {/* Online status */}
            <div className="flex items-center space-x-2 bg-white/5 rounded-full px-4 py-2 border border-white/10">
              <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-400 text-sm font-medium">Online</span>
            </div>

            {/* New Session button */}
            <button
              onClick={handleNewSession}
              className="px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white/70 hover:text-white hover:bg-white/10 transition-all text-sm"
              title="Start New Session"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        </div>

        {/* Memory context panel */}
        {showMemory && memoryStatus && (
          <div className="px-6 py-3 bg-purple-500/10 border-b border-purple-400/20">
            <div className="flex items-center space-x-2 mb-1">
              <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <span className="text-purple-300 text-xs font-semibold">LSTM Memory Active</span>
            </div>
            <p className="text-white/60 text-xs">
              {contextSummary || `Tracking ${memoryStatus.message_count} messages in this session.`}
              {memoryStatus.rag_used && ' | RAG knowledge retrieval enabled.'}
            </p>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((m) => (
            <div key={m.id} className={`flex ${m.isUser ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[80%] rounded-2xl p-4 ${
                  m.isUser
                    ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg shadow-blue-500/25'
                    : 'bg-white/5 backdrop-blur-sm text-white border border-white/10 shadow-lg'
                }`}
              >
                <p className="whitespace-pre-wrap text-sm leading-relaxed">{m.text}</p>

                <div className="mt-2 text-xs text-white/60 text-right">
                  {formatTime(m.timestamp)}
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-4 border border-white/10 shadow-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center">
                    <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white"></div>
                  </div>
                  <span className="text-white/70 font-medium">AI is thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-6 border-t border-white/10">
          <div className="flex space-x-3">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything... (I remember our conversation)"
              disabled={isLoading}
              className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-300 disabled:opacity-50"
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isLoading}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 disabled:from-blue-500/40 disabled:to-cyan-500/40 text-white font-semibold rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg shadow-blue-500/25"
            >
              {isLoading ? '...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedAIChat;
