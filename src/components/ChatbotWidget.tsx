import React, { useState, useRef, useEffect } from 'react';
import {
  sendMessageFull,
  getSessionId,
  createSession,
  clearChat,
} from '../services/messageClient';

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  isRedirect?: boolean;
}

interface ChatbotWidgetProps {
  className?: string;
  onNavigateToChat?: () => void;
}

const WEATHER_KEYWORDS = [
  'weather', 'temperature', 'temp', 'rain', 'rainfall', 'raining', 'rainy',
  'wind', 'windy', 'gust', 'humidity', 'humid', 'cloud', 'cloudy', 'overcast',
  'sunny', 'sunshine', 'fog', 'foggy', 'mist', 'snow', 'snowy', 'blizzard',
  'storm', 'thunder', 'thunderstorm', 'lightning', 'hail', 'frost', 'dew',
  'drizzle', 'shower', 'monsoon', 'cyclone', 'hurricane', 'typhoon', 'tornado',
  'flood', 'drought', 'heatwave', 'heat wave', 'cold wave', 'forecast',
  'climate', 'pressure', 'precipitation', 'uv index', 'air quality', 'aqi',
  'visibility', 'el niño', 'el nino', 'la niña', 'la nina', 'season',
  'winter', 'summer', 'spring', 'autumn', 'breeze', 'gale', 'smog',
  'dewpoint', 'dew point', 'barometric', 'celsius', 'fahrenheit', 'degree',
  'imd', 'meteorolog', 'atmosphere', 'atmospheric', 'hot', 'cold', 'chilly',
  'freeze', 'freezing', 'humid', 'dry spell', 'wet season', 'dry season',
];

function isWeatherRelated(text: string): boolean {
  const lower = text.toLowerCase();
  return WEATHER_KEYWORDS.some(kw => lower.includes(kw));
}

const REDIRECT_TEXT =
  "I'm your dedicated Weather Chatbot — I only answer weather-related questions!\n\nFor general questions, translations, facts, and more, use the AI Assistant (click below).";

const ChatbotWidget: React.FC<ChatbotWidgetProps> = ({
  className = '',
  onNavigateToChat,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'bot',
      content:
        "Hi! I'm your Weather Chatbot. Ask me anything about weather — current conditions, forecasts, monsoons, climate, and more.\n\nFor general questions unrelated to weather, use the AI Assistant tab.",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const [memoryCount, setMemoryCount] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    getSessionId();
  }, []);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const query = inputValue.trim();
    setInputValue('');

    // Redirect non-weather questions immediately without hitting the backend
    if (!isWeatherRelated(query)) {
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: REDIRECT_TEXT,
          timestamp: new Date(),
          isRedirect: true,
        },
      ]);
      return;
    }

    setIsLoading(true);
    try {
      const data = await sendMessageFull(query);
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: data.reply,
          timestamp: new Date(),
        },
      ]);
      if (data.memory) {
        setMemoryCount(data.memory.message_count);
      }
    } catch {
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
        },
      ]);
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
        type: 'bot',
        content: 'Memory cleared! Ask me a weather question.',
        timestamp: new Date(),
      },
    ]);
    setMemoryCount(0);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (date: Date) =>
    date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className={`fixed bottom-6 right-6 z-50 ${className}`}>
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          onMouseEnter={() => setIsHovering(true)}
          onMouseLeave={() => setIsHovering(false)}
          className={`bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 rounded-full shadow-2xl transition-all duration-300 flex items-center space-x-2 ${
            isHovering ? 'animate-pulse scale-110 shadow-blue-500/50' : 'scale-100'
          } ${isOpen ? 'rotate-45' : ''}`}
        >
          <svg
            className={`w-6 h-6 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            {isOpen ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            )}
          </svg>
          <span className={`hidden sm:inline transition-all duration-300 ${isOpen ? 'opacity-0' : 'opacity-100'}`}>
            Weather Chatbot
          </span>
          {memoryCount > 0 && !isOpen && (
            <span className="absolute -top-1 -right-1 w-5 h-5 bg-purple-500 rounded-full text-xs flex items-center justify-center font-bold">
              {memoryCount > 99 ? '99+' : memoryCount}
            </span>
          )}
        </button>

        {isOpen && (
          <div className="absolute bottom-20 right-0 w-96 max-w-[calc(100vw-3rem)] bg-white/10 backdrop-blur-sm rounded-xl shadow-2xl border border-white/20 flex flex-col h-[500px] animate-in slide-in-from-bottom-4 duration-300">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-white/20">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                <h3 className="text-white font-semibold">Weather Chatbot</h3>
                <span className="text-xs text-blue-300 bg-blue-500/20 px-2 py-0.5 rounded-full">
                  Weather Only
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleNewSession}
                  className="text-white/50 hover:text-white transition-colors p-1"
                  title="New Session"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
                <button
                  onClick={() => setIsOpen(false)}
                  className="text-white/70 hover:text-white transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-3 ${
                      message.type === 'user'
                        ? 'bg-blue-600 text-white'
                        : message.isRedirect
                        ? 'bg-amber-500/20 text-white border border-amber-400/30'
                        : 'bg-white/10 text-white'
                    }`}
                  >
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>

                    {message.isRedirect && onNavigateToChat && (
                      <button
                        onClick={() => {
                          setIsOpen(false);
                          onNavigateToChat();
                        }}
                        className="mt-3 w-full px-3 py-2 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white text-xs font-semibold rounded-lg transition-all duration-200"
                      >
                        Open AI Assistant
                      </button>
                    )}

                    <div className="flex items-center justify-between mt-2">
                      <span className="text-xs opacity-70">{formatTime(message.timestamp)}</span>
                    </div>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white/10 text-white rounded-lg p-3">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-white/70 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-white/70 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-white/70 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t border-white/20">
              <div className="flex space-x-2">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about weather..."
                  className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                  disabled={isLoading}
                />
                <button
                  onClick={sendMessage}
                  disabled={!inputValue.trim() || isLoading}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatbotWidget;
