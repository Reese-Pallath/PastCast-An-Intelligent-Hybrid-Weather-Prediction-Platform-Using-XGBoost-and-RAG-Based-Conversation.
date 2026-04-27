import React, { useState, useEffect, useCallback } from 'react';

interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  description?: string;
}

interface ToastProps {
  message: ToastMessage;
  onDismiss: (id: string) => void;
}

const Toast: React.FC<ToastProps> = ({ message, onDismiss }) => {
  useEffect(() => {
    const timer = setTimeout(() => onDismiss(message.id), 5000);
    return () => clearTimeout(timer);
  }, [message.id, onDismiss]);

  const bgColors = {
    success: 'from-green-500/20 to-emerald-500/20 border-green-500/30',
    error: 'from-red-500/20 to-rose-500/20 border-red-500/30',
    warning: 'from-amber-500/20 to-yellow-500/20 border-amber-500/30',
    info: 'from-blue-500/20 to-cyan-500/20 border-blue-500/30',
  };

  const icons = {
    success: (
      <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
    error: (
      <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    ),
    warning: (
      <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
      </svg>
    ),
    info: (
      <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  };

  return (
    <div
      className={`
        bg-gradient-to-r ${bgColors[message.type]} backdrop-blur-md
        rounded-xl p-4 border shadow-xl
        animate-in slide-in-from-right duration-300
        flex items-start space-x-3 min-w-[320px] max-w-md
      `}
    >
      <div className="flex-shrink-0 mt-0.5">{icons[message.type]}</div>
      <div className="flex-1 min-w-0">
        <p className="text-white font-semibold text-sm">{message.title}</p>
        {message.description && (
          <p className="text-white/70 text-xs mt-1">{message.description}</p>
        )}
      </div>
      <button
        onClick={() => onDismiss(message.id)}
        className="flex-shrink-0 text-white/40 hover:text-white/80 transition-colors"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
};

// ── Toast container + hook ─────────────────────────────────────

let _addToast: ((msg: Omit<ToastMessage, 'id'>) => void) | null = null;

export function showToast(type: ToastMessage['type'], title: string, description?: string) {
  _addToast?.({ type, title, description });
}

export const ToastContainer: React.FC = () => {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const addToast = useCallback((msg: Omit<ToastMessage, 'id'>) => {
    const id = Date.now().toString() + Math.random().toString(36).slice(2);
    setToasts((prev) => [...prev, { ...msg, id }]);
  }, []);

  useEffect(() => {
    _addToast = addToast;
    return () => { _addToast = null; };
  }, [addToast]);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <div className="fixed top-4 right-4 z-[9999] flex flex-col space-y-3">
      {toasts.map((t) => (
        <Toast key={t.id} message={t} onDismiss={dismissToast} />
      ))}
    </div>
  );
};

export default Toast;
