
const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

// Session ID persisted in localStorage for cross-refresh memory
const SESSION_KEY = "pastcast_session_id";

function getStoredSessionId(): string | null {
  try {
    return localStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}

function storeSessionId(id: string): void {
  try {
    localStorage.setItem(SESSION_KEY, id);
  } catch {
    // localStorage may not be available
  }
}

let _sessionId: string | null = getStoredSessionId();

export async function createSession(): Promise<string> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) throw new Error("Failed to create session");
    const data = await res.json();
    _sessionId = data.session_id;
    storeSessionId(_sessionId!);
    return _sessionId!;
  } catch {
    // Fallback: generate a local ID
    _sessionId = `local_${Date.now()}`;
    storeSessionId(_sessionId);
    return _sessionId;
  }
}

export async function getSessionId(): Promise<string> {
  if (!_sessionId) {
    return await createSession();
  }
  return _sessionId;
}

export function clearSession(): void {
  _sessionId = null;
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch {
    // noop
  }
}

export interface MessageResponse {
  reply: string;
  session_id: string;
  status: string;
  timestamp: string;
  memory?: {
    rag_used: boolean;
    lstm_active: boolean;
    message_count: number;
  };
}

export async function sendMessage(text: string): Promise<string> {
  try {
    const sessionId = await getSessionId();
    const res = await fetch(`${API_BASE_URL}/api/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, session_id: sessionId }),
    });

    if (!res.ok) throw new Error("Network error");

    const data: MessageResponse = await res.json();

    // Update session ID if server returned one
    if (data.session_id && data.session_id !== _sessionId) {
      _sessionId = data.session_id;
      storeSessionId(_sessionId);
    }

    return data.reply;
  } catch {
    return "AI server error. Please try again.";
  }
}

export async function sendMessageFull(text: string): Promise<MessageResponse> {
  const sessionId = await getSessionId();
  const res = await fetch(`${API_BASE_URL}/api/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, session_id: sessionId }),
  });

  if (!res.ok) throw new Error("Network error");
  const data: MessageResponse = await res.json();

  if (data.session_id) {
    _sessionId = data.session_id;
    storeSessionId(_sessionId);
  }

  return data;
}

export async function getSessionContext(): Promise<{
  context_summary: string;
  message_count: number;
} | null> {
  try {
    const sessionId = await getSessionId();
    const res = await fetch(
      `${API_BASE_URL}/api/session/${sessionId}/context`
    );
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function clearChat(): Promise<void> {
  try {
    const sessionId = _sessionId;
    await fetch(`${API_BASE_URL}/api/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    clearSession();
  } catch {
    // noop
  }
}
