
export interface ChatbotMessage {
  text: string;
  conversation_history?: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

export interface ChatbotResponse {
  reply: string;
  timestamp: string;
  model: string;
  status: string;
  confidence?: number;
  dataSource?: string;
}

export interface ChatbotError {
  error: string;
  message?: string;
  timestamp: string;
}

class ChatbotService {
  private baseUrl: string;

  constructor() {

    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  async sendMessage(
    message: string,
    conversationHistory?: Array<{ role: 'user' | 'assistant'; content: string }>
  ): Promise<ChatbotResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: message,
          conversation_history: conversationHistory || [],
        }),
      });

      if (!response.ok) {
        const errorData: ChatbotError = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data: ChatbotResponse = await response.json();
      return data;
    } catch (error) {
      console.error('Chatbot service error:', error);
      throw error;
    }
  }

  async sendChatbotMessage(text: string): Promise<ChatbotResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          timestamp: new Date().toISOString(),
        }),
      });

      if (!response.ok) {
        const errorData: ChatbotError = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data: ChatbotResponse = await response.json();
      return data;
    } catch (error) {
      console.error('Chatbot service error:', error);
      throw error;
    }
  }

  async checkHealth(): Promise<{
    status: string;
    service: string;
    timestamp: string;
    openai_configured: boolean;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }
}

export const chatbotService = new ChatbotService();
