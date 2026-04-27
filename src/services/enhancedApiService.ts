
import { API_KEYS, API_ENDPOINTS } from '../config/apiKeys';

export interface GeminiResponse {
  text: string;
  confidence: number;
  sources?: string[];
}

export interface WeatherApiData {
  location: {
    name: string;
    country: string;
    lat: number;
    lon: number;
  };
  current: {
    temp_c: number;
    temp_f: number;
    humidity: number;
    pressure_mb: number;
    wind_kph: number;
    wind_dir: string;
    condition: {
      text: string;
      icon: string;
    };
    uv: number;
    feelslike_c: number;
  };
  forecast?: {
    forecastday: Array<{
      date: string;
      day: {
        maxtemp_c: number;
        mintemp_c: number;
        condition: {
          text: string;
          icon: string;
        };
      };
    }>;
  };
}

export class EnhancedApiService {
  private geminiApiKey: string;
  private openWeatherApiKey: string;
  private weatherApiKey: string;

  constructor() {
    this.geminiApiKey = API_KEYS.GEMINI;
    this.openWeatherApiKey = API_KEYS.OPENWEATHERMAP;
    this.weatherApiKey = API_KEYS.WEATHERAPI;
  }

  async generateGeminiResponse(prompt: string, context?: string): Promise<GeminiResponse> {
    try {

      if (!this.geminiApiKey || this.geminiApiKey === 'YOUR_GEMINI_API_KEY_HERE') {
        throw new Error('Gemini API key not configured');
      }

      const url = `${API_ENDPOINTS.GEMINI.CHAT}?key=${this.geminiApiKey}`;

      const requestBody = {
        contents: [{
          parts: [{
            text: context ? `${context}\n\nUser: ${prompt}` : prompt
          }]
        }],
        generationConfig: {
          temperature: 0.7,
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 1024,
        }
      };

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Gemini API Error Response:', errorText);
        throw new Error(`Gemini API Error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();

      if (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) {
        return {
          text: data.candidates[0].content.parts[0].text,
          confidence: 0.9,
          sources: ['Gemini AI']
        };
      } else {
        console.error('Invalid Gemini API response format:', data);
        throw new Error('Invalid Gemini API response format');
      }
    } catch (error) {
      console.error('Gemini API Error:', error);
      throw new Error('Failed to generate AI response');
    }
  }

  async getWeatherApiData(location: string, days: number = 1): Promise<WeatherApiData> {
    try {

      const url = `${API_ENDPOINTS.OPENWEATHERMAP.CURRENT}?q=${encodeURIComponent(location)}&units=metric&appid=${this.openWeatherApiKey}`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`OpenWeatherMap Error: ${response.status}`);
      }

      const data = await response.json();

      return {
        location: {
          name: data.name,
          country: data.sys.country,
          lat: data.coord.lat,
          lon: data.coord.lon
        },
        current: {
          temp_c: data.main.temp,
          temp_f: (data.main.temp * 9/5) + 32,
          humidity: data.main.humidity,
          pressure_mb: data.main.pressure,
          wind_kph: data.wind.speed * 3.6, 
          wind_dir: data.wind.deg ? `${data.wind.deg}°` : 'N/A',
          condition: {
            text: data.weather[0].description,
            icon: data.weather[0].icon
          },
          uv: 0, 
          feelslike_c: data.main.feels_like
        },
        forecast: undefined 
      };
    } catch (error) {
      console.error('OpenWeatherMap Error:', error);
      throw new Error('Failed to fetch weather data');
    }
  }

  async getOpenWeatherData(city: string, country?: string): Promise<any> {
    try {
      const location = country ? `${city},${country}` : city;
      const url = `${API_ENDPOINTS.OPENWEATHERMAP.CURRENT}?q=${location}&units=metric&appid=${this.openWeatherApiKey}`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`OpenWeatherMap Error: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('OpenWeatherMap Error:', error);
      throw new Error('Failed to fetch OpenWeatherMap data');
    }
  }

  async getWeatherAnalysis(location: string): Promise<{
    weather: WeatherApiData;
    analysis: GeminiResponse;
  }> {
    try {

      const weatherData = await this.getWeatherApiData(location, 3);

      const weatherContext = `
        Current Weather Data for ${location}:
        - Temperature: ${weatherData.current.temp_c}°C (feels like ${weatherData.current.feelslike_c}°C)
        - Humidity: ${weatherData.current.humidity}%
        - Pressure: ${weatherData.current.pressure_mb} mb
        - Wind: ${weatherData.current.wind_kph} km/h ${weatherData.current.wind_dir}
        - UV Index: ${weatherData.current.uv}
        - Condition: ${weatherData.current.condition.text}
      `;

      const analysis = await this.generateGeminiResponse(
        `Analyze this weather data and provide insights about current conditions, recommendations for outdoor activities, and any weather-related warnings or advice.`,
        weatherContext
      );

      return {
        weather: weatherData,
        analysis
      };
    } catch (error) {
      console.error('Weather Analysis Error:', error);
      throw new Error('Failed to analyze weather data');
    }
  }

  async getEnhancedChatbotResponse(userMessage: string, location?: string): Promise<GeminiResponse> {
    try {
      let context = "You are a helpful weather assistant. Provide accurate, helpful information about weather, climate, and meteorological topics.";

      if (location) {
        try {
          const weatherData = await this.getWeatherApiData(location);
          context += `\n\nCurrent weather context for ${location}: ${weatherData.current.condition.text}, ${weatherData.current.temp_c}°C, humidity ${weatherData.current.humidity}%.`;
        } catch (error) {
          console.log('Could not fetch weather context:', error);
        }
      }

      return await this.generateGeminiResponse(userMessage, context);
    } catch (error) {
      console.error('Enhanced Chatbot Error:', error);

      let fallbackText = `I understand you're asking about "${userMessage}". While I'm having trouble connecting to my AI service right now, I can still help you with weather information.`;

      if (location) {
        fallbackText += ` Please try using the Global Weather tab to get real-time weather data for ${location} or any other location worldwide.`;
      } else {
        fallbackText += ` Please try using the Global Weather tab to get real-time weather data for any location worldwide.`;
      }

      return {
        text: fallbackText,
        confidence: 0.5,
        sources: ['Fallback Response']
      };
    }
  }

  async getWeatherForecastWithInsights(location: string, days: number = 5): Promise<{
    forecast: WeatherApiData;
    insights: GeminiResponse;
  }> {
    try {
      const weatherData = await this.getWeatherApiData(location, days);

      const forecastContext = `
        Weather Forecast for ${location} (${days} days):
        ${weatherData.forecast?.forecastday.map(day => 
          `${day.date}: ${day.day.condition.text}, High: ${day.day.maxtemp_c}°C, Low: ${day.day.mintemp_c}°C`
        ).join('\n')}
      `;

      const insights = await this.generateGeminiResponse(
        `Analyze this weather forecast and provide insights about the upcoming weather patterns, recommendations for planning activities, and any notable weather changes or trends.`,
        forecastContext
      );

      return {
        forecast: weatherData,
        insights
      };
    } catch (error) {
      console.error('Weather Forecast Analysis Error:', error);
      throw new Error('Failed to analyze weather forecast');
    }
  }

  private async retryRequest<T>(requestFn: () => Promise<T>, maxRetries: number = 3): Promise<T> {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await requestFn();
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
      }
    }
    throw new Error('Max retries exceeded');
  }
}

export const enhancedApiService = new EnhancedApiService();
