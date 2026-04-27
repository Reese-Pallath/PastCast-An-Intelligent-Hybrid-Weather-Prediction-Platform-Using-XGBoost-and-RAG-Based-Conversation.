export const API_KEYS = {
  GEMINI: process.env.REACT_APP_GEMINI || 'YOUR_GEMINI_API_KEY_HERE',
  OPENWEATHERMAP:
    process.env.REACT_APP_OPENWEATHER_API || 'YOUR_OPENWEATHERMAP_API_KEY',
  WEATHERAPI: '', 
};

export const API_ENDPOINTS = {
  GEMINI: {
    CHAT:
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent',
  },
  OPENWEATHERMAP: {
    CURRENT: 'https://api.openweathermap.org/data/2.5/weather',
  },
};