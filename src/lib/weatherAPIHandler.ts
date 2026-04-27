import { defaultApiConfig } from '../services/apiService';

export interface GlobalWeatherLocation {
  name: string;
  country: string;
  lat: number;
  lon: number;
}

export interface GlobalWeatherCurrent {
  temperature: number;
  humidity: number;
  pressure: number;
  windSpeed: number;
  description: string;
  icon: string;
}

export interface GlobalWeatherForecastDay {
  date: string;
  icon: string;
  description: string;
  temperature: {
    max: number;
    min: number;
  };
  rainProbability: number;
}

export interface GlobalWeatherData {
  location: GlobalWeatherLocation;
  current: GlobalWeatherCurrent;
  forecast: GlobalWeatherForecastDay[];
}

const OPENWEATHER_API_KEY =
  process.env.REACT_APP_OPENWEATHER_API || defaultApiConfig.weather.openWeatherMap;

const BASE_URL = 'https://api.openweathermap.org/data/2.5';

async function fetchJson(url: string): Promise<any> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Weather API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const WeatherAPIHandler = {
  async searchLocations(query: string, limit: number = 5): Promise<GlobalWeatherLocation[]> {
    if (!OPENWEATHER_API_KEY || OPENWEATHER_API_KEY.startsWith('YOUR_')) {
      throw new Error('OpenWeather API key not configured for frontend');
    }

    const url = `https://api.openweathermap.org/geo/1.0/direct?q=${encodeURIComponent(query)}&limit=${limit}&appid=${OPENWEATHER_API_KEY}`;

    const data = await fetchJson(url);
    return (data as any[]).map((item) => ({
      name: item.name,
      country: item.country,
      lat: item.lat,
      lon: item.lon,
    }));
  },

  async getWeatherByCoordinates(lat: number, lon: number): Promise<GlobalWeatherData> {
    if (!OPENWEATHER_API_KEY || OPENWEATHER_API_KEY.startsWith('YOUR_')) {
      throw new Error('OpenWeather API key not configured for frontend');
    }

    const currentUrl = `${BASE_URL}/weather?lat=${lat}&lon=${lon}&units=metric&appid=${OPENWEATHER_API_KEY}`;
    const forecastUrl = `${BASE_URL}/forecast?lat=${lat}&lon=${lon}&units=metric&appid=${OPENWEATHER_API_KEY}`;

    const [current, forecast] = await Promise.all([
      fetchJson(currentUrl),
      fetchJson(forecastUrl).catch(() => null),
    ]);

    const currentData: GlobalWeatherCurrent = {
      temperature: current.main.temp,
      humidity: current.main.humidity,
      pressure: current.main.pressure,
      windSpeed: current.wind.speed,
      description: current.weather?.[0]?.description ?? 'Unknown',
      icon: current.weather?.[0]?.icon ?? '01d',
    };

    let forecastDays: GlobalWeatherForecastDay[] = [];

    if (forecast && Array.isArray(forecast.list)) {
      const byDate: Record<string, any[]> = {};

      for (const entry of forecast.list) {
        const date = entry.dt_txt?.split(' ')[0];
        if (!date) continue;
        if (!byDate[date]) byDate[date] = [];
        byDate[date].push(entry);
      }

      forecastDays = Object.entries(byDate)
        .slice(0, 5)
        .map(([date, entries]) => {
          const temps = (entries as any[]).map((e) => e.main.temp as number);
          const min = Math.min(...temps);
          const max = Math.max(...temps);
          const first = (entries as any[])[0];

          const popValues = (entries as any[])
            .map((e) => (typeof e.pop === 'number' ? e.pop : 0))
            .filter((v) => !Number.isNaN(v));

          const avgPop =
            popValues.length > 0
              ? popValues.reduce((a, b) => a + b, 0) / popValues.length
              : 0;

          return {
            date,
            icon: first.weather?.[0]?.icon ?? '01d',
            description: first.weather?.[0]?.description ?? 'Unknown',
            temperature: { max, min },
            rainProbability: avgPop,
          };
        });
    }

    return {
      location: {
        name: current.name,
        country: current.sys.country,
        lat: current.coord.lat,
        lon: current.coord.lon,
      },
      current: currentData,
      forecast: forecastDays,
    };
  },
};