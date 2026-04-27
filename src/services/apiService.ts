
export interface ApiConfig {
  weather: {
    openWeatherMap: string;
    weatherStack: string;
  };
  news: {
    newsApi: string;
    gNews: string;
  };
  finance: {
    alphaVantage: string;
    yahooFinance: string;
  };
  knowledge: {
    wikipedia: string;
    duckDuckGo: string;
  };
}

export interface WeatherData {
  location: string;
  temperature: number;
  humidity: number;
  pressure: number;
  windSpeed: number;
  description: string;
  icon: string;
  country: string;
  coordinates: {
    lat: number;
    lon: number;
  };
}

export interface NewsArticle {
  title: string;
  description: string;
  url: string;
  publishedAt: string;
  source: string;
  imageUrl?: string;
}

export interface StockData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap?: number;
  high: number;
  low: number;
  open: number;
}

export interface KnowledgeResult {
  title: string;
  description: string;
  url: string;
  source: string;
  imageUrl?: string;
}

export class ApiService {
  private config: ApiConfig;

  constructor(config: ApiConfig) {
    this.config = config;
  }

  async getCurrentWeather(city: string, country?: string): Promise<WeatherData> {
    try {
      const location = country ? `${city},${country}` : city;
      const url = `https://api.openweathermap.org/data/2.5/weather?q=${location}&units=metric&appid=${this.config.weather.openWeatherMap}`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Weather API Error: ${response.status}`);
      }

      const data = await response.json();

      return {
        location: data.name,
        temperature: data.main.temp,
        humidity: data.main.humidity,
        pressure: data.main.pressure,
        windSpeed: data.wind.speed,
        description: data.weather[0].description,
        icon: data.weather[0].icon,
        country: data.sys.country,
        coordinates: {
          lat: data.coord.lat,
          lon: data.coord.lon
        }
      };
    } catch (error) {
      console.error('Weather API Error:', error);
      throw new Error('Failed to fetch weather data');
    }
  }

  async getWeatherForecast(city: string, country?: string, days: number = 5): Promise<WeatherData[]> {
    try {
      const location = country ? `${city},${country}` : city;
      const url = `https://api.openweathermap.org/data/2.5/forecast?q=${location}&units=metric&appid=${this.config.weather.openWeatherMap}`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Weather API Error: ${response.status}`);
      }

      const data = await response.json();

      const dailyForecasts = data.list.slice(0, days * 8).filter((_: any, index: number) => index % 8 === 0);

      return dailyForecasts.map((item: any) => ({
        location: data.city.name,
        temperature: item.main.temp,
        humidity: item.main.humidity,
        pressure: item.main.pressure,
        windSpeed: item.wind.speed,
        description: item.weather[0].description,
        icon: item.weather[0].icon,
        country: data.city.country,
        coordinates: {
          lat: data.city.coord.lat,
          lon: data.city.coord.lon
        }
      }));
    } catch (error) {
      console.error('Weather Forecast API Error:', error);
      throw new Error('Failed to fetch weather forecast');
    }
  }

  async getNews(country: string = 'in', category?: string, query?: string): Promise<NewsArticle[]> {
    try {
      let url = `https://newsapi.org/v2/top-headlines?country=${country}&apiKey=${this.config.news.newsApi}`;

      if (category) {
        url += `&category=${category}`;
      }

      if (query) {
        url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&apiKey=${this.config.news.newsApi}`;
      }

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`News API Error: ${response.status}`);
      }

      const data = await response.json();

      return data.articles.map((article: any) => ({
        title: article.title,
        description: article.description,
        url: article.url,
        publishedAt: article.publishedAt,
        source: article.source.name,
        imageUrl: article.urlToImage
      }));
    } catch (error) {
      console.error('News API Error:', error);
      throw new Error('Failed to fetch news data');
    }
  }

  async getGlobalNews(query: string): Promise<NewsArticle[]> {
    try {
      const url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&sortBy=publishedAt&apiKey=${this.config.news.newsApi}`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`News API Error: ${response.status}`);
      }

      const data = await response.json();

      return data.articles.map((article: any) => ({
        title: article.title,
        description: article.description,
        url: article.url,
        publishedAt: article.publishedAt,
        source: article.source.name,
        imageUrl: article.urlToImage
      }));
    } catch (error) {
      console.error('Global News API Error:', error);
      throw new Error('Failed to fetch global news');
    }
  }

  async getStockData(symbol: string): Promise<StockData> {
    try {

      const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${this.config.finance.alphaVantage}`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Finance API Error: ${response.status}`);
      }

      const data = await response.json();
      const quote = data['Global Quote'];

      if (!quote || !quote['01. symbol']) {
        throw new Error('Invalid stock symbol or API response');
      }

      return {
        symbol: quote['01. symbol'],
        price: parseFloat(quote['05. price']),
        change: parseFloat(quote['09. change']),
        changePercent: parseFloat(quote['10. change percent'].replace('%', '')),
        volume: parseInt(quote['06. volume']),
        high: parseFloat(quote['03. high']),
        low: parseFloat(quote['04. low']),
        open: parseFloat(quote['02. open'])
      };
    } catch (error) {
      console.error('Stock API Error:', error);
      throw new Error('Failed to fetch stock data');
    }
  }

  async getIndianStocks(): Promise<StockData[]> {
    const indianStocks = ['RELIANCE.BSE', 'TCS.BSE', 'HDFCBANK.BSE', 'INFY.BSE', 'HINDUNILVR.BSE'];
    const stockData: StockData[] = [];

    for (const symbol of indianStocks) {
      try {
        const data = await this.getStockData(symbol);
        stockData.push(data);
      } catch (error) {
        console.error(`Failed to fetch data for ${symbol}:`, error);
      }
    }

    return stockData;
  }

  async searchWikipedia(query: string): Promise<KnowledgeResult[]> {
    try {
      const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(query)}`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Wikipedia API Error: ${response.status}`);
      }

      const data = await response.json();

      return [{
        title: data.title,
        description: data.extract,
        url: data.content_urls?.desktop?.page || data.content_urls?.mobile?.page,
        source: 'Wikipedia',
        imageUrl: data.thumbnail?.source
      }];
    } catch (error) {
      console.error('Wikipedia API Error:', error);
      throw new Error('Failed to fetch Wikipedia data');
    }
  }

  async searchDuckDuckGo(query: string): Promise<KnowledgeResult[]> {
    try {

      const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`DuckDuckGo API Error: ${response.status}`);
      }

      const data = await response.json();

      const results: KnowledgeResult[] = [];

      if (data.Abstract) {
        results.push({
          title: data.Heading || query,
          description: data.Abstract,
          url: data.AbstractURL,
          source: 'DuckDuckGo',
          imageUrl: data.Image
        });
      }

      if (data.RelatedTopics) {
        data.RelatedTopics.slice(0, 3).forEach((topic: any) => {
          if (topic.Text && topic.FirstURL) {
            results.push({
              title: topic.Text.split(' - ')[0],
              description: topic.Text,
              url: topic.FirstURL,
              source: 'DuckDuckGo'
            });
          }
        });
      }

      return results;
    } catch (error) {
      console.error('DuckDuckGo API Error:', error);
      throw new Error('Failed to fetch DuckDuckGo data');
    }
  }

  async getLocationFromCoordinates(lat: number, lon: number): Promise<string> {
    try {
      const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${this.config.weather.openWeatherMap}`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Location API Error: ${response.status}`);
      }

      const data = await response.json();
      return `${data.name}, ${data.sys.country}`;
    } catch (error) {
      console.error('Location API Error:', error);
      return `${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E`;
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

export const defaultApiConfig: ApiConfig = {
  weather: {
    openWeatherMap: process.env.REACT_APP_OPENWEATHER_API || process.env.REACT_APP_OPENWEATHER_API_KEY || '',
    weatherStack: ''
  },
  news: {
    newsApi: 'YOUR_NEWSAPI_KEY',
    gNews: 'YOUR_GNEWS_API_KEY'
  },
  finance: {
    alphaVantage: 'YOUR_ALPHAVANTAGE_API_KEY',
    yahooFinance: 'YOUR_YAHOO_FINANCE_API_KEY'
  },
  knowledge: {
    wikipedia: 'YOUR_WIKIPEDIA_API_KEY',
    duckDuckGo: 'YOUR_DUCKDUCKGO_API_KEY'
  }
};

export const apiService = new ApiService(defaultApiConfig);
