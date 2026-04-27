import React, { useState } from 'react';
import {
  WeatherAPIHandler,
  GlobalWeatherData,
  GlobalWeatherForecastDay,
} from '../lib/weatherAPIHandler';

interface GlobalWeatherProps {
  onLocationSelect?: (lat: number, lng: number, name: string) => void;
}

const GlobalWeather: React.FC<GlobalWeatherProps> = ({ onLocationSelect }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [weatherData, setWeatherData] = useState<GlobalWeatherData | null>(null);
  const [searchResults, setSearchResults] = useState<
    Array<{ name: string; country: string; lat: number; lon: number }>
  >([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isGettingLocation, setIsGettingLocation] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const results = await WeatherAPIHandler.searchLocations(searchQuery, 5);
      setSearchResults(results);
    } catch (err) {
      setError('Failed to search locations');
      console.error('Search error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLocationSelect = async (lat: number, lon: number, name: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const weather = await WeatherAPIHandler.getWeatherByCoordinates(lat, lon);
      setWeatherData(weather);

      if (onLocationSelect) {
        onLocationSelect(lat, lon, name);
      }
    } catch (err) {
      setError('Failed to fetch weather data');
      console.error('Weather fetch error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by this browser.');
      return;
    }

    setIsGettingLocation(true);
    setError(null);

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;

        try {

          const weather = await WeatherAPIHandler.getWeatherByCoordinates(
            latitude,
            longitude
          );
          setWeatherData(weather);

          const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}&addressdetails=1`
          );
          const data = await response.json();
          const address =
            data.display_name || `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;

          if (onLocationSelect) {
            onLocationSelect(latitude, longitude, address);
          }
        } catch (error) {
          console.error('Error getting weather or address:', error);
          setError('Failed to get weather data for your location');
        }

        setIsGettingLocation(false);
      },
      (error) => {
        console.error('Error getting location:', error);
        setIsGettingLocation(false);

        let errorMessage = 'Unable to get your location. ';
        switch (error.code) {
          case error.PERMISSION_DENIED:
            errorMessage += 'Please allow location access and try again.';
            break;
          case error.POSITION_UNAVAILABLE:
            errorMessage += 'Location information is unavailable.';
            break;
          case error.TIMEOUT:
            errorMessage += 'Location request timed out.';
            break;
          default:
            errorMessage += 'An unknown error occurred.';
            break;
        }
        setError(errorMessage);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 300000, 
      }
    );
  };

  const getWeatherIconUrl = (icon: string) => {
    return `https://openweathermap.org/img/wn/${icon}@2x.png`;
  };

  const formatTemperature = (temp: number) => {
    return `${Math.round(temp)}°C`;
  };

  const formatWindSpeed = (speed: number) => {
    return `${Math.round(speed * 3.6)} km/h`; 
  };

  return (
    <div className="relative">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-cyan-500/5 rounded-2xl blur-xl"></div>
      <div className="relative bg-white/5 backdrop-blur-md rounded-2xl p-8 border border-white/10 shadow-2xl">
        <div className="flex items-center space-x-3 mb-8">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-xl flex items-center justify-center">
            <svg
              className="w-6 h-6 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h2 className="text-3xl font-bold text-white bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">
            Global Weather
          </h2>
        </div>

        {}
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center">
              <svg
                className="w-4 h-4 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>
            <h3 className="text-white font-semibold text-lg">Search Global Weather</h3>
          </div>

          <div className="flex gap-3 mb-6">
            <input
              type="text"
              placeholder="Search for any city worldwide..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              className="flex-1 px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
            />
            <button
              onClick={handleSearch}
              disabled={isLoading || !searchQuery.trim()}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-semibold rounded-xl transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-500/25 flex items-center space-x-2"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                  <span>Search</span>
                </>
              )}
            </button>
            <button
              onClick={getCurrentLocation}
              disabled={isGettingLocation}
              className="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white font-semibold rounded-xl transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-green-500/25 flex items-center space-x-2"
            >
              {isGettingLocation ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Getting...</span>
                </>
              ) : (
                <>
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                  </svg>
                  <span>My Location</span>
                </>
              )}
            </button>
          </div>

          {}
          {searchResults.length > 0 && (
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 mb-6 border border-white/10 shadow-lg animate-in fade-in-50 duration-500">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center">
                  <svg
                    className="w-3 h-3 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                </div>
                <h3 className="text-white font-semibold text-lg">Search Results</h3>
              </div>
              <div className="space-y-3">
                {searchResults.map((result, index) => (
                  <button
                    key={index}
                    onClick={() =>
                      handleLocationSelect(result.lat, result.lon, result.name)
                    }
                    className="w-full text-left p-4 bg-white/5 rounded-xl hover:bg-white/10 transition-all duration-300 transform hover:scale-105 border border-white/10 hover:border-white/20"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-white font-semibold text-lg">
                          {result.name}
                        </div>
                        <div className="text-white/70 text-sm">{result.country}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-white/50 text-xs">
                          {result.lat.toFixed(4)}°N, {result.lon.toFixed(4)}°E
                        </div>
                        <div className="w-6 h-6 bg-blue-500/20 rounded-lg flex items-center justify-center mt-1">
                          <svg
                            className="w-3 h-3 text-blue-400"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                            />
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                            />
                          </svg>
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-6 backdrop-blur-sm">
              <div className="flex items-center space-x-3">
                <div className="w-6 h-6 bg-red-500/20 rounded-lg flex items-center justify-center">
                  <svg
                    className="w-3 h-3 text-red-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
                <p className="text-red-200 font-medium">{error}</p>
              </div>
            </div>
          )}
        </div>

        {}
        {weatherData && (
          <div className="space-y-8 animate-in fade-in-50 duration-500">
            {}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 shadow-2xl hover:shadow-blue-500/10 transition-all duration-300">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center space-x-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-xl flex items-center justify-center">
                    <svg
                      className="w-6 h-6 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-white bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">
                      {weatherData.location.name}, {weatherData.location.country}
                    </h3>
                    <p className="text-white/70 text-sm font-medium">
                      {weatherData.location.lat.toFixed(4)}°N,{' '}
                      {weatherData.location.lon.toFixed(4)}°E
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-5xl font-bold text-white bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                    {formatTemperature(weatherData.current.temperature)}
                  </div>
                  <div className="text-white/70 capitalize text-lg font-medium">
                    {weatherData.current.description}
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-center mb-8">
                <div className="relative">
                  <img
                    src={getWeatherIconUrl(weatherData.current.icon)}
                    alt={weatherData.current.description}
                    className="w-32 h-32 animate-pulse"
                  />
                  <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-400 rounded-full animate-ping"></div>
                </div>
              </div>

              {}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 text-center border border-white/10 shadow-lg hover:shadow-blue-500/10 transition-all duration-300 transform hover:scale-105">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <svg
                      className="w-4 h-4 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"
                      />
                    </svg>
                  </div>
                  <div className="text-white/70 text-sm mb-2 font-medium">Humidity</div>
                  <div className="text-white font-bold text-2xl">
                    {weatherData.current.humidity}%
                  </div>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 text-center border border-white/10 shadow-lg hover:shadow-blue-500/10 transition-all duration-300 transform hover:scale-105">
                  <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-400 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <svg
                      className="w-4 h-4 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                      />
                    </svg>
                  </div>
                  <div className="text-white/70 text-sm mb-2 font-medium">Pressure</div>
                  <div className="text-white font-bold text-2xl">
                    {weatherData.current.pressure} hPa
                  </div>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 text-center border border-white/10 shadow-lg hover:shadow-blue-500/10 transition-all duration-300 transform hover:scale-105">
                  <div className="w-8 h-8 bg-gradient-to-br from-yellow-500 to-orange-400 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <svg
                      className="w-4 h-4 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10"
                      />
                    </svg>
                  </div>
                  <div className="text-white/70 text-sm mb-2 font-medium">
                    Wind Speed
                  </div>
                  <div className="text-white font-bold text-2xl">
                    {formatWindSpeed(weatherData.current.windSpeed)}
                  </div>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 text-center border border-white/10 shadow-lg hover:shadow-blue-500/10 transition-all duration-300 transform hover:scale-105">
                  <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-400 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <svg
                      className="w-4 h-4 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                    </svg>
                  </div>
                  <div className="text-white/70 text-sm mb-2 font-medium">
                    Coordinates
                  </div>
                  <div className="text-white font-bold text-sm">
                    {weatherData.location.lat.toFixed(2)}°,{' '}
                    {weatherData.location.lon.toFixed(2)}°
                  </div>
                </div>
              </div>
            </div>

            {}
            {weatherData.forecast.length > 0 && (
              <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 shadow-2xl hover:shadow-blue-500/10 transition-all duration-300">
                <div className="flex items-center space-x-3 mb-6">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center">
                    <svg
                      className="w-4 h-4 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                      />
                    </svg>
                  </div>
                  <h3 className="text-white font-bold text-xl">5-Day Forecast</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
                  {weatherData.forecast.map(
                    (day: GlobalWeatherForecastDay, index: number) => (
                      <div
                        key={index}
                        className="bg-white/5 backdrop-blur-sm rounded-xl p-6 text-center border border-white/10 shadow-lg hover:shadow-blue-500/10 transition-all duration-300 transform hover:scale-105 animate-in fade-in-50"
                        style={{ animationDelay: `${index * 100}ms` }}
                      >
                        <div className="text-white/70 text-sm mb-4 font-medium">
                          {new Date(day.date).toLocaleDateString('en-US', {
                            weekday: 'short',
                            month: 'short',
                            day: 'numeric',
                          })}
                        </div>
                        <div className="flex justify-center mb-4">
                          <img
                            src={getWeatherIconUrl(day.icon)}
                            alt={day.description}
                            className="w-16 h-16 animate-pulse"
                          />
                        </div>
                        <div className="text-white font-bold text-2xl mb-2">
                          {formatTemperature(day.temperature.max)}
                        </div>
                        <div className="text-white/70 text-lg mb-3">
                          {formatTemperature(day.temperature.min)}
                        </div>
                        <div className="text-white/60 text-sm capitalize mb-3">
                          {day.description}
                        </div>
                        <div className="bg-blue-500/20 text-blue-300 text-xs px-3 py-1 rounded-full">
                          Rain: {Math.round(day.rainProbability * 100)}%
                        </div>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {}
        {!weatherData && (
          <div className="text-center py-12">
            <div className="mb-6">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <svg
                  className="w-10 h-10 text-blue-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1}
                    d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
            </div>
            <h3 className="text-white text-xl font-semibold mb-2">
              Search Global Weather
            </h3>
            <p className="text-white/70 text-lg mb-2">
              Search for any city worldwide to get current weather data
            </p>
            <p className="text-white/60 text-sm">
              Supports cities from all countries and continents
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default GlobalWeather;