import React from 'react';
import { LocationInput } from '../types/weather';
import LocationMap from './LocationMap';

interface WeatherFormProps {
  onSubmit: (location: LocationInput, startDate: string, endDate?: string, datasetMode?: 'IMD' | 'Global' | 'Combined') => Promise<void>;
  isLoading: boolean;
}

const WeatherForm: React.FC<WeatherFormProps> = ({ onSubmit, isLoading }) => {
  const [selectedLocation, setSelectedLocation] = React.useState<{
    lat: number;
    lng: number;
    address?: string;
  } | null>({
    lat: 12.9716,
    lng: 77.5946,
    address: 'Bengaluru, Karnataka, India'
  });
  const [isGettingLocation, setIsGettingLocation] = React.useState(false);
  const [startDate, setStartDate] = React.useState('');
  const [endDate, setEndDate] = React.useState('');
  const [datasetMode, setDatasetMode] = React.useState<'IMD' | 'Global' | 'Combined'>('Combined');

  const handleLocationSelect = (lat: number, lng: number, address?: string) => {
    setSelectedLocation({ lat, lng, address });
  };

  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by this browser.');
      return;
    }

    setIsGettingLocation(true);

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;

        try {

          const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}&addressdetails=1`
          );
          const data = await response.json();
          const address = data.display_name || `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;

          setSelectedLocation({
            lat: latitude,
            lng: longitude,
            address: address
          });
        } catch (error) {
          console.error('Error getting address:', error);
          setSelectedLocation({
            lat: latitude,
            lng: longitude,
            address: `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`
          });
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
        alert(errorMessage);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 300000 
      }
    );
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedLocation) {
      alert('Please select a location on the map');
      return;
    }

    if (!startDate) {
      alert('Please select a start date');
      return;
    }

    onSubmit({ 
      latitude: selectedLocation.lat, 
      longitude: selectedLocation.lng, 
      city_name: selectedLocation.address 
    }, startDate, endDate || undefined, datasetMode);
  };

  return (
    <div className="relative">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-cyan-500/5 rounded-2xl blur-xl"></div>
      <div className="relative bg-white/5 backdrop-blur-md rounded-2xl p-8 border border-white/10 shadow-2xl">
        <div className="flex items-center space-x-3 mb-8">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
            </svg>
          </div>
          <h2 className="text-3xl font-bold text-white bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">
            Weather Probability Analysis
          </h2>
        </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-white font-semibold text-lg flex items-center space-x-2">
              <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <span>Select Location</span>
            </h3>
            <button
              type="button"
              onClick={getCurrentLocation}
              disabled={isGettingLocation}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 disabled:from-blue-500/50 disabled:to-cyan-500/50 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg shadow-blue-500/25 flex items-center space-x-2"
            >
              {isGettingLocation ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Getting Location...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <span>Use My Location</span>
                </>
              )}
            </button>
          </div>
          {selectedLocation && (
            <div className="p-4 bg-white/5 rounded-xl border border-white/10 shadow-lg">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                  <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div className="flex-1">
                  <p className="text-white font-medium">
                    {selectedLocation.address}
                  </p>
                  <p className="text-white/60 text-sm">
                    Coordinates: {selectedLocation.lat.toFixed(4)}, {selectedLocation.lng.toFixed(4)}
                  </p>
                </div>
              </div>
            </div>
          )}
          <LocationMap 
            onLocationSelect={handleLocationSelect}
            initialLat={selectedLocation?.lat || 12.9716}
            initialLng={selectedLocation?.lng || 77.5946}
            height="300px"
          />
        </div>

        {}
        <div className="space-y-4">
          <h3 className="text-white font-semibold text-lg flex items-center space-x-2">
            <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span>Date Range</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="block text-white text-sm font-medium">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
                required
              />
            </div>
            <div className="space-y-2">
              <label className="block text-white text-sm font-medium">End Date (Optional)</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
              />
            </div>
          </div>
        </div>

        {}
        <div className="space-y-4">
          <h3 className="text-white font-semibold text-lg flex items-center space-x-2">
            <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span>Dataset Source</span>
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {(['IMD', 'Global', 'Combined'] as const).map(mode => (
              <button
                key={mode}
                type="button"
                onClick={() => setDatasetMode(mode)}
                className={`px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-300 transform hover:scale-105 ${
                  datasetMode === mode
                    ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg shadow-blue-500/25'
                    : 'bg-white/5 text-white/70 hover:bg-white/10 border border-white/10'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
          <p className="text-white/60 text-sm">IMD focuses on India; Global uses NASA/NOAA; Combined merges when available.</p>
        </div>

        {}
        <button
          type="submit"
          disabled={isLoading || !selectedLocation}
          className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-bold py-4 px-8 rounded-xl transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-500/25 flex items-center justify-center space-x-3"
        >
          {isLoading ? (
            <>
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Analyzing Weather Data...</span>
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span>Get Weather Probabilities</span>
            </>
          )}
        </button>
      </form>

      {}
      <div className="mt-6 p-4 bg-white/5 rounded-lg">
        <h3 className="text-white font-medium mb-2">What PastCast Analyzes:</h3>
        <ul className="text-white/80 text-sm space-y-1">
          <li>• Rain probability (precipitation {'>'} 1mm/day)</li>
          <li>• Extreme heat probability (temperature {'>'} 40°C for India, 35°C elsewhere)</li>
          <li>• High wind probability (wind speed {'>'} 20 km/h)</li>
          <li>• Cloudy conditions (cloud cover {'>'} 70%)</li>
          <li>• Overall good weather probability</li>
        </ul>
      </div>
      </div>
    </div>
  );
};

export default WeatherForm;
