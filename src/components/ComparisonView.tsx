import React, { useState } from 'react';
import { LocationInput } from '../types/weather';
import LocationMap from './LocationMap';

interface ComparisonViewProps {
  onSubmit: (locations: LocationInput[], startDate: string, endDate?: string, datasetMode?: 'IMD' | 'Global' | 'Combined') => void;
  isLoading: boolean;
}

const ComparisonView: React.FC<ComparisonViewProps> = ({ onSubmit, isLoading }) => {
  const [locations, setLocations] = useState<Array<LocationInput & { id: string }>>([
    { id: '1', latitude: 0, longitude: 0, city_name: '' },
    { id: '2', latitude: 0, longitude: 0, city_name: '' }
  ]);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [useDateRange, setUseDateRange] = useState(false);
  const [datasetMode, setDatasetMode] = useState<'IMD' | 'Global' | 'Combined'>('Combined');

  const addLocation = () => {
    const newId = (locations.length + 1).toString();
    setLocations([...locations, { id: newId, latitude: 0, longitude: 0, city_name: '' }]);
  };

  const removeLocation = (index: number) => {
    if (locations.length > 2) {
      setLocations(locations.filter((_, i) => i !== index));
    }
  };

  const handleLocationSelect = (index: number, lat: number, lng: number, address?: string) => {
    const updated = [...locations];
    updated[index] = { 
      ...updated[index], 
      latitude: lat, 
      longitude: lng, 
      city_name: address 
    };
    setLocations(updated);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const validLocations = locations.filter(loc => 
      loc.latitude !== 0 && loc.longitude !== 0
    );

    if (validLocations.length < 2) {
      alert('Please select at least 2 locations on the maps to compare');
      return;
    }

    if (!startDate) {
      alert('Please select a start date');
      return;
    }

    const locationsToSubmit = validLocations.map(({ id, ...loc }) => loc);
    onSubmit(locationsToSubmit, startDate, useDateRange ? endDate : undefined, datasetMode);
  };

  return (
    <div className="relative">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-cyan-500/5 rounded-2xl blur-xl"></div>
      <div className="relative bg-white/5 backdrop-blur-md rounded-2xl p-8 border border-white/10 shadow-2xl">
        <div className="flex items-center space-x-3 mb-8">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h2 className="text-3xl font-bold text-white bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">
            Compare Weather Probabilities
          </h2>
        </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-white font-semibold text-lg flex items-center space-x-2">
              <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <span>Select Locations to Compare</span>
            </h3>
            <button
              type="button"
              onClick={addLocation}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white text-sm font-semibold rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg shadow-blue-500/25 flex items-center space-x-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              <span>Add Location</span>
            </button>
          </div>

          <div className="space-y-6">
            {locations.map((location, index) => (
              <div key={location.id} className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 shadow-lg hover:shadow-blue-500/10 transition-all duration-300">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center">
                      <span className="text-white font-bold text-sm">{index + 1}</span>
                    </div>
                    <h4 className="text-white font-semibold text-lg">Location {index + 1}</h4>
                  </div>
                  {locations.length > 2 && (
                    <button
                      type="button"
                      onClick={() => removeLocation(index)}
                      className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 hover:text-red-200 text-sm font-medium rounded-lg transition-all duration-300 flex items-center space-x-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                      <span>Remove</span>
                    </button>
                  )}
                </div>

                <LocationMap 
                  onLocationSelect={(lat, lng, address) => handleLocationSelect(index, lat, lng, address)}
                  initialLat={location.latitude || undefined}
                  initialLng={location.longitude || undefined}
                  height="250px"
                />
              </div>
            ))}
          </div>
        </div>

        {}
        <div className="space-y-4">
          <h3 className="text-white font-semibold text-lg flex items-center space-x-2">
            <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span>Analysis Date</span>
          </h3>
          <div className="space-y-2">
            <label className="block text-white text-sm font-medium">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
            />
          </div>
        </div>

        {}
        <div className="flex items-center space-x-4 p-4 bg-white/5 rounded-xl border border-white/10">
          <input
            type="checkbox"
            id="useDateRange"
            checked={useDateRange}
            onChange={(e) => setUseDateRange(e.target.checked)}
            className="w-5 h-5 text-blue-500 bg-white/10 border-white/30 rounded focus:ring-blue-500 focus:ring-2"
          />
          <label htmlFor="useDateRange" className="text-white font-medium">
            Use date range (optional)
          </label>
        </div>

        {}
        {useDateRange && (
          <div className="space-y-2">
            <label className="block text-white text-sm font-medium">End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
            />
          </div>
        )}

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
          disabled={isLoading}
          className={`w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-bold py-4 px-8 rounded-xl transition-all duration-500 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-500/25 flex items-center justify-center space-x-3 relative overflow-hidden ${
            isLoading ? 'animate-pulse' : ''
          }`}
        >
          {}
          {isLoading && (
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse"></div>
          )}

          {isLoading ? (
            <>
              <div className="relative z-10 w-6 h-6 border-3 border-white border-t-transparent rounded-full animate-spin"></div>
              <span className="relative z-10 animate-pulse">Analyzing Weather Data...</span>
              <div className="absolute inset-0 bg-gradient-to-r from-blue-400/20 to-cyan-400/20 animate-pulse"></div>
            </>
          ) : (
            <>
              <svg className="w-6 h-6 transition-transform duration-300 group-hover:rotate-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <span className="transition-all duration-300">Compare Weather Probabilities</span>
            </>
          )}
        </button>
      </form>

      {}
      <div className="mt-8 p-6 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 shadow-lg">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-400 rounded-lg flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-white font-semibold text-lg">Comparison Features</h3>
        </div>
        <ul className="text-white/80 text-sm space-y-2">
          <li className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
            <span>Side-by-side probability analysis</span>
          </li>
          <li className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
            <span>Risk level comparison</span>
          </li>
          <li className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
            <span>Best/worst locations for each condition</span>
          </li>
          <li className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
            <span>Data quality assessment</span>
          </li>
        </ul>
      </div>
      </div>
    </div>
  );
};

export default ComparisonView;
