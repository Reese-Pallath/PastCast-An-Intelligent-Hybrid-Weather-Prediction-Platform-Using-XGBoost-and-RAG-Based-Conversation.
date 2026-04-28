import React from 'react';
import { WeatherData } from '../types/weather';

export interface WeatherResultsProps {
  data: WeatherData;
  isComparison?: boolean;
}

const WeatherResults: React.FC<WeatherResultsProps> = ({ data, isComparison = false }) => {
  const { location, probabilities, ai_insights, data_sources, analysis_period } = data;

  const conditions = Object.entries(probabilities).filter(([key, value]) => 
    key !== 'summary' && 'label' in value
  );

  return (
    <div className={`relative ${isComparison ? 'h-fit' : ''}`}>
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-cyan-500/5 rounded-2xl blur-xl"></div>
      <div className="relative bg-white/5 backdrop-blur-md rounded-2xl p-8 border border-white/10 shadow-2xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-8 gap-4">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <div className="min-w-0 flex-1">
              <h3 className="text-xl sm:text-2xl font-bold text-white bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent break-words">
                {location.city_name || `${location.latitude.toFixed(2)}°N, ${location.longitude.toFixed(2)}°E`}
              </h3>
              <p className="text-white/70 text-sm font-medium">{analysis_period}</p>
            </div>
          </div>
          <div className="text-right sm:text-right">
            <div className="text-lg sm:text-xl font-bold text-white bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              {probabilities.summary.risk_level}
            </div>
            <div className="text-sm text-white/70 font-medium">
              {probabilities.summary.data_quality} Quality
            </div>
          </div>
        </div>

      {}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {conditions.map(([key, condition], index) => {
          if ('label' in condition) {
            const icon: Record<string, string> = {
              rain: '🌧️', cloudy: '☁️', extreme_heat: '☀️',
              high_wind: '💨', good_weather: '✅',
            };
            return (
              <div
                key={key}
                className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 shadow-lg hover:shadow-blue-500/10 transition-all duration-300 transform hover:scale-105 animate-in fade-in-50 relative overflow-hidden group"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>

                {/* Card title + icon — icon sits inline beside the title, vertically centred */}
                <div className="mb-4 flex items-center gap-2">
                  <h4 className="text-white font-semibold text-base leading-tight">
                    {condition.description}
                  </h4>
                  {icon[key] && (
                    <span className="text-base leading-none flex-shrink-0 opacity-90">
                      {icon[key]}
                    </span>
                  )}
                </div>

                {/* Probability + bar + explanation */}
                <div className="space-y-3">
                  {condition.probability === null ? (
                    <div className="text-white/70 text-sm italic">No data available</div>
                  ) : (
                    <>
                      <div className="text-3xl font-bold text-white bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                        {condition.probability.toFixed(1)}%
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full transition-all duration-700 shadow-lg shadow-blue-500/25"
                          style={{ width: `${Math.min(100, Math.max(0, condition.probability))}%` }}
                        ></div>
                      </div>
                      <div className="pt-1 border-t border-white/5">
                        <span className="text-white/60 text-xs font-semibold uppercase tracking-wider">
                          {condition.label}
                        </span>
                        {condition.threshold && (
                          <p className="text-white/50 text-xs mt-1 leading-relaxed">
                            {condition.threshold}
                          </p>
                        )}
                      </div>
                    </>
                  )}
                </div>
              </div>
            );
          }
          return null;
        })}
      </div>

      {}
      {ai_insights && !isComparison && (
        <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 rounded-xl p-6 mb-8 border border-white/10 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h4 className="text-white font-semibold text-lg">AI Weather Insights</h4>
            </div>
            <div className="group relative">
              <button aria-label="How probabilities are calculated" className="text-blue-300 text-sm hover:text-blue-200 transition-colors">How it works</button>
              <div className="invisible opacity-0 group-hover:visible group-hover:opacity-100 transition-opacity absolute right-0 mt-2 w-72 bg-black/80 text-white text-xs p-3 rounded-lg border border-white/20 z-10">
                Probabilities are computed from historical datasets over the selected date range. IMD covers India; Global uses NASA/NOAA archives. Combined merges both where available. Missing data shows as "No data available".
              </div>
            </div>
          </div>
          <p className="text-white/80 text-sm leading-relaxed">{ai_insights}</p>
        </div>
      )}

      {}
      {!isComparison && (
        <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 shadow-lg">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-400 rounded-lg flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h4 className="text-white font-semibold text-lg">Data Sources</h4>
          </div>
          <div className="flex flex-wrap gap-3 mb-4">
            {data_sources.map((source, index) => (
              <span
                key={index}
                className="bg-gradient-to-r from-blue-500/20 to-cyan-500/20 text-blue-200 px-4 py-2 rounded-lg text-sm font-medium border border-blue-400/20"
              >
                {source}
              </span>
            ))}
          </div>
          <div className="text-white/60 text-sm">
            Analysis based on <span className="font-semibold text-white">{probabilities.summary.data_points}</span> data points from <span className="font-semibold text-white">{probabilities.summary.date_range}</span>
          </div>
        </div>
      )}
      </div>
    </div>
  );
};

export default WeatherResults;
