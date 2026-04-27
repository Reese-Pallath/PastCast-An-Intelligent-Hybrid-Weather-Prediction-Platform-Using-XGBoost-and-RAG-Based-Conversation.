import React from 'react';

interface ComparisonData {
  locations: Array<{
    location: {
      latitude: number;
      longitude: number;
      city_name?: string;
    };
    probabilities: any;
    data_quality: string;
    risk_level: string;
  }>;
  analysis_period: string;
  data_sources: string[];
  comparison_summary: {
    best_locations: { [key: string]: string };
    worst_locations: { [key: string]: string };
    overall_risk: string;
  };
}

interface ComparisonResultsProps {
  data: ComparisonData;
}

const ComparisonResults: React.FC<ComparisonResultsProps> = ({ data }) => {
  const { locations, comparison_summary, analysis_period, data_sources } = data;

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
            Weather Comparison Results
          </h2>
        </div>

        {}
        <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 rounded-xl p-6 mb-8 border border-white/10 shadow-lg animate-in fade-in-50 duration-700 hover:shadow-blue-500/20 transition-all duration-500">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-400 rounded-lg flex items-center justify-center animate-pulse">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-white font-semibold text-lg bg-gradient-to-r from-white to-green-200 bg-clip-text text-transparent">
              Comparison Summary
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-green-400 font-semibold mb-3">Best Locations</h4>
              <div className="space-y-2">
                {Object.entries(comparison_summary.best_locations).map(([condition, location]) => (
                  <div key={condition} className="flex justify-between items-center p-2 bg-green-500/10 rounded-lg">
                    <span className="text-white text-sm capitalize">{condition.replace('_', ' ')}</span>
                    <span className="text-green-300 font-medium text-sm">{location}</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-red-400 font-semibold mb-3">Worst Locations</h4>
              <div className="space-y-2">
                {Object.entries(comparison_summary.worst_locations).map(([condition, location]) => (
                  <div key={condition} className="flex justify-between items-center p-2 bg-red-500/10 rounded-lg">
                    <span className="text-white text-sm capitalize">{condition.replace('_', ' ')}</span>
                    <span className="text-red-300 font-medium text-sm">{location}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 p-4 bg-white/5 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-white font-medium">Overall Risk Level:</span>
              <span className="text-yellow-400 font-semibold">{comparison_summary.overall_risk}</span>
            </div>
          </div>
        </div>

        {}
        <div className="space-y-6">
          <h3 className="text-white font-semibold text-lg flex items-center space-x-2">
            <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <span>Individual Location Analysis</span>
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {locations.map((locationData, index) => (
              <div 
                key={index} 
                className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 shadow-lg hover:shadow-blue-500/10 transition-all duration-500 transform hover:scale-105 animate-in fade-in-50 duration-700"
                style={{ 
                  animationDelay: `${index * 200}ms`,
                  animation: `slideInFromBottom 0.7s ease-out ${index * 0.2}s both`
                }}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center">
                      <span className="text-white font-bold text-sm">{index + 1}</span>
                    </div>
                    <div>
                      <h4 className="text-white font-semibold text-lg">
                        {locationData.location.city_name}
                      </h4>
                      <p className="text-white/70 text-sm">
                        {locationData.location.latitude.toFixed(2)}°N, {locationData.location.longitude.toFixed(2)}°E
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold text-white">
                      {locationData.risk_level}
                    </div>
                    <div className="text-xs text-white/70">
                      {locationData.data_quality} Quality
                    </div>
                  </div>
                </div>

                {}
                <div className="space-y-3">
                  {Object.entries(locationData.probabilities).filter(([key, value]) => 
                    key !== 'summary' && typeof value === 'object' && value !== null && 'probability' in value
                  ).map(([key, condition]: [string, any]) => (
                    <div key={key} className="bg-white/5 rounded-lg p-3">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-white text-sm font-medium capitalize">
                          {condition.label || key.replace('_', ' ')}
                        </span>
                        <span className="text-white font-semibold">
                          {condition.probability !== null ? `${condition.probability.toFixed(2)}%` : 'N/A'}
                        </span>
                      </div>
                      {condition.probability !== null && (
                        <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-cyan-500 h-3 rounded-full transition-all duration-1000 ease-out shadow-lg shadow-blue-500/25"
                            style={{ 
                              width: `${Math.min(100, Math.max(0, condition.probability))}%`,
                              animation: `progressFill 1.5s ease-out ${index * 0.1}s both`
                            }}
                          ></div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {}
        <div className="mt-8 p-6 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 shadow-lg">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-400 rounded-lg flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h3 className="text-white font-semibold text-lg">Analysis Information</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-white/80">
            <div>
              <span className="font-medium">Analysis Period:</span> {analysis_period}
            </div>
            <div>
              <span className="font-medium">Data Sources:</span> {data_sources.join(', ')}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComparisonResults;
