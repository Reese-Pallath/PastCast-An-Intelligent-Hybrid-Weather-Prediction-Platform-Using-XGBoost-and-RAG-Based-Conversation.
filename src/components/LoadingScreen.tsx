import React, { useState, useEffect } from 'react';
import Logo from './Logo';

interface LoadingScreenProps {
  onComplete: () => void;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ onComplete }) => {
  const [progress, setProgress] = useState(0);
  const [showLogo, setShowLogo] = useState(false);
  const [showText, setShowText] = useState(false);

useEffect(() => {

  const timer1 = setTimeout(() => setShowLogo(true), 200);

  const timer2 = setTimeout(() => setShowText(true), 600);

  const timer3 = setTimeout(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(onComplete, 300); 
          return 100;
        }
        return prev + 5; 
      });
    }, 40);
  }, 600);

  return () => {
    clearTimeout(timer1);
    clearTimeout(timer2);
    clearTimeout(timer3);
  };
}, [onComplete]);

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center z-50">
      {}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-20 -right-20 w-64 h-64 bg-gradient-to-r from-blue-600/10 to-cyan-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-20 -left-20 w-64 h-64 bg-gradient-to-r from-indigo-600/10 to-purple-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>

        {}
        {[...Array(8)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-white/20 rounded-full animate-ping"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${2 + Math.random() * 3}s`
            }}
          ></div>
        ))}
      </div>

      {}
      <div className="relative z-10 text-center">
        {}
        <div className={`transition-all duration-1000 ${showLogo ? 'opacity-100 scale-100' : 'opacity-0 scale-75'}`}>
          <Logo size="xl" animated={true} />
        </div>

        {}
        <div className={`mt-8 transition-all duration-700 delay-500 ${showText ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
          <h2 className="text-2xl font-bold text-white mb-2">Welcome to PastCast</h2>
          <p className="text-white/70 text-sm mb-6">Historical Weather Intelligence Platform</p>
        </div>

        {}
        <div className={`w-64 mx-auto transition-all duration-700 delay-700 ${showText ? 'opacity-100' : 'opacity-0'}`}>
          <div className="w-full bg-white/10 rounded-full h-2 mb-2">
            <div 
              className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <div className="text-white/60 text-xs font-medium">
            {progress < 100 ? `Loading... ${progress}%` : 'Ready!'}
          </div>
        </div>

        {}
        <div className={`flex justify-center space-x-2 mt-4 transition-all duration-700 delay-900 ${showText ? 'opacity-100' : 'opacity-0'}`}>
          {[...Array(3)].map((_, i) => (
            <div
              key={i}
              className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"
              style={{ animationDelay: `${i * 0.2}s` }}
            ></div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;
