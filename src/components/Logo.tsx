import React, { useState, useEffect } from 'react';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  animated?: boolean;
  showFullText?: boolean;
  className?: string;
}

const Logo: React.FC<LogoProps> = ({ 
  size = 'md', 
  animated = false, 
  showFullText = true, 
  className = '' 
}) => {
  const [isExpanded, setIsExpanded] = useState(!animated);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (animated) {

      const timer1 = setTimeout(() => setIsVisible(true), 100);
      const timer2 = setTimeout(() => setIsExpanded(true), 800);

      return () => {
        clearTimeout(timer1);
        clearTimeout(timer2);
      };
    } else {
      setIsVisible(true);
      setIsExpanded(true);
    }
  }, [animated]);

  const sizeClasses = {
    sm: 'w-8 h-8 text-lg',
    md: 'w-12 h-12 text-xl',
    lg: 'w-16 h-16 text-2xl',
    xl: 'w-20 h-20 text-3xl'
  };

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-lg',
    lg: 'text-xl',
    xl: 'text-2xl'
  };

  return (
    <div className={`flex items-center space-x-3 ${className}`}>
      {}
      <div className={`${sizeClasses[size]} relative`}>
        <div className={`w-full h-full bg-gradient-to-br from-blue-500 via-cyan-400 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/25 transition-all duration-500 ${isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-75'}`}>
          {}
          <div className="relative w-6 h-6">
            {}
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-gradient-to-br from-yellow-300 to-orange-400 rounded-full animate-pulse"></div>
            {}
            <div className="absolute top-1 left-0 w-5 h-3 bg-white/90 rounded-full"></div>
            <div className="absolute top-1.5 left-1 w-3 h-2 bg-white/90 rounded-full"></div>
            <div className="absolute top-1.5 left-2.5 w-3 h-2 bg-white/90 rounded-full"></div>
            {}
            <div className="absolute top-3 left-1 w-0.5 h-1.5 bg-blue-400 animate-bounce" style={{ animationDelay: '0s' }}></div>
            <div className="absolute top-3 left-1.5 w-0.5 h-1.5 bg-blue-400 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            <div className="absolute top-3 left-2 w-0.5 h-1.5 bg-blue-400 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
          </div>
        </div>

        {}
        <div className={`absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl blur-xl transition-all duration-500 ${isVisible ? 'opacity-100' : 'opacity-0'}`}></div>
      </div>

      {}
      <div className={`transition-all duration-700 overflow-hidden ${isExpanded ? 'max-w-48' : 'max-w-0'}`}>
        <div className={`${textSizeClasses[size]} font-bold text-white bg-gradient-to-r from-white via-blue-200 to-purple-200 bg-clip-text text-transparent whitespace-nowrap`}>
          {isExpanded ? 'PastCast' : 'PC'}
        </div>
        {isExpanded && (
          <div className="text-xs text-white/70 font-medium mt-0.5">
            Weather Intelligence
          </div>
        )}
      </div>
    </div>
  );
};

export default Logo;
