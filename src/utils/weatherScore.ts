export interface WeatherScoreInput {
  cloudCover: number;  // 0–100 %
  rainChance: number;  // 0–100 %
  temperature: number; // °C
  windSpeed: number;   // km/h
}

export type WeatherLabel = 'Excellent' | 'Good' | 'Fair' | 'Poor';

export interface ScoreColors {
  bar: string;   // Tailwind gradient for the progress bar
  text: string;  // Tailwind gradient for the score number
  badge: string; // Tailwind classes for the label badge
}

/**
 * Compute a 0–100 pleasantness score from raw weather inputs.
 *
 * Weights
 *   Rain chance  30 % — biggest single-day mood killer
 *   Cloud cover  25 % — affects UV, mood, solar warmth
 *   Temperature  25 % — comfort window 20–30 °C
 *   Wind speed   20 % — calm wind is generally preferred
 */
export function calculateWeatherScore({
  cloudCover,
  rainChance,
  temperature,
  windSpeed,
}: WeatherScoreInput): number {
  // Negative factors: higher value = worse → invert
  const cloudScore = (100 - cloudCover) * 0.25;
  const rainScore  = (100 - rainChance) * 0.30;

  // Temperature comfort window
  let tempScore: number;
  if (temperature >= 20 && temperature <= 30) {
    tempScore = 25; // ideal
  } else if (temperature >= 15 && temperature <= 35) {
    tempScore = 15; // acceptable
  } else {
    tempScore = 5;  // too cold or too hot
  }

  // Wind calmness
  let windScore: number;
  if (windSpeed < 10) {
    windScore = 20; // calm
  } else if (windSpeed < 25) {
    windScore = 10; // moderate
  } else {
    windScore = 5;  // strong
  }

  return Math.round(Math.min(100, Math.max(0, cloudScore + rainScore + tempScore + windScore)));
}

export function getWeatherLabel(score: number): WeatherLabel {
  if (score >= 75) return 'Excellent';
  if (score >= 55) return 'Good';
  if (score >= 35) return 'Fair';
  return 'Poor';
}

export function getScoreColors(label: WeatherLabel): ScoreColors {
  switch (label) {
    case 'Excellent':
      return {
        bar:   'from-green-500 to-emerald-400',
        text:  'from-green-400 to-emerald-300',
        badge: 'bg-green-500/20 text-green-300 border border-green-400/30',
      };
    case 'Good':
      return {
        bar:   'from-lime-500 to-green-400',
        text:  'from-lime-400 to-green-300',
        badge: 'bg-lime-500/20 text-lime-300 border border-lime-400/30',
      };
    case 'Fair':
      return {
        bar:   'from-yellow-500 to-amber-400',
        text:  'from-yellow-400 to-amber-300',
        badge: 'bg-yellow-500/20 text-yellow-300 border border-yellow-400/30',
      };
    default: // Poor
      return {
        bar:   'from-red-500 to-rose-400',
        text:  'from-red-400 to-rose-300',
        badge: 'bg-red-500/20 text-red-300 border border-red-400/30',
      };
  }
}

/**
 * Parse the actual maximum temperature (°C) embedded in the backend
 * explanation string, e.g. "Expected max ~12 C — Cold. Dress in warm layers."
 * Falls back to 25 °C (comfortable) if the string doesn't match.
 */
export function parseTempFromThreshold(threshold: string | undefined): number {
  const match = threshold?.match(/~?(\d+(?:\.\d+)?)\s*[Cc]/);
  return match ? parseFloat(match[1]) : 25;
}

/**
 * Parse the actual wind speed (km/h) embedded in the backend explanation
 * string, e.g. "2.5 km/h — Near calm. Perfect for all outdoor activities."
 * Falls back to 10 km/h (light breeze) if the string doesn't match.
 */
export function parseWindFromThreshold(threshold: string | undefined): number {
  const match = threshold?.match(/^(\d+(?:\.\d+)?)\s*km\/h/);
  return match ? parseFloat(match[1]) : 10;
}
