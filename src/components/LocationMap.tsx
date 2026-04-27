import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

interface LocationMapProps {
  onLocationSelect: (lat: number, lng: number, address?: string) => void;
  initialLat?: number;
  initialLng?: number;
  height?: string;
}

interface MapEventsProps {
  onLocationSelect: (lat: number, lng: number) => void;
}

const MapEvents: React.FC<MapEventsProps> = ({ onLocationSelect }) => {
  useMapEvents({
    click: (e) => {
      const { lat, lng } = e.latlng;
      onLocationSelect(lat, lng);
    },
  });
  return null;
};

const LocationMap: React.FC<LocationMapProps> = ({ 
  onLocationSelect, 
  initialLat = 12.9716, 
  initialLng = 77.5946,
  height = "400px"
}) => {
  const [selectedLocation, setSelectedLocation] = useState<{ lat: number; lng: number } | null>(
    initialLat && initialLng ? { lat: initialLat, lng: initialLng } : null
  );
  const [address, setAddress] = useState<string>('');

  const getAddressFromCoords = async (lat: number, lng: number) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&addressdetails=1`
      );
      const data = await response.json();

      if (data.display_name) {

        const city = data.address?.city || data.address?.town || data.address?.village || '';
        const state = data.address?.state || '';
        const country = data.address?.country || '';

        let address = '';
        if (city) address += city;
        if (state) address += `, ${state}`;
        if (country) address += `, ${country}`;

        return address || data.display_name;
      }
      return `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
    } catch (error) {
      console.error('Error getting address:', error);
      return `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
    }
  };

  const handleLocationSelect = async (lat: number, lng: number) => {
    setSelectedLocation({ lat, lng });
    const address = await getAddressFromCoords(lat, lng);
    setAddress(address);
    onLocationSelect(lat, lng, address);
  };

  const RecenterOnSelect: React.FC = () => {
    const map = useMap();
    React.useEffect(() => {
      if (selectedLocation) {
        map.setView([selectedLocation.lat, selectedLocation.lng], Math.max(map.getZoom(), 8), {
          animate: true
        });
      }
    }, [map, selectedLocation]);
    return null;
  };

  return (
    <div className="w-full">
      <div className="mb-4">
        <h3 className="text-white font-medium mb-2">Select Location on Map</h3>
        <p className="text-white/70 text-sm mb-4">
          Click anywhere on the map to select a location, or search for a specific place.
        </p>

        {}
        <div className="mb-4">
          <input
            type="text"
            placeholder="Search for a city or place..."
            className="w-full px-4 py-2 rounded-lg bg-white/20 border border-white/30 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white/50"
            onChange={async (e) => {
              const query = e.target.value;
              if (query.length > 2) {
                try {
                  const response = await fetch(
                    `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1&addressdetails=1`
                  );
                  const data = await response.json();
                  if (data.length > 0) {
                    const { lat, lon, display_name } = data[0];
                    const coords = { lat: parseFloat(lat), lng: parseFloat(lon) };
                    setSelectedLocation(coords);
                    setAddress(display_name);
                    onLocationSelect(coords.lat, coords.lng, display_name);
                  }
                } catch (error) {
                  console.error('Error searching location:', error);
                }
              }
            }}
          />
        </div>

        {}
        {selectedLocation && (
          <div className="bg-white/10 rounded-lg p-3 mb-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white font-medium">
                  {address || `${selectedLocation.lat.toFixed(4)}, ${selectedLocation.lng.toFixed(4)}`}
                </p>
                <p className="text-white/70 text-sm">
                  {selectedLocation.lat.toFixed(4)}°N, {selectedLocation.lng.toFixed(4)}°E
                </p>
              </div>
              <button
                onClick={() => {
                  setSelectedLocation(null);
                  setAddress('');
                }}
                className="text-white/60 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>
          </div>
        )}
      </div>

      {}
      <div 
        className="rounded-lg overflow-hidden border border-white/20"
        style={{ height }}
      >
        <MapContainer
          center={[selectedLocation?.lat ?? initialLat, selectedLocation?.lng ?? initialLng]}
          zoom={selectedLocation ? 10 : 3}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {selectedLocation && (
            <Marker position={[selectedLocation.lat, selectedLocation.lng]}>
              <Popup>
                <div className="text-center">
                  <p className="font-medium">{address || 'Selected Location'}</p>
                  <p className="text-sm text-gray-600">
                    {selectedLocation.lat.toFixed(4)}°, {selectedLocation.lng.toFixed(4)}°
                  </p>
                </div>
              </Popup>
            </Marker>
          )}

          <MapEvents onLocationSelect={handleLocationSelect} />
          <RecenterOnSelect />
        </MapContainer>
      </div>

      {}
      <div className="mt-4 text-center">
        <p className="text-white/60 text-sm">
          💡 Click anywhere on the map to select a location
        </p>
      </div>
    </div>
  );
};

export default LocationMap;
