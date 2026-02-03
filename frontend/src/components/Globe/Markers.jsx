import React, { useMemo } from 'react';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

const locations = [
    { name: 'Beijing', lat: 39.9042, lon: 116.4074, value: 4385 },
    { name: 'London', lat: 51.5074, lon: -0.1278, value: 1500 },
    { name: 'New York', lat: 40.7128, lon: -74.0060, value: 2100 },
    { name: 'Tokyo', lat: 35.6762, lon: 139.6503, value: 3200 },
    { name: 'Dubai', lat: 25.2048, lon: 55.2708, value: 900 },
    { name: 'Singapore', lat: 1.3521, lon: 103.8198, value: 1200 },
];

function latLongToVector3(lat, lon, radius) {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    const x = -(radius * Math.sin(phi) * Math.cos(theta));
    const z = (radius * Math.sin(phi) * Math.sin(theta));
    const y = (radius * Math.cos(phi));
    return new THREE.Vector3(x, y, z);
}

const Markers = () => {
    const markerData = useMemo(() => {
        return locations.map(loc => ({
            ...loc,
            position: latLongToVector3(loc.lat, loc.lon, 1.05)
        }));
    }, []);

    return (
        <group>
            {markerData.map((marker, i) => (
                <group key={i} position={marker.position}>
                    <mesh>
                        <sphereGeometry args={[0.015, 16, 16]} />
                        <meshBasicMaterial color="#00ffcc" />
                    </mesh>
                    <Html distanceFactor={1.5} occlude>
                        <div className="pointer-events-none select-none">
                            <div className="flex flex-col items-center">
                                <div className="px-2 py-1 bg-black/60 border border-cyan-500/50 backdrop-blur-sm rounded text-[8px] text-cyan-300 font-mono whitespace-nowrap">
                                    {marker.name}
                                </div>
                                <div className="h-4 w-[1px] bg-cyan-500/50"></div>
                            </div>
                        </div>
                    </Html>
                </group>
            ))}
        </group>
    );
};

export default Markers;
