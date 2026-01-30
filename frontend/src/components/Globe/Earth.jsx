import React, { useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, useVideoTexture } from '@react-three/drei';
import * as THREE from 'three';

const Earth = () => {
    const earthRef = useRef();

    // Load video texture
    const texture = useVideoTexture('/output_video.mp4');

    useEffect(() => {
        // Ensure video plays and loops
        if (texture.image) {
            texture.image.loop = true;
            texture.image.muted = true;
            texture.image.play();
        }
    }, [texture]);

    useFrame(() => {
        if (earthRef.current) {
            // Optional: Rotate the sphere if the video itself isn't a rotating animation
            // earthRef.current.rotation.y += 0.0005;
        }
    });

    return (
        <group>
            {/* Main Earth Sphere with Video Texture */}
            <Sphere ref={earthRef} args={[1, 64, 64]}>
                <meshBasicMaterial
                    map={texture}
                    toneMapped={false}
                />
            </Sphere>

            {/* Atmosphere Glow (Kept for aesthetics) */}
            <Sphere args={[1.02, 64, 64]}>
                <meshBasicMaterial
                    color="#4fa1d8"
                    transparent
                    opacity={0.1}
                    blending={THREE.AdditiveBlending}
                    side={THREE.BackSide}
                />
            </Sphere>
        </group>
    );
};

export default Earth;
