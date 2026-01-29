import React, { useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, useVideoTexture } from '@react-three/drei';
import * as THREE from 'three';

const Earth = () => {
  const earthRef = useRef();
  const atmosphereRef = useRef();

  /* ============================
     VIDEO TEXTURE (LIVE LAYER)
  ============================ */
  const texture = useVideoTexture('/output_video.mp4', {
    loop: true,
    muted: true,
    start: true,
    crossOrigin: 'Anonymous',
  });

  useEffect(() => {
    if (texture?.image) {
      texture.image.playbackRate = 1;
    }
  }, [texture]);

  /* ============================
     MOTION — SLOW, TACTICAL
  ============================ */
  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();

    if (earthRef.current) {
      // very slow axial drift (feels orbital, not spinning)
      earthRef.current.rotation.y = t * 0.02;
    }

    if (atmosphereRef.current) {
      // subtle pulse to imply sensor sweep
      atmosphereRef.current.material.opacity =
        0.08 + Math.sin(t * 1.2) * 0.015;
    }
  });

  return (
    <group>
      {/* ============================
         EARTH — LIVE FEED SURFACE
      ============================ */}
      <Sphere ref={earthRef} args={[1, 96, 96]}>
        <meshBasicMaterial
          map={texture}
          toneMapped={false}
          color={new THREE.Color(0x9fffe0)} // slight emerald bias
        />
      </Sphere>

      {/* ============================
         ATMOSPHERE — SENSOR HALO
      ============================ */}
      <Sphere ref={atmosphereRef} args={[1.035, 96, 96]}>
        <meshBasicMaterial
          transparent
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
          color={new THREE.Color(0x10b981)} // emerald-500
          opacity={0.08}
        />
      </Sphere>

      {/* ============================
         OUTER SCAN SHELL (VERY FAINT)
      ============================ */}
      <Sphere args={[1.06, 64, 64]}>
        <meshBasicMaterial
          transparent
          side={THREE.BackSide}
          color={new THREE.Color(0x10b981)}
          opacity={0.025}
        />
      </Sphere>
    </group>
  );
};

export default Earth;
