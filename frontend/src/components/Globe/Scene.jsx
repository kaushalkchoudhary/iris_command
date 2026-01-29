import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import Earth from './Earth';
import Markers from './Markers';

const Scene = () => {
    return (
        <Canvas camera={{ position: [0, 0, 3.5], fov: 45 }}>
            <color attach="background" args={['#000000']} />

            <Suspense fallback={null}>
                <ambientLight intensity={1.5} color="#4444ff" />
                <pointLight position={[10, 10, 5]} intensity={2} color="#ffffff" />
                <pointLight position={[-10, -10, -5]} intensity={1} color="#0044ff" />

                <Stars radius={300} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

                <Earth />
                <Markers />

                <OrbitControls
                    autoRotate
                    autoRotateSpeed={0.5}
                    enableZoom={true}
                    minDistance={2.5}
                    maxDistance={10}
                />
            </Suspense>
        </Canvas>
    );
};

export default Scene;
