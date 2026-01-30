import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Shield, Cpu, Activity, Wifi } from 'lucide-react';

const IRISLoader = () => {
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setProgress(prev => {
                const next = prev + (Math.random() * 15);
                return next >= 100 ? 100 : next;
            });
        }, 300);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="absolute inset-0 z-[100] bg-[#050a14] flex flex-col items-center justify-center overflow-hidden">
            {/* Background Cyber-Grid */}
            <div className="absolute inset-0 opacity-20 pointer-events-none"
                style={{
                    backgroundImage: 'linear-gradient(rgba(0, 255, 255, 0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 255, 255, 0.2) 1px, transparent 1px)',
                    backgroundSize: '20px 20px'
                }}>
            </div>

            {/* Compact Scanning Ring */}
            <div className="relative flex items-center justify-center">
                <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    className="w-20 h-20 border-2 border-dashed border-cyan-500/20 rounded-full"
                />

                <motion.div
                    animate={{ rotate: -360 }}
                    transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                    className="absolute w-28 h-28 border-t border-b border-cyan-500/30 rounded-full"
                />

                <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="absolute flex flex-col items-center"
                >
                    <Shield className="w-8 h-8 text-cyan-500 drop-shadow-[0_0_10px_rgba(6,182,212,0.5)]" />
                </motion.div>
            </div>

            {/* Compact Progress Section */}
            <div className="mt-4 w-40">
                <div className="flex justify-between items-center mb-1">
                    <span className="text-[8px] text-cyan-500/60 uppercase font-black tracking-wider">Initializing</span>
                    <span className="font-mono text-sm font-bold text-cyan-400">{Math.round(progress)}%</span>
                </div>

                {/* Progress Bar */}
                <div className="h-0.5 w-full bg-white/5 rounded-full overflow-hidden border border-white/10">
                    <motion.div
                        className="h-full bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.8)]"
                        initial={{ width: 0 }}
                        animate={{ width: `${progress}%` }}
                    />
                </div>

                {/* Mini Diagnostics */}
                <div className="flex justify-between mt-2 opacity-50">
                    <div className="flex items-center gap-1">
                        <Cpu className="w-2 h-2 text-cyan-500" />
                        <span className="text-[7px] text-white font-bold">SYS</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <Activity className="w-2 h-2 text-cyan-500" />
                        <span className="text-[7px] text-white font-bold">SYNC</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <Wifi className="w-2 h-2 text-cyan-500" />
                        <span className="text-[7px] text-white font-bold">LINK</span>
                    </div>
                </div>
            </div>

            {/* Small Corner Brackets */}
            <div className="absolute top-2 left-2 w-6 h-6 border-t border-l border-cyan-500/30" />
            <div className="absolute top-2 right-2 w-6 h-6 border-t border-r border-cyan-500/30" />
            <div className="absolute bottom-2 left-2 w-6 h-6 border-b border-l border-cyan-500/30" />
            <div className="absolute bottom-2 right-2 w-6 h-6 border-b border-r border-cyan-500/30" />
        </div>
    );
};

export default IRISLoader;
