import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
                    backgroundSize: '30px 30px'
                }}>
            </div>

            {/* Scanning Ring */}
            <div className="relative flex items-center justify-center">
                <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    className="w-48 h-48 border-4 border-dashed border-cyan-500/20 rounded-full"
                />

                <motion.div
                    animate={{ rotate: -360 }}
                    transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                    className="absolute w-64 h-64 border-t-2 border-b-2 border-cyan-500/30 rounded-full shadow-[0_0_50px_rgba(6,182,212,0.1)]"
                />

                <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="absolute flex flex-col items-center"
                >
                    <Shield className="w-16 h-16 text-cyan-500 mb-4 drop-shadow-[0_0_15px_rgba(6,182,212,0.5)]" />
                    <div className="text-4xl font-black text-white tracking-[0.4em] ml-2">IRIS</div>
                </motion.div>
            </div>

            {/* Intelligence Diagnostic Text */}
            <div className="mt-12 w-80">
                <div className="flex justify-between items-end mb-2">
                    <div className="flex flex-col">
                        <span className="text-[10px] text-cyan-500/60 uppercase font-black tracking-widest leading-none mb-1">System Initialization</span>
                        <span className="font-mono text-[10px] text-white/40 uppercase">Encrypted_Link_Layer: AKTIV</span>
                    </div>
                    <span className="font-mono text-xl font-bold text-cyan-400">{Math.round(progress)}%</span>
                </div>

                {/* Progress Bar Container */}
                <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden border border-white/10 relative">
                    <motion.div
                        className="h-full bg-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.8)]"
                        initial={{ width: 0 }}
                        animate={{ width: `${progress}%` }}
                    />
                </div>

                {/* Sub-Diagnostics */}
                <div className="grid grid-cols-3 gap-2 mt-4 opacity-50">
                    <div className="flex items-center gap-2">
                        <Cpu className="w-3 h-3 text-cyan-500" />
                        <span className="text-[8px] text-white font-bold uppercase tracking-tighter">Core:04</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Activity className="w-3 h-3 text-cyan-500" />
                        <span className="text-[8px] text-white font-bold uppercase tracking-tighter">Syncing</span>
                    </div>
                    <div className="flex items-center gap-2 text-right">
                        <Wifi className="w-3 h-3 text-cyan-500 ml-auto" />
                        <span className="text-[8px] text-white font-bold uppercase tracking-tighter">Link+</span>
                    </div>
                </div>
            </div>

            {/* Corner Bracket Frame */}
            <div className="absolute top-10 left-10 w-24 h-24 border-t-2 border-l-2 border-cyan-500/30" />
            <div className="absolute top-10 right-10 w-24 h-24 border-t-2 border-r-2 border-cyan-500/30" />
            <div className="absolute bottom-10 left-10 w-24 h-24 border-b-2 border-l-2 border-cyan-500/30" />
            <div className="absolute bottom-10 right-10 w-24 h-24 border-b-2 border-r-2 border-cyan-500/30" />

            <motion.div
                animate={{ opacity: [0.1, 0.4, 0.1] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="absolute bottom-10 text-[10px] text-cyan-500 tracking-[0.5em] font-black uppercase"
            >
                Authenticating Surveillance Nodes...
            </motion.div>
        </div>
    );
};

export default IRISLoader;
