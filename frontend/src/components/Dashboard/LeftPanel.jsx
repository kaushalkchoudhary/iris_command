import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const LeftPanel = () => {
    const [congestion, setCongestion] = useState(50);

    useEffect(() => {
        const interval = setInterval(() => {
            setCongestion(c => Math.min(100, Math.max(0, c + (Math.random() * 15 - 7.5))));
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    const speed = Math.round(100 - congestion);
    let status = 'SMOOTH';
    let color = 'text-emerald-500';
    let borderColor = 'border-emerald-500/50';

    if (congestion > 75) { status = 'HEAVY'; color = 'text-red-500'; borderColor = 'border-red-500/50'; }
    else if (congestion > 50) { status = 'SLOW'; color = 'text-yellow-500'; borderColor = 'border-yellow-500/50'; }
    else if (congestion > 25) { status = 'MODERATE'; color = 'text-cyan-400'; borderColor = 'border-cyan-400/50'; }

    return (
        <motion.div
            initial={{ x: -50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="absolute left-6 top-1/2 -translate-y-1/2 flex flex-col gap-12 pointer-events-none z-40"
        >
            {/* Global Metrics in New Minimalist Style */}
            <div className="flex flex-col gap-2">
                <div className="text-xs text-cyan-500/60 uppercase tracking-[0.3em] font-black">System Velocity</div>
                <div className="flex items-baseline gap-2">
                    <span className="text-[120px] font-black text-white font-mono leading-none tracking-tighter shadow-cyan-500/20 drop-shadow-2xl">
                        {speed}
                    </span>
                    <span className="text-xl text-cyan-500/40 font-black">KM/H</span>
                </div>
            </div>

            <div className="flex flex-col gap-2">
                <div className="text-xs text-cyan-500/60 uppercase tracking-[0.3em] font-black">Average Density</div>
                <div className="flex items-baseline gap-2">
                    <span className={`text-[80px] font-black font-mono leading-none ${color} drop-shadow-lg`}>
                        {Math.round(congestion)}%
                    </span>
                </div>
            </div>

            <div className={`px-4 py-2 border-l-[6px] ${borderColor} bg-black/40 backdrop-blur-md self-start transform skew-x-[-12deg]`}>
                <div className={`text-sm font-black uppercase tracking-[0.2em] ${color} skew-x-[12deg]`}>
                    Network: {status}
                </div>
            </div>

            {/* Target ID / Metadata */}
            <div className="mt-8 font-mono text-[10px] text-cyan-500/40 flex flex-col gap-1">
                <div className="flex gap-4">
                    <span className="text-cyan-500/60">NODE:</span>
                    <span>IRIS-BLR-MAIN</span>
                </div>
                <div className="flex gap-4">
                    <span className="text-cyan-500/60">LAT:</span>
                    <span>12.9716° N</span>
                </div>
                <div className="flex gap-4">
                    <span className="text-cyan-500/60">LNG:</span>
                    <span>77.5946° E</span>
                </div>
            </div>
        </motion.div>
    );
};

export default LeftPanel;
