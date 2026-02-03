import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const IRISLoader = ({ connected = false, ready = false }) => {
    const [progress, setProgress] = useState(0);
    const [scale, setScale] = useState(1);
    const [phase, setPhase] = useState('init'); // init → connecting → streaming → done
    const containerRef = useRef(null);

    // Derive phase from props
    useEffect(() => {
        if (ready) setPhase('done');
        else if (connected) setPhase('streaming');
        else if (progress > 5) setPhase('connecting');
        else setPhase('init');
    }, [connected, ready, progress]);

    // Progress tied to real state
    useEffect(() => {
        const interval = setInterval(() => {
            setProgress(prev => {
                if (ready) return 100;
                // Cap at 60 until connected, then cap at 90 until ready
                const cap = connected ? 92 : 55;
                if (prev >= cap) return cap;
                const speed = connected ? 8 + Math.random() * 12 : 2 + Math.random() * 5;
                return Math.min(prev + speed, cap);
            });
        }, 200);
        return () => clearInterval(interval);
    }, [connected, ready]);

    // Snap to 100 when ready
    useEffect(() => {
        if (ready) setProgress(100);
    }, [ready]);

    // Scale inner content to fit the cell
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const observer = new ResizeObserver(([entry]) => {
            const { width, height } = entry.contentRect;
            const fit = Math.min(width, height) / 220;
            setScale(Math.max(0.4, Math.min(fit, 1.5)));
        });
        observer.observe(el);
        return () => observer.disconnect();
    }, []);

    const phaseLabel = {
        init: 'INITIALIZING',
        connecting: 'ACQUIRING FEED',
        streaming: 'BUFFERING STREAM',
        done: 'FEED LOCKED',
    }[phase];

    const isDone = phase === 'done';

    return (
        <motion.div
            ref={containerRef}
            key="iris-loader"
            initial={{ opacity: 1 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="absolute inset-0 z-[100] bg-[#040810] flex flex-col items-center justify-center overflow-hidden"
        >
            {/* Animated scan lines */}
            <div className="absolute inset-0 pointer-events-none opacity-[0.03]"
                style={{
                    backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(6,182,212,0.4) 2px, rgba(6,182,212,0.4) 4px)',
                }} />

            {/* Sweeping scan bar */}
            <motion.div
                className="absolute left-0 right-0 h-[1px] pointer-events-none"
                style={{
                    background: 'linear-gradient(90deg, transparent 0%, rgba(6,182,212,0.3) 20%, rgba(6,182,212,0.6) 50%, rgba(6,182,212,0.3) 80%, transparent 100%)',
                    boxShadow: '0 0 20px 2px rgba(6,182,212,0.15)',
                }}
                animate={{ top: ['0%', '100%', '0%'] }}
                transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
            />

            {/* Scalable inner content */}
            <div
                className="flex flex-col items-center justify-center"
                style={{ transform: `scale(${scale})`, transformOrigin: 'center center' }}
            >
                {/* Radar assembly */}
                <div className="relative flex items-center justify-center w-32 h-32">
                    {/* Outermost ring — faint */}
                    <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 12, repeat: Infinity, ease: 'linear' }}
                        className="absolute w-32 h-32 rounded-full"
                        style={{ border: '1px solid rgba(6,182,212,0.1)' }}
                    />

                    {/* Outer arc */}
                    <svg className="absolute w-32 h-32" viewBox="0 0 128 128">
                        <motion.circle
                            cx="64" cy="64" r="58"
                            fill="none"
                            stroke="rgba(6,182,212,0.35)"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeDasharray="90 275"
                            animate={{ strokeDashoffset: [0, -365] }}
                            transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
                        />
                    </svg>

                    {/* Mid ring — counter rotate */}
                    <svg className="absolute w-24 h-24" viewBox="0 0 96 96">
                        <motion.circle
                            cx="48" cy="48" r="42"
                            fill="none"
                            stroke="rgba(6,182,212,0.2)"
                            strokeWidth="1"
                            strokeLinecap="round"
                            strokeDasharray="40 225"
                            animate={{ strokeDashoffset: [0, 265] }}
                            transition={{ duration: 5, repeat: Infinity, ease: 'linear' }}
                        />
                    </svg>

                    {/* Inner ring — dashed */}
                    <motion.div
                        animate={{ rotate: -360 }}
                        transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
                        className="absolute w-16 h-16 rounded-full"
                        style={{ border: '1px dashed rgba(6,182,212,0.15)' }}
                    />

                    {/* Radar sweep */}
                    <motion.div
                        className="absolute w-16 h-16"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 2.5, repeat: Infinity, ease: 'linear' }}
                    >
                        <div className="absolute top-1/2 left-1/2 w-1/2 h-[1px] origin-left"
                            style={{
                                background: 'linear-gradient(90deg, rgba(6,182,212,0.8), transparent)',
                            }}
                        />
                        <div className="absolute top-1/2 left-1/2 origin-left"
                            style={{
                                width: '50%',
                                height: '50%',
                                marginTop: '-25%',
                                background: 'conic-gradient(from 0deg, rgba(6,182,212,0.15), transparent 60deg)',
                                borderRadius: '0 100% 0 0',
                            }}
                        />
                    </motion.div>

                    {/* Center core */}
                    <motion.div
                        className="absolute rounded-full"
                        animate={{
                            boxShadow: isDone
                                ? ['0 0 15px 4px rgba(6,182,212,0.6)', '0 0 25px 8px rgba(6,182,212,0.8)', '0 0 15px 4px rgba(6,182,212,0.6)']
                                : ['0 0 8px 2px rgba(6,182,212,0.3)', '0 0 15px 4px rgba(6,182,212,0.5)', '0 0 8px 2px rgba(6,182,212,0.3)'],
                        }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        style={{
                            width: 10, height: 10,
                            background: isDone ? '#22d3ee' : '#06b6d4',
                        }}
                    />
                </div>

                {/* Status + Progress */}
                <div className="mt-5 w-44">
                    {/* Phase label */}
                    <div className="flex justify-between items-center mb-1.5">
                        <div className="flex items-center gap-1.5">
                            <motion.div
                                className="w-1 h-1 rounded-full"
                                style={{ background: isDone ? '#22d3ee' : '#06b6d4' }}
                                animate={{ opacity: isDone ? 1 : [1, 0.3, 1] }}
                                transition={{ duration: 1, repeat: isDone ? 0 : Infinity }}
                            />
                            <span className="text-[8px] uppercase font-black tracking-[0.2em]"
                                style={{ color: isDone ? '#22d3ee' : 'rgba(6,182,212,0.5)' }}>
                                {phaseLabel}
                            </span>
                        </div>
                        <span className="font-mono text-xs font-bold"
                            style={{ color: isDone ? '#22d3ee' : 'rgba(6,182,212,0.7)' }}>
                            {Math.round(progress)}%
                        </span>
                    </div>

                    {/* Progress bar */}
                    <div className="h-[3px] w-full bg-white/[0.06] rounded-full overflow-hidden">
                        <motion.div
                            className="h-full rounded-full"
                            style={{
                                background: isDone
                                    ? 'linear-gradient(90deg, #06b6d4, #22d3ee)'
                                    : 'linear-gradient(90deg, rgba(6,182,212,0.6), #06b6d4)',
                                boxShadow: isDone
                                    ? '0 0 12px rgba(34,211,238,0.8)'
                                    : '0 0 8px rgba(6,182,212,0.5)',
                            }}
                            initial={{ width: 0 }}
                            animate={{ width: `${progress}%` }}
                            transition={{ duration: 0.3, ease: 'easeOut' }}
                        />
                    </div>

                    {/* Sub-systems */}
                    <div className="flex justify-between mt-2.5 gap-1">
                        {[
                            { label: 'RTSP', active: progress > 10 },
                            { label: 'WebRTC', active: connected },
                            { label: 'DECODE', active: ready },
                        ].map((sys) => (
                            <div key={sys.label} className="flex items-center gap-1">
                                <div className="w-1 h-1 rounded-full"
                                    style={{
                                        background: sys.active ? '#22d3ee' : 'rgba(255,255,255,0.15)',
                                        boxShadow: sys.active ? '0 0 4px rgba(34,211,238,0.6)' : 'none',
                                    }}
                                />
                                <span className="text-[7px] font-bold tracking-wider"
                                    style={{ color: sys.active ? 'rgba(34,211,238,0.8)' : 'rgba(255,255,255,0.2)' }}>
                                    {sys.label}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Corner accents */}
            <div className="absolute top-2 left-2 w-5 h-5 border-t border-l border-cyan-500/25 pointer-events-none" />
            <div className="absolute top-2 right-2 w-5 h-5 border-t border-r border-cyan-500/25 pointer-events-none" />
            <div className="absolute bottom-2 left-2 w-5 h-5 border-b border-l border-cyan-500/25 pointer-events-none" />
            <div className="absolute bottom-2 right-2 w-5 h-5 border-b border-r border-cyan-500/25 pointer-events-none" />
        </motion.div>
    );
};

export default IRISLoader;
