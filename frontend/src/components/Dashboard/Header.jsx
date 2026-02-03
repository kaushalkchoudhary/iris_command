import React, { useState, useEffect, useRef } from 'react';
import { LogOut } from 'lucide-react';

/* Animated IRIS icon — radar sweep with pulsing core and orbiting arcs */
const IrisIcon = ({ className = '' }) => (
    <svg viewBox="0 0 24 24" fill="none" className={className}>
        {/* Outer ring — slow rotate */}
        <g>
            <animateTransform attributeName="transform" type="rotate"
                from="0 12 12" to="360 12 12" dur="10s" repeatCount="indefinite" />
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="0.8" opacity="0.25" />
            <path d="M12 2 A10 10 0 0 1 22 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.7" />
        </g>
        {/* Mid ring — counter-rotate */}
        <g>
            <animateTransform attributeName="transform" type="rotate"
                from="360 12 12" to="0 12 12" dur="6s" repeatCount="indefinite" />
            <circle cx="12" cy="12" r="6.5" stroke="currentColor" strokeWidth="0.6" opacity="0.2" />
            <path d="M12 5.5 A6.5 6.5 0 0 1 18.5 12" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" opacity="0.5" />
            <path d="M12 18.5 A6.5 6.5 0 0 1 5.5 12" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" opacity="0.5" />
        </g>
        {/* Radar sweep line */}
        <g>
            <animateTransform attributeName="transform" type="rotate"
                from="0 12 12" to="360 12 12" dur="3s" repeatCount="indefinite" />
            <line x1="12" y1="12" x2="12" y2="3" stroke="currentColor" strokeWidth="0.8" strokeLinecap="round" opacity="0.4" />
        </g>
        {/* Core — pulses */}
        <circle cx="12" cy="12" r="2" fill="currentColor">
            <animate attributeName="r" values="1.5;2.5;1.5" dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="1;0.6;1" dur="2s" repeatCount="indefinite" />
        </circle>
    </svg>
);

const Header = ({ onReset, onLogout }) => {
    const [now, setNow] = useState(new Date());
    const [uptime, setUptime] = useState(0);
    const [showFull, setShowFull] = useState(false);
    const username = localStorage.getItem('iris_username') || 'OPERATOR';
    const startRef = useRef(Date.now());

    useEffect(() => {
        const timer = setInterval(() => {
            setNow(new Date());
            setUptime(Math.floor((Date.now() - startRef.current) / 1000));
        }, 1000);
        return () => clearInterval(timer);
    }, []);

    // Toggle full form every 2 seconds
    useEffect(() => {
        const toggle = setInterval(() => setShowFull(p => !p), 2000);
        return () => clearInterval(toggle);
    }, []);

    const pad = (n) => String(n).padStart(2, '0');
    const timeStr = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    const dateStr = now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase();

    const uptimeH = Math.floor(uptime / 3600);
    const uptimeM = Math.floor((uptime % 3600) / 60);
    const uptimeS = uptime % 60;
    const uptimeStr = `${pad(uptimeH)}:${pad(uptimeM)}:${pad(uptimeS)}`;

    return (
        <div className="w-full h-14 bg-black/60 backdrop-blur-md border-b border-white/10 flex items-center justify-between px-5 font-mono select-none z-50 shrink-0">

            {/* LEFT — Branding */}
            <div className="flex items-center gap-2.5 min-w-0">
                <button
                    onClick={onReset}
                    className="flex items-center gap-2 hover:opacity-80 transition-opacity"
                    title="Back to mission select"
                >
                    <IrisIcon className="w-5 h-5 text-cyan-400" />
                    <span className="text-sm font-bold text-cyan-400 tracking-[0.15em] transition-opacity duration-500"
                          style={{ opacity: showFull ? 0 : 1, position: showFull ? 'absolute' : 'relative', pointerEvents: showFull ? 'none' : 'auto' }}>
                        IRIS
                    </span>
                    <span className="text-[11px] font-bold text-cyan-400/80 tracking-[0.12em] transition-opacity duration-500 whitespace-nowrap"
                          style={{ opacity: showFull ? 1 : 0, position: showFull ? 'relative' : 'absolute', pointerEvents: showFull ? 'auto' : 'none' }}>
                        INTEGRATED REALTIME INTELLIGENCE SYSTEM
                    </span>
                </button>
            </div>

            {/* CENTER — Telemetry */}
            <div className="flex items-center gap-3 text-xs">
                <span className="text-white/80 tabular-nums">{timeStr}</span>
                <span className="text-white/15">|</span>
                <span className="text-white/80 tabular-nums">{dateStr}</span>
                <span className="text-white/15">|</span>
                <span className="text-white/40 tracking-wider">CMD</span>
                <span className="text-white/80 tabular-nums">12.9716°N 77.5946°E</span>
                <span className="text-white/15">|</span>
                <span className="text-white/40 tracking-wider">UP</span>
                <span className="text-white/80 tabular-nums">{uptimeStr}</span>
            </div>

            {/* RIGHT — Username + Logout */}
            <div className="flex items-center gap-3">
                <span className="px-2 py-0.5 bg-amber-500/10 border border-amber-500/30 text-[10px] text-amber-400/80 tracking-widest font-bold">
                    RESTRICTED
                </span>
                <span className="text-white/15 text-xs">|</span>
                <span className="text-xs text-white/80 font-bold tracking-wider uppercase">{username}</span>
                <button
                    onClick={onLogout}
                    className="p-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 hover:border-red-500/40 transition-all text-red-400/80 hover:text-red-400"
                    title="Logout"
                >
                    <LogOut className="w-3.5 h-3.5" />
                </button>
            </div>
        </div>
    );
};

export default Header;
