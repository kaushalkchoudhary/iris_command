import React, { useState, useEffect, useRef } from 'react';
import { LogOut } from 'lucide-react';

/* Animated crosshair/reticle icon — inner ring pulses, outer ticks rotate */
const ReticleIcon = ({ className = '' }) => (
    <svg viewBox="0 0 24 24" fill="none" className={className}>
        {/* Outer rotating ticks */}
        <g>
            <animateTransform attributeName="transform" type="rotate"
                from="0 12 12" to="360 12 12" dur="8s" repeatCount="indefinite" />
            <line x1="12" y1="2" x2="12" y2="5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="22" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="12" y1="22" x2="12" y2="19" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="2" y1="12" x2="5" y2="12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </g>
        {/* Mid ring — counter-rotates */}
        <g>
            <animateTransform attributeName="transform" type="rotate"
                from="360 12 12" to="0 12 12" dur="6s" repeatCount="indefinite" />
            <path d="M12 7 L14 9" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.6" />
            <path d="M17 12 L15 14" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.6" />
            <path d="M12 17 L10 15" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.6" />
            <path d="M7 12 L9 10" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.6" />
        </g>
        {/* Center dot — pulses */}
        <circle cx="12" cy="12" r="2" fill="currentColor">
            <animate attributeName="r" values="1.5;2.5;1.5" dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="1;0.5;1" dur="2s" repeatCount="indefinite" />
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
                    <ReticleIcon className="w-5 h-5 text-cyan-400" />
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
                <span className="text-white/40 tracking-wider">GPS</span>
                <span className="text-white/80 tabular-nums">12.97°N 77.59°E</span>
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
