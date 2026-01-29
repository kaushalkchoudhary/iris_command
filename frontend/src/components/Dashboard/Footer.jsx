import React, { useState, useEffect } from 'react';
import { Video, MonitorPlay, Check, Activity, Globe, Wifi } from 'lucide-react';
import clsx from 'clsx';

const Footer = ({ selectedVideos, onVideosChange, videos }) => {
    const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());

    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
        return () => clearInterval(timer);
    }, []);

    const availableVideos = Array.isArray(videos) ? videos : [];

    const toggleVideo = (video) => {
        const isCurrentlySelected = selectedVideos.some(v => v.id === video.id);
        if (isCurrentlySelected) {
            if (selectedVideos.length === 1) return;
            onVideosChange(selectedVideos.filter(v => v.id !== video.id));
        } else {
            onVideosChange([...selectedVideos, video]);
        }
    };

    const isSelected = (videoId) => selectedVideos.some(v => v.id === videoId);

    return (
        <div className="w-full h-12 bg-black/60 backdrop-blur-xl border-t border-cyan-500/30 flex items-center justify-between px-6 z-50">
            {/* System Status - Left */}
            <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]" />
                    <span className="text-[10px] text-emerald-500 font-bold uppercase tracking-[0.2em]">Network: Online</span>
                </div>
                <div className="flex items-center gap-2 border-l border-white/10 pl-6">
                    <Globe className="w-3 h-3 text-cyan-500/60" />
                    <span className="text-[10px] text-cyan-500/60 font-medium uppercase tracking-widest">Nodes: {selectedVideos.length}/04</span>
                </div>
                <div className="flex items-center gap-2 border-l border-white/10 pl-6">
                    <Activity className="w-3 h-3 text-cyan-500/60" />
                    <span className="text-[10px] text-cyan-500/60 font-medium uppercase tracking-widest">Feed: Active</span>
                </div>
            </div>

            {/* Camera Selector - Center */}
            <div className="flex items-center gap-1 bg-white/5 p-1 rounded-md border border-white/10">
                {availableVideos.map((vid, idx) => (
                    <button
                        key={vid.id}
                        onClick={() => toggleVideo(vid)}
                        className={clsx(
                            "flex items-center gap-2 px-3 py-1 text-[10px] font-mono font-bold uppercase tracking-wider transition-all duration-300 rounded",
                            isSelected(vid.id)
                                ? "bg-cyan-500 text-black shadow-[0_0_10px_rgba(6,182,212,0.5)]"
                                : "text-white/40 hover:text-white/60 hover:bg-white/5"
                        )}
                    >
                        {vid.label}
                    </button>
                ))}
            </div>

            {/* Diagnostics - Right */}
            <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-white/30 font-medium uppercase tracking-widest">Uptime: 01:24:45</span>
                </div>
                <div className="flex items-center gap-2 border-l border-white/10 pl-6">
                    <span className="text-[10px] text-cyan-500/80 font-mono">{currentTime}</span>
                </div>
                <div className="flex items-center gap-2 border-l border-white/10 pl-6">
                    <Wifi className="w-3 h-3 text-white/20" />
                    <span className="text-[10px] text-white/20 font-mono">5.2 GBPS</span>
                </div>
            </div>
        </div>
    );
};

export default Footer;
