import React, { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { Radio, Wifi, Clock, Drone, Gauge, Timer, Signal, Activity, Zap } from 'lucide-react';

const API_BASE_URL = import.meta.env.DEV ? '/api' : `http://${window.location.hostname}:9010`;

const DRONE_REGION_MAP = {
  bcpdrone1: 'MG Road Junction',
  bcpdrone2: 'Outer Ring Road',
  bcpdrone3: 'Whitefield Main Road',
  bcpdrone4: 'Silk Board Signal',
  bcpdrone5: 'Marathahalli Bridge',
  bcpdrone6: 'Electronic City Flyover',
  bcpdrone7: 'Hebbal Flyover',
  bcpdrone8: 'KR Puram Junction',
  bcpdrone9: 'Bellandur Lake Road',
  bcpdrone10: 'HSR Layout Sector 7',
  bcpdrone11: 'Yelahanka New Town',
  bcpdrone12: 'JP Nagar Phase 6',
};

const Footer = ({ selectedVideos, onVideosChange, videos }) => {
  const [time, setTime] = useState(() =>
    new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
  );
  const [date, setDate] = useState(() => 
    new Date().toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' }).toUpperCase()
  );

  // Real-time performance metrics
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(42);
  const [bandwidth, setBandwidth] = useState(0);
  const [signalStrength, setSignalStrength] = useState(0);

  useEffect(() => {
    const t = setInterval(() => {
      const now = new Date();
      setTime(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }));
      setDate(now.toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' }).toUpperCase());
    }, 1000);
    return () => clearInterval(t);
  }, []);

  // Fetch real FPS from backend metrics API
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        if (response.ok) {
          const data = await response.json();
          // Calculate average FPS across all selected videos
          const selectedIds = selectedVideos.map(v => v.id);
          let totalFps = 0;
          let count = 0;
          for (const [sourceName, metrics] of Object.entries(data)) {
            if (selectedIds.includes(sourceName) && metrics.fps !== undefined) {
              totalFps += metrics.fps;
              count++;
            }
          }
          if (count > 0) {
            setFps(totalFps / count);
          }
        }
      } catch (e) {
        console.error('Failed to fetch metrics for FPS:', e);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, [selectedVideos]);

  // Simulate other performance metrics (latency, bandwidth, signal)
  useEffect(() => {
    const interval = setInterval(() => {
      setLatency(prev => Math.max(28, Math.min(85, prev + (Math.random() - 0.5) * 8)));
      setBandwidth(prev => Math.max(12.5, Math.min(48.2, prev + (Math.random() - 0.5) * 2.5)));
      setSignalStrength(prev => Math.max(65, Math.min(98, prev + (Math.random() - 0.5) * 10)));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const availableVideos = Array.isArray(videos) ? videos : [];
  const total = availableVideos.length;
  const activeCount = useMemo(() => selectedVideos?.length || 0, [selectedVideos]);

  const toggleVideo = (video) => {
    const isSelected = selectedVideos.some(v => v.id === video.id);
    if (isSelected) {
      if (selectedVideos.length === 1) return;
      onVideosChange(selectedVideos.filter(v => v.id !== video.id));
    } else {
      onVideosChange([...selectedVideos, video]);
    }
  };

  const isSelected = (id) => selectedVideos.some(v => v.id === id);

  // Signal strength color
  const getSignalColor = (strength) => {
    if (strength >= 80) return 'text-emerald-400';
    if (strength >= 60) return 'text-yellow-400';
    return 'text-orange-400';
  };

  // FPS color based on performance
  const getFpsColor = (fps) => {
    if (fps === 0) return 'text-white/40';
    if (fps >= 25) return 'text-emerald-400';
    if (fps >= 15) return 'text-yellow-400';
    return 'text-orange-400';
  };

  // Latency color
  const getLatencyColor = (lat) => {
    if (lat <= 50) return 'text-emerald-400';
    if (lat <= 75) return 'text-yellow-400';
    return 'text-orange-400';
  };

  return (
    <footer className="w-full h-10 bg-black/90 backdrop-blur-xl border-t border-emerald-500/20 flex items-center px-4 font-mono text-xs z-50 shadow-[0_-2px_20px_rgba(16,185,129,0.1)]">

      {/* LEFT: SYSTEM ID + STATUS */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="relative">
            <Radio className="w-4 h-4 text-emerald-400" strokeWidth={2} />
            <div className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-[10px] text-white/40 tracking-wider">SYSTEM</span>
            <span className="font-bold tracking-widest text-emerald-400">
              IRIS
            </span>
          </div>
        </div>

        <div className="h-6 w-px bg-white/10" />

        <div className="flex items-center gap-1.5">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400/40 animate-ping" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
          </span>
          <div className="flex flex-col leading-none">
            <span className="text-[9px] text-white/40">STATUS</span>
            <span className="text-emerald-400 font-semibold tracking-wide">ONLINE</span>
          </div>
        </div>
      </div>

      {/* CENTER: DRONE SELECTOR - TACTICAL LAYOUT */}
      <div className="flex-1 flex items-center justify-center">
        <div className="flex items-center gap-1 px-3 py-1 bg-white/[0.02] border border-white/5 rounded">
          <span className="text-[10px] text-white/40 tracking-wider mr-2">UNITS</span>
          {availableVideos.map((vid, idx) => {
            const selected = isSelected(vid.id);

            return (
              <div key={vid.id} className="relative group">
                <button
                  onClick={() => toggleVideo(vid)}
                  className={clsx(
                    'relative flex items-center justify-center w-8 h-6 transition-all border',
                    selected
                      ? 'bg-emerald-500/20 border-emerald-500/60 text-emerald-400'
                      : 'bg-white/[0.02] border-white/10 text-white/40 hover:border-white/30 hover:text-white/70'
                  )}
                >
                  {/* ACTIVE INDICATOR */}
                  {selected && (
                    <div className="absolute top-0 left-0 right-0 h-[2px] bg-emerald-400" />
                  )}

                  <span className="font-bold text-[11px] tracking-wider">
                    {String(idx + 1).padStart(2, '0')}
                  </span>
                </button>

                {/* TACTICAL TOOLTIP */}
<div className="pointer-events-none absolute bottom-full mb-2 left-1/2 -translate-x-1/2
                opacity-0 group-hover:opacity-100 transition-opacity duration-200">
  <div className="px-2 py-1 bg-black/95 border border-emerald-500/30
                  text-[10px] text-emerald-400 whitespace-nowrap
                  tracking-wider backdrop-blur-sm">

    {/* REGION — SAME AS RIGHT PANEL */}
    <div className="font-bold text-emerald-400">
      {DRONE_REGION_MAP[vid.id] || 'UNKNOWN REGION'}
    </div>

    {/* DRONE LABEL — SECONDARY */}
    <div className="mt-0.5 text-[9px] text-white/50 tracking-widest">
      {vid.label}
    </div>
  </div>

  {/* ARROW */}
  <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px">
    <div className="border-4 border-transparent border-t-emerald-500/30" />
  </div>
</div>

              </div>
            );
          })}
        </div>
      </div>

      {/* RIGHT: TECHNICAL METRICS */}
      <div className="flex items-center gap-4">
        {/* Performance Metrics */}
        <div className="flex items-center gap-3">
          <div className="flex flex-col items-end leading-none">
            <span className="text-[9px] text-white/40 tracking-wider">FRAMERATE</span>
            <div className="flex items-center gap-1">
              <Activity className="w-3 h-3 text-white/30" strokeWidth={1.5} />
              <span className={clsx('font-bold tabular-nums', getFpsColor(fps))}>
                {fps > 0 ? fps.toFixed(0) : '--'}
              </span>
              <span className="text-white/40 text-[10px]">FPS</span>
            </div>
          </div>

          <div className="flex flex-col items-end leading-none">
            <span className="text-[9px] text-white/40 tracking-wider">LATENCY</span>
            <div className="flex items-center gap-1">
              <Zap className="w-3 h-3 text-white/30" strokeWidth={1.5} />
              <span className={clsx('font-bold tabular-nums', getLatencyColor(latency))}>
                {latency.toFixed(0)}
              </span>
              <span className="text-white/40 text-[10px]">MS</span>
            </div>
          </div>

          <div className="flex flex-col items-end leading-none">
            <span className="text-[9px] text-white/40 tracking-wider">BANDWIDTH</span>
            <div className="flex items-center gap-1">
              <Signal className="w-3 h-3 text-white/30" strokeWidth={1.5} />
              <span className="font-bold text-cyan-400 tabular-nums">
                {bandwidth.toFixed(1)}
              </span>
              <span className="text-white/40 text-[10px]">MB/S</span>
            </div>
          </div>
        </div>

        <div className="h-6 w-px bg-white/10" />

        {/* Connection Status */}
        <div className="flex flex-col items-end leading-none">
          <span className="text-[9px] text-white/40 tracking-wider">SIGNAL</span>
          <div className="flex items-center gap-1">
            <Wifi className="w-3 h-3 text-white/30" strokeWidth={1.5} />
            <span className={clsx('font-bold tabular-nums', getSignalColor(signalStrength))}>
              {signalStrength.toFixed(0)}%
            </span>
            <span className="text-white/40">({activeCount}/{total})</span>
          </div>
        </div>

        <div className="h-6 w-px bg-white/10" />

        {/* Time Display */}
        <div className="flex flex-col items-end leading-none">
          <span className="text-[9px] text-white/40 tracking-wider">{date}</span>
          <div className="flex items-center gap-1">
            <Clock className="w-3 h-3 text-white/30" strokeWidth={1.5} />
            <span className="text-emerald-400 font-bold tracking-wider tabular-nums">
              {time}
            </span>
            <span className="text-white/40 text-[10px]">UTC</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;