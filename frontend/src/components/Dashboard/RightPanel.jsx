import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, AlertCircle, XCircle, X, Clock, Activity, Users, Gauge, Search, Square, Play, Loader2, Crosshair, ScanSearch, BarChart3, Car, Truck, GitBranch } from 'lucide-react';
import clsx from 'clsx';

const API_BASE_URL = import.meta.env.DEV
  ? '/api'
  : `http://${window.location.hostname}:9010`;

const severityConfig = {
  critical: {
    color: 'text-red-400',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/30',
    icon: XCircle,
    label: 'CRITICAL'
  },
  high: {
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30',
    icon: AlertTriangle,
    label: 'HIGH'
  },
  medium: {
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500/30',
    icon: AlertCircle,
    label: 'MEDIUM'
  },
};

const formatTimeAgo = (timestamp) => {
  const seconds = Math.floor(Date.now() / 1000 - timestamp);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
};

/* ============================
   IRIS FORENSICS PANEL (TOP)
============================ */
const ForensicsPanel = ({ selectedVideos = [] }) => {
  const [prompt, setPrompt] = useState('');
  const [source, setSource] = useState('');
  const [confidence, setConfidence] = useState(0.7);
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [samStatus, setSamStatus] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    if (selectedVideos.length > 0 && !source) {
      setSource(selectedVideos[0].id);
    }
  }, [selectedVideos, source]);

  useEffect(() => {
    if (!isRunning || !source) {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }
    const fetchResult = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/sam/result/${source}`);
        if (res.ok) {
          const data = await res.json();
          if (data.annotated_frame) {
            setResult(data);
            setIsLoading(false);
          }
        }
      } catch (e) {
        console.error('Result poll failed:', e);
      }
    };
    fetchResult();
    pollRef.current = setInterval(fetchResult, 2500);
    return () => clearInterval(pollRef.current);
  }, [isRunning, source]);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/sam/status`);
        if (res.ok) {
          const data = await res.json();
          setSamStatus(data);
          if (source && data.active_sources?.includes(source)) {
            setIsRunning(true);
          }
        }
      } catch (e) { }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [source]);

  const handleConfidenceChange = (val) => {
    setConfidence(val);
  };

  const handleStart = async () => {
    if (!prompt.trim() || !source) return;
    setIsLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE_URL}/sam/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source,
          prompt: prompt.trim(),
          confidence,
          show_boxes: true,
          show_masks: true,
        }),
      });
      if (res.ok) {
        setIsRunning(true);
      } else {
        setIsLoading(false);
      }
    } catch (e) {
      console.error('Forensics start failed:', e);
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    try {
      await fetch(`${API_BASE_URL}/sam/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source }),
      });
    } catch (e) {
      console.error('Forensics stop failed:', e);
    }
    setIsRunning(false);
    setIsLoading(false);
    setResult(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !isRunning) {
      handleStart();
    }
  };

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/10 bg-[#060a12]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Crosshair className="w-4 h-4 text-cyan-500/70" />
            <div className="text-xs text-cyan-500 font-black uppercase tracking-[0.25em]">
              IRIS Forensics
            </div>
            <div className={clsx(
              'w-2 h-2 rounded-full',
              isRunning ? 'bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.6)]' : samStatus?.model_loaded ? 'bg-cyan-500' : 'bg-white/20'
            )} />
          </div>
          {isRunning && result && (
            <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/15 px-2 py-0.5 border border-emerald-500/20">
              {result.count} FOUND
            </span>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="px-4 pt-3 space-y-2">
        <select
          value={source}
          onChange={(e) => setSource(e.target.value)}
          disabled={isRunning}
          className="w-full bg-black/60 border border-white/10 text-[11px] text-white/80 font-mono uppercase tracking-wider px-3 py-2 focus:outline-none focus:border-cyan-500/50 disabled:opacity-40"
        >
          <option value="">SELECT SOURCE</option>
          {selectedVideos.map((v) => (
            <option key={v.id} value={v.id}>
              {v.label || v.id}
            </option>
          ))}
        </select>

        <div className="flex gap-1.5">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isRunning}
            placeholder="e.g. yellow vehicle, stalled car"
            className="flex-1 bg-black/60 border border-white/10 text-xs text-white font-mono px-3 py-2.5 placeholder:text-white/20 focus:outline-none focus:border-cyan-500/50 disabled:opacity-40"
          />
          {isRunning ? (
            <button
              onClick={handleStop}
              className="px-4 py-2.5 bg-red-500/20 border border-red-500/40 text-red-400 text-[10px] font-black uppercase tracking-wider hover:bg-red-500/30 transition-colors flex items-center gap-1.5"
            >
              <Square className="w-3 h-3" />
              STOP
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={!prompt.trim() || !source}
              className="px-4 py-2.5 bg-cyan-500/20 border border-cyan-500/40 text-cyan-400 text-[10px] font-black uppercase tracking-wider hover:bg-cyan-500 hover:text-black transition-colors disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1.5"
            >
              <Play className="w-3 h-3" />
              RUN
            </button>
          )}
        </div>

        <div className="flex items-center gap-3">
          <span className="text-[9px] text-white/40 font-mono uppercase tracking-wider">Conf</span>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={confidence}
            onChange={(e) => handleConfidenceChange(parseFloat(e.target.value))}
            className="flex-1 h-1 appearance-none bg-white/10 rounded accent-cyan-500"
          />
          <span className="text-[11px] text-cyan-400 font-mono font-bold w-8 text-right">{confidence.toFixed(2)}</span>
        </div>

      </div>

      {/* Result Area */}
      <div className="flex-1 overflow-y-auto min-h-0 px-4 py-3">
        {isLoading && !result && (
          <div className="flex items-center justify-center h-36 border border-white/5 bg-black/30">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-6 h-6 text-cyan-500/60 animate-spin" />
              <span className="text-[10px] text-white/30 font-mono uppercase tracking-wider">Processing frame...</span>
            </div>
          </div>
        )}

        {result && (
          <div className="space-y-2">
            <div className="bg-black/60 border border-white/10 p-3 font-mono text-[10px] space-y-1.5">
              <div className="flex items-center justify-between border-b border-white/5 pb-1.5 mb-1.5">
                <span className="text-cyan-500/70 font-black uppercase tracking-wider">Output</span>
                <span className="text-white/50">
                  {new Date(result.timestamp * 1000).toLocaleTimeString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/40">Prompt</span>
                <span className="text-cyan-400">"{result.prompt}"</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/40">Detections</span>
                <span className="text-emerald-400 font-bold">{result.count}</span>
              </div>
              {result.detections && result.detections.length > 0 && (
                <div className="mt-2 pt-1.5 border-t border-white/5 space-y-1">
                  <span className="text-white/30 uppercase tracking-wider text-[9px]">Scores</span>
                  {result.detections.map((det, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="text-white/30 w-4">#{i + 1}</span>
                      <div className="flex-1 h-1 bg-white/5 rounded overflow-hidden">
                        <div
                          className="h-full bg-cyan-500/70 rounded"
                          style={{ width: `${det.score * 100}%` }}
                        />
                      </div>
                      <span className="text-white/60 w-10 text-right">{(det.score * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {!isLoading && !result && !isRunning && (
          <div className="flex items-center justify-center h-36 border border-dashed border-white/5 bg-black/20">
            <div className="text-center">
              <ScanSearch className="w-5 h-5 text-white/10 mx-auto mb-2" />
              <span className="text-[10px] text-white/20 font-mono uppercase block">Enter a prompt to detect</span>
              <span className="text-[9px] text-white/10 font-mono block mt-1">Results update every 2s</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

/* ============================
   CONGESTION PANEL (metrics + alerts)
============================ */
const CongestionPanel = ({ selectedVideos = [], alerts = [], onSelectAlert }) => {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (res.ok) {
          const data = await res.json();
          setMetrics(data);
        }
      } catch (e) {}
    };
    poll();
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, []);

  // Aggregate metrics across selected videos
  const sourceIds = selectedVideos.map(v => v.id);
  let totalCongestion = 0, totalDensity = 0, totalMobility = 0;
  let totalStalled = 0, totalSlow = 0, totalMedium = 0, totalFast = 0;
  let totalDetections = 0;
  let count = 0;

  if (metrics) {
    for (const src of sourceIds) {
      const m = metrics[src];
      if (!m) continue;
      count++;
      totalCongestion += m.congestion_index || 0;
      totalDensity += m.traffic_density || 0;
      totalMobility += m.mobility_index || 0;
      totalStalled += m.stalled_pct || 0;
      totalSlow += m.slow_pct || 0;
      totalMedium += m.medium_pct || 0;
      totalFast += m.fast_pct || 0;
      totalDetections += m.detection_count || 0;
    }
  }

  const avg = (v) => count > 0 ? Math.round(v / count) : 0;
  const congestion = avg(totalCongestion);
  const density = avg(totalDensity);
  const mobility = avg(totalMobility);
  const stalled = avg(totalStalled);
  const slow = avg(totalSlow);
  const medium = avg(totalMedium);
  const fast = avg(totalFast);

  const congestionColor = congestion >= 60 ? 'text-red-400' : congestion >= 40 ? 'text-orange-400' : congestion >= 20 ? 'text-yellow-400' : 'text-emerald-400';
  const congestionBorder = congestion >= 60 ? 'border-red-500/20' : congestion >= 40 ? 'border-orange-500/20' : congestion >= 20 ? 'border-yellow-500/20' : 'border-emerald-500/20';

  const speedBands = [
    { label: 'STALLED', value: stalled, color: 'bg-red-500', text: 'text-red-400' },
    { label: 'SLOW', value: slow, color: 'bg-orange-500', text: 'text-orange-400' },
    { label: 'MEDIUM', value: medium, color: 'bg-yellow-500', text: 'text-yellow-400' },
    { label: 'FAST', value: fast, color: 'bg-emerald-500', text: 'text-emerald-400' },
  ];

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/10 bg-[#060a12] shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-500/70" />
            <div className="text-xs text-cyan-500 font-black uppercase tracking-[0.25em]">Congestion</div>
          </div>
          <span className="text-[10px] font-bold text-cyan-400 bg-cyan-500/15 px-2 py-0.5 border border-cyan-500/20">
            {count} SOURCE{count !== 1 ? 'S' : ''}
          </span>
        </div>
      </div>

      {/* Scrollable content: metrics + alerts */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {/* Metrics section */}
        <div className="px-4 py-3 space-y-3">
          {/* Hero congestion index */}
          <div className={`bg-black/40 border ${congestionBorder} p-4 text-center`}>
            <div className="text-[10px] text-white/40 uppercase tracking-widest mb-1">Congestion Index</div>
            <div className={`text-5xl font-black leading-none ${congestionColor}`}>{congestion}%</div>
            <div className="text-[9px] text-white/20 mt-1 font-mono">{count} source{count !== 1 ? 's' : ''} aggregated</div>
          </div>

          {/* Key metrics row */}
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-black/30 border border-white/5 p-2 text-center">
              <div className="text-[8px] text-white/30 uppercase tracking-wider">Density</div>
              <div className="text-lg font-black text-orange-400 leading-tight">{density}%</div>
            </div>
            <div className="bg-black/30 border border-white/5 p-2 text-center">
              <div className="text-[8px] text-white/30 uppercase tracking-wider">Mobility</div>
              <div className="text-lg font-black text-cyan-400 leading-tight">{mobility}</div>
            </div>
            <div className="bg-black/30 border border-white/5 p-2 text-center">
              <div className="text-[8px] text-white/30 uppercase tracking-wider">Vehicles</div>
              <div className="text-lg font-black text-white/80 leading-tight">{totalDetections}</div>
            </div>
          </div>

          {/* Speed distribution */}
          <div className="space-y-2">
            <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Speed Distribution</div>
            {speedBands.map(band => (
              <div key={band.label} className="space-y-0.5">
                <div className="flex justify-between items-center">
                  <span className={`text-[10px] font-bold uppercase tracking-wider ${band.text}`}>{band.label}</span>
                  <span className="text-[10px] font-black text-white/70">{band.value}%</span>
                </div>
                <div className="h-1.5 bg-white/5 rounded overflow-hidden">
                  <motion.div
                    className={`h-full rounded ${band.color}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${band.value}%` }}
                    transition={{ type: 'spring', stiffness: 60 }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Per-source breakdown */}
          {sourceIds.length > 1 && metrics && (
            <div className="space-y-1.5 pt-2 border-t border-white/5">
              <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Per Source</div>
              {sourceIds.map(src => {
                const m = metrics?.[src];
                const ci = m?.congestion_index || 0;
                const droneMatch = src.match(/bcpdrone(\d+)/i);
                const label = droneMatch ? `DRONE ${droneMatch[1]}` : src;
                const ciColor = ci >= 60 ? 'text-red-400' : ci >= 40 ? 'text-orange-400' : ci >= 20 ? 'text-yellow-400' : 'text-emerald-400';
                return (
                  <div key={src} className="flex justify-between items-center">
                    <span className="text-[10px] text-white/50 font-mono uppercase">{label}</span>
                    <span className={`text-[10px] font-bold ${ciColor}`}>{ci}%</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Alerts section — contained within scroll */}
        <div className="border-t border-white/10">
          <div className="px-4 py-2 bg-[#060a12] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-3 h-3 text-red-400/60" />
              <div className="text-[9px] text-red-400/60 font-black uppercase tracking-[0.3em]">Alerts</div>
              {alerts.length > 0 && (
                <span className="px-1.5 py-0.5 bg-red-500/15 text-red-400 text-[9px] font-bold border border-red-500/20">
                  {alerts.length}
                </span>
              )}
            </div>
          </div>
          <div className="px-2 pb-2">
            {alerts.length === 0 ? (
              <div className="px-2 py-4 text-center">
                <div className="text-white/15 text-[10px] font-mono">No active alerts</div>
              </div>
            ) : (
              <div className="space-y-1">
                {alerts.map((alert) => {
                  const config = severityConfig[alert.severity] || severityConfig.medium;
                  const Icon = config.icon;
                  const droneMatch = alert.source.match(/bcpdrone(\d+)/i);
                  const droneLabel = droneMatch ? `DRONE ${droneMatch[1]}` : alert.source.toUpperCase();

                  return (
                    <div
                      key={alert.id}
                      onClick={() => onSelectAlert(alert)}
                      className={clsx(
                        'flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors rounded',
                        'border border-white/5 hover:border-white/10',
                        config.bgColor
                      )}
                    >
                      <div className={clsx('w-1 h-8 rounded-full shrink-0', config.color.replace('text-', 'bg-'))} />
                      <Icon className={clsx('w-3.5 h-3.5 shrink-0', config.color)} />
                      <div className="flex-1 min-w-0 font-mono">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-bold text-white truncate">{droneLabel}</span>
                          <span className={clsx('text-[10px] font-black', config.color)}>{alert.congestion}%</span>
                        </div>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span className={clsx('text-[8px] uppercase font-bold', config.color)}>{config.label}</span>
                          <span className="text-[8px] text-white/30">{formatTimeAgo(alert.timestamp)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

/* ============================
   VEHICLE STATS PANEL
============================ */
const CLASS_COLORS = {
  car: { bar: 'bg-cyan-500', text: 'text-cyan-400', shadow: 'shadow-[0_0_8px_rgba(6,182,212,0.4)]' },
  van: { bar: 'bg-blue-500', text: 'text-blue-400', shadow: 'shadow-[0_0_8px_rgba(59,130,246,0.4)]' },
  truck: { bar: 'bg-orange-500', text: 'text-orange-400', shadow: 'shadow-[0_0_8px_rgba(249,115,22,0.4)]' },
  bus: { bar: 'bg-yellow-500', text: 'text-yellow-400', shadow: 'shadow-[0_0_8px_rgba(234,179,8,0.4)]' },
  motor: { bar: 'bg-purple-500', text: 'text-purple-400', shadow: 'shadow-[0_0_8px_rgba(168,85,247,0.4)]' },
  bicycle: { bar: 'bg-emerald-500', text: 'text-emerald-400', shadow: 'shadow-[0_0_8px_rgba(16,185,129,0.4)]' },
};

const VehicleStatsPanel = ({ selectedVideos = [] }) => {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (res.ok) {
          const data = await res.json();
          setMetrics(data);
        }
      } catch (e) {}
    };
    poll();
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, []);

  // Aggregate class_counts across selected videos
  const aggregated = {};
  let totalVehicles = 0;
  const sourceIds = selectedVideos.map(v => v.id);

  if (metrics) {
    for (const src of sourceIds) {
      const m = metrics[src];
      if (!m) continue;
      totalVehicles += m.detection_count || 0;
      const cc = m.class_counts || {};
      for (const [cls, count] of Object.entries(cc)) {
        aggregated[cls] = (aggregated[cls] || 0) + count;
      }
    }
  }

  const sorted = Object.entries(aggregated).sort((a, b) => b[1] - a[1]);
  const maxCount = sorted.length > 0 ? sorted[0][1] : 1;

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      <div className="px-4 py-3 border-b border-white/10 bg-[#060a12]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Car className="w-4 h-4 text-emerald-500/70" />
            <div className="text-xs text-emerald-500 font-black uppercase tracking-[0.25em]">Vehicle Analytics</div>
          </div>
          <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/15 px-2 py-0.5 border border-emerald-500/20">
            {totalVehicles} TOTAL
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0 px-4 py-3 space-y-4">
        {/* Hero count */}
        <div className="bg-black/40 border border-emerald-500/20 p-4 text-center">
          <div className="text-[10px] text-white/40 uppercase tracking-widest mb-1">Total Vehicles Detected</div>
          <div className="text-5xl font-black text-emerald-400 leading-none">{totalVehicles}</div>
          <div className="text-[9px] text-white/20 mt-1 font-mono">{sourceIds.length} source{sourceIds.length !== 1 ? 's' : ''} active</div>
        </div>

        {/* Per-class breakdown */}
        <div className="space-y-2">
          <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Classification Breakdown</div>
          {sorted.length === 0 ? (
            <div className="text-[10px] text-white/20 font-mono py-4 text-center">No detections yet</div>
          ) : (
            sorted.map(([cls, count]) => {
              const cc = CLASS_COLORS[cls] || { bar: 'bg-white/40', text: 'text-white/60', shadow: '' };
              const pct = Math.round((count / maxCount) * 100);
              return (
                <div key={cls} className="space-y-1">
                  <div className="flex justify-between items-center">
                    <span className={`text-[10px] font-bold uppercase tracking-wider ${cc.text}`}>{cls}</span>
                    <span className="text-[11px] font-black text-white/80">{count}</span>
                  </div>
                  <div className="h-1.5 bg-white/5 rounded overflow-hidden">
                    <motion.div
                      className={`h-full rounded ${cc.bar} ${cc.shadow}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{ type: 'spring', stiffness: 60 }}
                    />
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Per-source breakdown */}
        {sourceIds.length > 1 && (
          <div className="space-y-2 pt-2 border-t border-white/5">
            <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Per Source</div>
            {sourceIds.map(src => {
              const m = metrics?.[src];
              const cnt = m?.detection_count || 0;
              const droneMatch = src.match(/bcpdrone(\d+)/i);
              const label = droneMatch ? `DRONE ${droneMatch[1]}` : src;
              return (
                <div key={src} className="flex justify-between items-center">
                  <span className="text-[10px] text-white/50 font-mono uppercase">{label}</span>
                  <span className="text-[10px] font-bold text-emerald-400">{cnt}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

/* ============================
   FLOW STATS PANEL
============================ */
const FlowStatsPanel = ({ selectedVideos = [] }) => {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (res.ok) {
          const data = await res.json();
          setMetrics(data);
        }
      } catch (e) {}
    };
    poll();
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, []);

  // Aggregate across selected videos
  const sourceIds = selectedVideos.map(v => v.id);
  let totalMobility = 0;
  let totalStalled = 0, totalSlow = 0, totalMedium = 0, totalFast = 0;
  let count = 0;

  if (metrics) {
    for (const src of sourceIds) {
      const m = metrics[src];
      if (!m) continue;
      count++;
      totalMobility += m.mobility_index || 0;
      totalStalled += m.stalled_pct || 0;
      totalSlow += m.slow_pct || 0;
      totalMedium += m.medium_pct || 0;
      totalFast += m.fast_pct || 0;
    }
  }

  const avg = (v) => count > 0 ? Math.round(v / count) : 0;
  const mobility = avg(totalMobility);
  const stalled = avg(totalStalled);
  const slow = avg(totalSlow);
  const medium = avg(totalMedium);
  const fast = avg(totalFast);

  const speedBands = [
    { label: 'STALLED', value: stalled, color: 'bg-red-500', text: 'text-red-400' },
    { label: 'SLOW', value: slow, color: 'bg-orange-500', text: 'text-orange-400' },
    { label: 'MEDIUM', value: medium, color: 'bg-yellow-500', text: 'text-yellow-400' },
    { label: 'FAST', value: fast, color: 'bg-emerald-500', text: 'text-emerald-400' },
  ];

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      <div className="px-4 py-3 border-b border-white/10 bg-[#060a12]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-purple-500/70" />
            <div className="text-xs text-purple-500 font-black uppercase tracking-[0.25em]">Traffic Flow</div>
          </div>
          <span className="text-[10px] font-bold text-purple-400 bg-purple-500/15 px-2 py-0.5 border border-purple-500/20">
            IDX {mobility}
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0 px-4 py-3 space-y-4">
        {/* Hero mobility */}
        <div className="bg-black/40 border border-purple-500/20 p-4 text-center">
          <div className="text-[10px] text-white/40 uppercase tracking-widest mb-1">Mobility Index</div>
          <div className="text-5xl font-black text-purple-400 leading-none">{mobility}</div>
          <div className="text-[9px] text-white/20 mt-1 font-mono">{count} source{count !== 1 ? 's' : ''} aggregated</div>
        </div>

        {/* Speed distribution */}
        <div className="space-y-3">
          <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Speed Distribution</div>
          {speedBands.map(band => (
            <div key={band.label} className="space-y-1">
              <div className="flex justify-between items-center">
                <span className={`text-[10px] font-bold uppercase tracking-wider ${band.text}`}>{band.label}</span>
                <span className="text-[11px] font-black text-white/80">{band.value}%</span>
              </div>
              <div className="h-2 bg-white/5 rounded overflow-hidden">
                <motion.div
                  className={`h-full rounded ${band.color}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${band.value}%` }}
                  transition={{ type: 'spring', stiffness: 60 }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const RightPanel = ({ useCase = 'congestion', selectedVideos = [] }) => {
  const [alerts, setAlerts] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState(null);
  // Track session start time — only show alerts newer than this
  const [sessionStart] = useState(() => Date.now() / 1000);

  // Only poll alerts when in congestion mode
  useEffect(() => {
    if (useCase !== 'congestion') {
      setAlerts([]);
      return;
    }
    const fetchAlerts = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/alerts?limit=50`);
        if (res.ok) {
          const data = await res.json();
          // Only show alerts from this session (newer than sessionStart)
          const fresh = (data.alerts || []).filter(a => a.timestamp >= sessionStart);
          setAlerts(fresh);
        }
      } catch (e) {
        console.error('Failed to fetch alerts:', e);
      }
    };
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 3000);
    return () => clearInterval(interval);
  }, [useCase, sessionStart]);

  return (
    <>
      <motion.div
        initial={{ x: 40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        className="w-[320px] h-full bg-[#070b14]/80 border-l border-white/10 flex flex-col z-40"
      >
        {useCase === 'congestion' && (
          <CongestionPanel selectedVideos={selectedVideos} alerts={alerts} onSelectAlert={setSelectedAlert} />
        )}

        {useCase === 'vehicle' && (
          <VehicleStatsPanel selectedVideos={selectedVideos} />
        )}

        {useCase === 'flow' && (
          <FlowStatsPanel selectedVideos={selectedVideos} />
        )}

        {useCase === 'forensics' && (
          <ForensicsPanel selectedVideos={selectedVideos} />
        )}
      </motion.div>

      {/* ================= ALERT DETAIL MODAL ================= */}
      <AnimatePresence>
        {selectedAlert && (
          <motion.div
            className="fixed inset-0 bg-black/90 z-[100] flex items-center justify-center p-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedAlert(null)}
          >
            <motion.div
              onClick={(e) => e.stopPropagation()}
              className="bg-[#0a1120] max-w-4xl w-full border border-white/10 relative overflow-hidden"
              initial={{ scale: 0.96 }}
              animate={{ scale: 1 }}
            >
              <button
                onClick={() => setSelectedAlert(null)}
                className="absolute top-4 right-4 z-10 border border-white/10 p-2 bg-black/50 hover:bg-white/10 transition-colors"
              >
                <X className="w-4 h-4 text-white/70" />
              </button>

              <div className="p-4 border-b border-white/10 bg-black/30">
                <div className="flex items-center gap-3">
                  {(() => {
                    const config = severityConfig[selectedAlert.severity] || severityConfig.medium;
                    const Icon = config.icon;
                    const droneMatch = selectedAlert.source.match(/bcpdrone(\d+)/i);
                    const droneLabel = droneMatch ? `DRONE ${droneMatch[1]}` : selectedAlert.source.toUpperCase();
                    return (
                      <>
                        <Icon className={clsx('w-6 h-6', config.color)} />
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-lg font-black text-white uppercase tracking-wide">
                              {droneLabel}
                            </span>
                          </div>
                          <div className={clsx('text-sm font-bold', config.color)}>
                            {config.label} ALERT {selectedAlert.congestion > 0 && `\u2022 ${selectedAlert.congestion}% Congestion`}
                          </div>
                        </div>
                      </>
                    );
                  })()}
                </div>
              </div>

              <div className="flex flex-col md:flex-row">
                <div className="flex-1 relative bg-black">
                  <img
                    src={`${API_BASE_URL}/alerts/${selectedAlert.id}/screenshot`}
                    alt="Alert Screenshot"
                    className="w-full h-auto max-h-[50vh] object-contain"
                    onError={(e) => {
                      e.target.src = '/placeholder-alert.jpg';
                      e.target.onerror = null;
                    }}
                  />
                  <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/70 text-xs text-white/60 font-mono">
                    {selectedAlert.time_str}
                  </div>
                </div>

                <div className="w-full md:w-64 p-4 bg-black/20 border-t md:border-t-0 md:border-l border-white/10">
                  <div className="text-xs text-cyan-500/60 font-black uppercase tracking-widest mb-4">
                    Metrics Snapshot
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <Gauge className="w-4 h-4 text-red-400" />
                      <div className="flex-1">
                        <div className="text-[10px] text-white/40 uppercase">Congestion</div>
                        <div className="text-lg font-black text-red-400">
                          {selectedAlert.metrics?.congestion_index || selectedAlert.congestion || 0}%
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <Activity className="w-4 h-4 text-orange-400" />
                      <div className="flex-1">
                        <div className="text-[10px] text-white/40 uppercase">Traffic Density</div>
                        <div className="text-lg font-black text-orange-400">
                          {selectedAlert.metrics?.traffic_density || 0}%
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <Activity className="w-4 h-4 text-cyan-400" />
                      <div className="flex-1">
                        <div className="text-[10px] text-white/40 uppercase">Mobility Index</div>
                        <div className="text-lg font-black text-cyan-400">
                          {selectedAlert.metrics?.mobility_index || 0}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <Users className="w-4 h-4 text-white/60" />
                      <div className="flex-1">
                        <div className="text-[10px] text-white/40 uppercase">Vehicles Detected</div>
                        <div className="text-lg font-black text-white">
                          {selectedAlert.metrics?.detection_count || 0}
                        </div>
                      </div>
                    </div>

                    {selectedAlert.metrics && Object.keys(selectedAlert.metrics).length > 0 && (
                      <div className="mt-4 pt-3 border-t border-white/10">
                        <div className="text-[10px] text-white/40 uppercase mb-2">Speed Distribution</div>
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-red-400">Stalled</span>
                            <span className="text-white/70">{selectedAlert.metrics?.stalled_pct || 0}%</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-orange-400">Slow</span>
                            <span className="text-white/70">{selectedAlert.metrics?.slow_pct || 0}%</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-yellow-400">Medium</span>
                            <span className="text-white/70">{selectedAlert.metrics?.medium_pct || 0}%</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-emerald-400">Fast</span>
                            <span className="text-white/70">{selectedAlert.metrics?.fast_pct || 0}%</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default RightPanel;
