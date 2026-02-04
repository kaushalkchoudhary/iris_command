import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, AlertCircle, XCircle, X, Clock, Activity, Users, Gauge, Search, Square, Play, Loader2, Crosshair, ScanSearch, BarChart3, Car, Truck, GitBranch, ShieldAlert, Flame, ArrowUpRight, ArrowDownRight, Minus, Radio, Siren, Ban, Wrench, Navigation, MapPin, SlidersHorizontal, TrendingUp, TrendingDown, Shield, Zap, Eye } from 'lucide-react';
import clsx from 'clsx';
import { API_BASE_URL } from '../../config';

/* ============================
   CONFIDENCE CONTROL COMPONENT
============================ */
const ConfidenceControl = ({ selectedVideos = [], accentColor = 'cyan' }) => {
  const [confidence, setConfidence] = useState(0.15);
  const [isUpdating, setIsUpdating] = useState(false);
  const debounceRef = useRef(null);

  // Fetch current confidence from first selected source
  useEffect(() => {
    if (selectedVideos.length === 0) return;
    const fetchConfidence = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/confidence/${selectedVideos[0].id}`);
        if (res.ok) {
          const data = await res.json();
          setConfidence(data.confidence || 0.15);
        }
      } catch (e) {}
    };
    fetchConfidence();
  }, [selectedVideos]);

  // Update confidence for all selected sources
  const updateConfidence = useCallback(async (newConf) => {
    setIsUpdating(true);
    try {
      await Promise.all(
        selectedVideos.map(v =>
          fetch(`${API_BASE_URL}/confidence/${v.id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confidence: newConf }),
          })
        )
      );
    } catch (e) {
      console.error('Failed to update confidence:', e);
    }
    setIsUpdating(false);
  }, [selectedVideos]);

  const handleChange = (e) => {
    const newVal = parseFloat(e.target.value);
    setConfidence(newVal);

    // Debounce API calls
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      updateConfidence(newVal);
    }, 150);
  };

  if (selectedVideos.length === 0) return null;

  const colors = {
    cyan: { text: 'text-cyan-400', bg: 'bg-cyan-500/15', border: 'border-cyan-500/30', accent: 'accent-cyan-500' },
    emerald: { text: 'text-emerald-400', bg: 'bg-emerald-500/15', border: 'border-emerald-500/30', accent: 'accent-emerald-500' },
    purple: { text: 'text-purple-400', bg: 'bg-purple-500/15', border: 'border-purple-500/30', accent: 'accent-purple-500' },
    amber: { text: 'text-amber-400', bg: 'bg-amber-500/15', border: 'border-amber-500/30', accent: 'accent-amber-500' },
  };
  const c = colors[accentColor] || colors.cyan;

  return (
    <div className={`${c.bg} border ${c.border} p-3 space-y-2`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className={`w-3.5 h-3.5 ${c.text}/60`} />
          <span className="text-[9px] font-black uppercase tracking-widest text-white/50">
            Detection Confidence
          </span>
        </div>
        <span className={`text-sm font-black ${c.text} tabular-nums`}>
          {(confidence * 100).toFixed(0)}%
        </span>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-[8px] text-white/30 font-mono">5%</span>
        <input
          type="range"
          min="0.05"
          max="0.95"
          step="0.05"
          value={confidence}
          onChange={handleChange}
          className={`flex-1 h-1.5 appearance-none bg-white/10 rounded cursor-pointer ${c.accent}`}
          style={{
            background: `linear-gradient(to right, ${accentColor === 'cyan' ? '#06b6d4' : accentColor === 'emerald' ? '#10b981' : accentColor === 'purple' ? '#a855f7' : '#f59e0b'} 0%, ${accentColor === 'cyan' ? '#06b6d4' : accentColor === 'emerald' ? '#10b981' : accentColor === 'purple' ? '#a855f7' : '#f59e0b'} ${(confidence - 0.05) / 0.9 * 100}%, rgba(255,255,255,0.1) ${(confidence - 0.05) / 0.9 * 100}%, rgba(255,255,255,0.1) 100%)`
          }}
        />
        <span className="text-[8px] text-white/30 font-mono">95%</span>
      </div>
      <div className="flex items-center justify-between text-[8px] text-white/25">
        <span>More detections</span>
        <span>Higher precision</span>
      </div>
      {isUpdating && (
        <div className="text-[8px] text-white/30 text-center animate-pulse">Updating...</div>
      )}
    </div>
  );
};

/* ============================
   OVERLAY TOGGLES
============================ */
const OverlayToggles = ({ selectedVideos = [], accentColor = 'cyan', useCase = '' }) => {
  const [overlay, setOverlay] = useState({
    heatmap: true,
    heatmap_full: true,
    heatmap_trails: true,
    bboxes: false,
  });
  const [isUpdating, setIsUpdating] = useState(false);

  useEffect(() => {
    if (selectedVideos.length === 0) return;
    const fetchOverlay = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/overlays/${selectedVideos[0].id}`);
        if (res.ok) {
          const data = await res.json();
          setOverlay({
            heatmap: data.heatmap ?? true,
            heatmap_full: data.heatmap_full ?? true,
            heatmap_trails: data.heatmap_trails ?? true,
            bboxes: data.bboxes ?? false,
          });
        }
      } catch (e) {}
    };
    fetchOverlay();
  }, [selectedVideos]);

  const updateOverlay = useCallback(async (updates) => {
    setIsUpdating(true);
    try {
      await Promise.all(
        selectedVideos.map(v =>
          fetch(`${API_BASE_URL}/overlays/${v.id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates),
          })
        )
      );
    } catch (e) {
      console.error('Failed to update overlay:', e);
    }
    setIsUpdating(false);
  }, [selectedVideos]);

  const toggle = (key) => {
    const next = !overlay[key];
    const updates = { [key]: next };

    // When turning off master heatmap, turn off both types
    if (key === 'heatmap' && !next) {
      updates.heatmap_full = false;
      updates.heatmap_trails = false;
    }
    // When turning on a heatmap type, ensure master is on
    if ((key === 'heatmap_full' || key === 'heatmap_trails') && next) {
      updates.heatmap = true;
    }

    setOverlay(prev => ({ ...prev, ...updates }));
    updateOverlay(updates);
  };

  if (selectedVideos.length === 0) return null;

  const isCrowd = useCase === 'crowd';

  // For congestion mode: Heatmap (master), Full, Per-Vehicle, Boxes
  // For crowd mode: Heatmap, Full only
  const toggleItems = isCrowd
    ? [
        { key: 'heatmap', label: 'Heatmap' },
        { key: 'heatmap_full', label: 'Full' },
      ]
    : [
        { key: 'heatmap', label: 'Heat' },
        { key: 'heatmap_full', label: 'Full' },
        { key: 'heatmap_trails', label: 'Trail' },
        { key: 'bboxes', label: 'Box' },
      ];

  return (
    <div className="bg-black/40 border border-white/10 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className={`w-3.5 h-3.5 text-${accentColor}-400/80`} />
          <div className={`text-[9px] text-${accentColor}-400/80 uppercase tracking-widest font-bold`}>Overlays</div>
        </div>
        {isUpdating && <Loader2 className="w-3 h-3 text-white/40 animate-spin" />}
      </div>
      <div className={`grid gap-1.5 ${isCrowd ? 'grid-cols-2' : 'grid-cols-4'}`}>
        {toggleItems.map(t => (
          <button
            key={t.key}
            onClick={() => toggle(t.key)}
            className={clsx(
              'px-2 py-1 border text-[8px] font-bold uppercase tracking-wider transition-all',
              overlay[t.key]
                ? `bg-${accentColor}-500/20 border-${accentColor}-500/40 text-${accentColor}-300`
                : 'bg-white/5 border-white/10 text-white/40 hover:text-white/60'
            )}
            title={`Toggle ${t.label}`}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );
};

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
  const [showBoxes, setShowBoxes] = useState(true);
  const [showMasks, setShowMasks] = useState(true);
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
        console.error('SAM result poll failed:', e);
      }
    };
    fetchResult();
    pollRef.current = setInterval(fetchResult, 1000);
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
          show_boxes: showBoxes,
          show_masks: showMasks,
        }),
      });
      if (res.ok) {
        setIsRunning(true);
      } else {
        setIsLoading(false);
      }
    } catch (e) {
      console.error('SAM start failed:', e);
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
      console.error('SAM stop failed:', e);
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

  const handleToggle = async (field, value) => {
    if (field === 'showBoxes') setShowBoxes(value);
    else setShowMasks(value);
    if (!isRunning || !source) return;
    try {
      await fetch(`${API_BASE_URL}/sam/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source,
          show_boxes: field === 'showBoxes' ? value : showBoxes,
          show_masks: field === 'showMasks' ? value : showMasks,
        }),
      });
    } catch (e) {
      console.error('SAM update failed:', e);
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

        {/* Overlay Toggles */}
        <div className="flex items-center gap-1.5 pt-1">
          <button
            onClick={() => handleToggle('showBoxes', !showBoxes)}
            className={clsx(
              'flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 border text-[9px] font-black uppercase tracking-wider transition-all',
              showBoxes
                ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-400 shadow-[inset_0_0_12px_rgba(6,182,212,0.1)]'
                : 'bg-black/40 border-white/10 text-white/25 hover:border-white/20 hover:text-white/40'
            )}
          >
            <div className={clsx('w-1.5 h-1.5', showBoxes ? 'bg-cyan-400 shadow-[0_0_6px_rgba(6,182,212,0.8)]' : 'bg-white/20')} />
            BOUNDING BOXES
          </button>
          <button
            onClick={() => handleToggle('showMasks', !showMasks)}
            className={clsx(
              'flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 border text-[9px] font-black uppercase tracking-wider transition-all',
              showMasks
                ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-400 shadow-[inset_0_0_12px_rgba(6,182,212,0.1)]'
                : 'bg-black/40 border-white/10 text-white/25 hover:border-white/20 hover:text-white/40'
            )}
          >
            <div className={clsx('w-1.5 h-1.5', showMasks ? 'bg-cyan-400 shadow-[0_0_6px_rgba(6,182,212,0.8)]' : 'bg-white/20')} />
            SEGMENTATION
          </button>
        </div>

      </div>

      {/* Result Area - text metrics only (SAM3 output shows in main video area) */}
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
            {/* Compact Output — text-based metrics only */}
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
              <span className="text-[9px] text-white/10 font-mono block mt-1">Results update every 1s</span>
            </div>
          </div>
        )}
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

const TYPE_BAR_COLORS = {
  car: 'bg-cyan-500',
  van: 'bg-blue-500',
  truck: 'bg-orange-500',
  bus: 'bg-yellow-500',
  motor: 'bg-purple-500',
  bicycle: 'bg-emerald-500',
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

  // Aggregate class_counts and new vehicle analytics across selected videos
  const aggregated = {};
  let totalVehicles = 0;
  const sourceIds = selectedVideos.map(v => v.id);

  const aggState = { moving: 0, stopped: 0, abnormal: 0 };
  const aggBehavior = { stable: 0, start_stop: 0, erratic: 0 };
  const aggTypeInfluence = {};
  let aggDwellAvg = 0, aggDwellMax = 0, aggDwellOver = 0;
  let dwellSourceCount = 0;
  let aggLongestVehicles = [];
  let aggAttention = [];
  let aggHighImpact = [];

  if (metrics) {
    for (const src of sourceIds) {
      const m = metrics[src];
      if (!m) continue;
      totalVehicles += m.detection_count || 0;
      const cc = m.class_counts || {};
      for (const [cls, count] of Object.entries(cc)) {
        aggregated[cls] = (aggregated[cls] || 0) + count;
      }
      // State counts
      const sc = m.state_counts;
      if (sc) {
        aggState.moving += sc.moving || 0;
        aggState.stopped += sc.stopped || 0;
        aggState.abnormal += sc.abnormal || 0;
      }
      // Behavior counts
      const bc = m.behavior_counts;
      if (bc) {
        aggBehavior.stable += bc.stable || 0;
        aggBehavior.start_stop += bc.start_stop || 0;
        aggBehavior.erratic += bc.erratic || 0;
      }
      // Type influence
      const ti = m.type_influence;
      if (ti) {
        for (const [cls, frac] of Object.entries(ti)) {
          aggTypeInfluence[cls] = (aggTypeInfluence[cls] || 0) + frac;
        }
      }
      // Dwell stats
      const ds = m.dwell_stats;
      if (ds) {
        dwellSourceCount++;
        aggDwellAvg += ds.avg_dwell || 0;
        aggDwellMax = Math.max(aggDwellMax, ds.max_dwell || 0);
        aggDwellOver += ds.over_threshold || 0;
        if (ds.longest_vehicles) {
          aggLongestVehicles = aggLongestVehicles.concat(ds.longest_vehicles);
        }
      }
      // Attention list
      if (m.attention_list) {
        aggAttention = aggAttention.concat(m.attention_list);
      }
      if (m.high_impact_vehicles) {
        aggHighImpact = aggHighImpact.concat(m.high_impact_vehicles);
      }
    }
  }

  // Normalize type influence
  const tiTotal = Object.values(aggTypeInfluence).reduce((a, b) => a + b, 0) || 1;
  const normalizedTypeInfluence = {};
  for (const [cls, val] of Object.entries(aggTypeInfluence)) {
    normalizedTypeInfluence[cls] = Math.round((val / tiTotal) * 100);
  }
  const typeInfluenceSorted = Object.entries(normalizedTypeInfluence).sort((a, b) => b[1] - a[1]);

  const avgDwell = dwellSourceCount > 0 ? (aggDwellAvg / dwellSourceCount).toFixed(1) : '0.0';
  aggLongestVehicles.sort((a, b) => b.dwell - a.dwell);
  const topLongest = aggLongestVehicles.slice(0, 3);

  aggAttention.sort((a, b) => b.impact - a.impact);
  const topAttention = aggAttention.slice(0, 5);

  const sorted = Object.entries(aggregated).sort((a, b) => b[1] - a[1]);
  const maxCount = sorted.length > 0 ? sorted[0][1] : 1;

  const stateTotal = aggState.moving + aggState.stopped + aggState.abnormal;
  const statePcts = stateTotal > 0
    ? { moving: Math.round(aggState.moving / stateTotal * 100), stopped: Math.round(aggState.stopped / stateTotal * 100), abnormal: Math.round(aggState.abnormal / stateTotal * 100) }
    : { moving: 0, stopped: 0, abnormal: 0 };

  const attentionCount = topAttention.length;

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      <div className="px-4 py-3 border-b border-white/10 bg-[#060a12]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Car className="w-4 h-4 text-emerald-500/70" />
            <div className="text-xs text-emerald-500 font-black uppercase tracking-[0.25em]">Vehicle Analytics</div>
          </div>
          <div className="flex items-center gap-2">
            {attentionCount > 0 && (
              <span className="relative flex items-center px-1.5 py-0.5 bg-red-500/20 text-red-400 text-[9px] font-bold border border-red-500/30">
                <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-red-500 rounded-full animate-ping" />
                <ShieldAlert className="w-3 h-3 mr-1" />
                {attentionCount}
              </span>
            )}
            <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/15 px-2 py-0.5 border border-emerald-500/20">
              {totalVehicles} TOTAL
            </span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0 px-4 py-3 space-y-4">
        {/* Confidence Control */}
        <ConfidenceControl selectedVideos={selectedVideos} accentColor="emerald" />

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

        {/* 2A: Vehicle State Distribution */}
        {stateTotal > 0 && (
          <div className="space-y-3 pt-3 border-t border-white/5">
            <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Vehicle State</div>
            {/* State cards row */}
            <div className="grid grid-cols-3 gap-1.5">
              {[
                { label: 'Moving', count: aggState.moving, pct: statePcts.moving, color: 'emerald', icon: '>' },
                { label: 'Stopped', count: aggState.stopped, pct: statePcts.stopped, color: 'amber', icon: '||' },
                { label: 'Abnormal', count: aggState.abnormal, pct: statePcts.abnormal, color: 'red', icon: '!' },
              ].map(s => (
                <div key={s.label} className={`bg-black/50 border border-${s.color}-500/20 p-2 relative overflow-hidden`}>
                  {/* Background fill */}
                  <motion.div
                    className={`absolute bottom-0 left-0 right-0 bg-${s.color}-500/10`}
                    initial={{ height: 0 }}
                    animate={{ height: `${s.pct}%` }}
                    transition={{ type: 'spring', stiffness: 40, damping: 15 }}
                  />
                  <div className="relative">
                    <div className={`text-lg font-black text-${s.color}-400 leading-none`}>{s.count}</div>
                    <div className={`text-[8px] text-${s.color}-400/60 uppercase font-bold mt-1 tracking-wider`}>{s.label}</div>
                    <div className="text-[9px] text-white/25 font-mono mt-0.5">{s.pct}%</div>
                  </div>
                </div>
              ))}
            </div>
            {/* Stacked bar summary */}
            <div className="h-2 flex rounded-sm overflow-hidden bg-white/5">
              {statePcts.moving > 0 && (
                <motion.div className="bg-emerald-500 h-full shadow-[0_0_6px_rgba(16,185,129,0.4)]" initial={{ width: 0 }} animate={{ width: `${statePcts.moving}%` }} transition={{ type: 'spring', stiffness: 50 }} />
              )}
              {statePcts.stopped > 0 && (
                <motion.div className="bg-amber-500 h-full shadow-[0_0_6px_rgba(245,158,11,0.4)]" initial={{ width: 0 }} animate={{ width: `${statePcts.stopped}%` }} transition={{ type: 'spring', stiffness: 50 }} />
              )}
              {statePcts.abnormal > 0 && (
                <motion.div className="bg-red-500 h-full shadow-[0_0_6px_rgba(239,68,68,0.4)]" initial={{ width: 0 }} animate={{ width: `${statePcts.abnormal}%` }} transition={{ type: 'spring', stiffness: 50 }} />
              )}
            </div>
          </div>
        )}

        {/* 2B: Behavior Pattern */}
        {(aggBehavior.stable + aggBehavior.start_stop + aggBehavior.erratic) > 0 && (() => {
          const behaviorTotal = aggBehavior.stable + aggBehavior.start_stop + aggBehavior.erratic;
          const behaviors = [
            { label: 'Stable', count: aggBehavior.stable, color: 'emerald', desc: 'Consistent flow' },
            { label: 'Start-Stop', count: aggBehavior.start_stop, color: 'amber', desc: 'Intermittent stops' },
            { label: 'Erratic', count: aggBehavior.erratic, color: 'red', desc: 'Irregular path' },
          ];
          return (
            <div className="space-y-2 pt-3 border-t border-white/5">
              <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Behavior Pattern</div>
              <div className="space-y-1.5">
                {behaviors.map(b => {
                  const pct = behaviorTotal > 0 ? Math.round(b.count / behaviorTotal * 100) : 0;
                  return (
                    <div key={b.label} className="flex items-center gap-2.5">
                      <div className={`w-1.5 h-1.5 rounded-full bg-${b.color}-500 shrink-0 shadow-[0_0_6px] shadow-${b.color}-500/50`} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-0.5">
                          <span className={`text-[10px] font-bold text-${b.color}-400`}>{b.label}</span>
                          <div className="flex items-baseline gap-1.5">
                            <span className="text-[11px] font-black text-white/80">{b.count}</span>
                            <span className="text-[9px] text-white/25 font-mono">{pct}%</span>
                          </div>
                        </div>
                        <div className="h-1 bg-white/5 rounded overflow-hidden">
                          <motion.div
                            className={`h-full rounded bg-${b.color}-500/70`}
                            initial={{ width: 0 }}
                            animate={{ width: `${pct}%` }}
                            transition={{ type: 'spring', stiffness: 50 }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}

        {/* 2C: Traffic Influence by Type */}
        {typeInfluenceSorted.length > 0 && (
          <div className="space-y-2 pt-3 border-t border-white/5">
            <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Traffic Influence</div>
            <div className="space-y-1.5">
              {typeInfluenceSorted.map(([cls, pct]) => {
                const cc = CLASS_COLORS[cls] || { bar: 'bg-white/40', text: 'text-white/60', shadow: '' };
                return (
                  <div key={cls} className="flex items-center gap-2">
                    <span className={`text-[9px] font-bold uppercase w-12 truncate ${cc.text}`}>{cls}</span>
                    <div className="flex-1 h-2 bg-white/5 rounded overflow-hidden">
                      <motion.div
                        className={`h-full rounded ${cc.bar} ${cc.shadow}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ type: 'spring', stiffness: 50 }}
                      />
                    </div>
                    <span className="text-[10px] font-black text-white/70 w-8 text-right font-mono">{pct}%</span>
                  </div>
                );
              })}
            </div>
            {/* Mini summary bar */}
            <div className="h-1.5 flex rounded-sm overflow-hidden bg-white/5">
              {typeInfluenceSorted.map(([cls, pct]) => (
                <motion.div
                  key={cls}
                  className={`${TYPE_BAR_COLORS[cls] || 'bg-white/30'} h-full`}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ type: 'spring', stiffness: 50 }}
                />
              ))}
            </div>
          </div>
        )}

        {/* 2D: Dwell Time Stats */}
        {dwellSourceCount > 0 && (
          <div className="space-y-2.5 pt-3 border-t border-white/5">
            <div className="flex items-center gap-2">
              <Clock className="w-3 h-3 text-white/30" />
              <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Dwell Time</div>
            </div>
            <div className="grid grid-cols-3 gap-1.5">
              <div className="bg-black/50 border border-cyan-500/15 p-2.5 text-center">
                <div className="text-[8px] text-cyan-400/50 uppercase font-bold tracking-wider">Avg</div>
                <div className="text-base font-black text-cyan-400 leading-tight mt-0.5">{avgDwell}<span className="text-[9px] text-cyan-400/50">s</span></div>
              </div>
              <div className="bg-black/50 border border-orange-500/15 p-2.5 text-center">
                <div className="text-[8px] text-orange-400/50 uppercase font-bold tracking-wider">Max</div>
                <div className="text-base font-black text-orange-400 leading-tight mt-0.5">{aggDwellMax.toFixed(1)}<span className="text-[9px] text-orange-400/50">s</span></div>
              </div>
              <div className="bg-black/50 border border-red-500/15 p-2.5 text-center">
                <div className="text-[8px] text-red-400/50 uppercase font-bold tracking-wider">{'>'}10s</div>
                <div className="text-base font-black text-red-400 leading-tight mt-0.5">{aggDwellOver}</div>
              </div>
            </div>
            {topLongest.length > 0 && (
              <div className="space-y-1">
                <div className="text-[8px] text-white/25 uppercase tracking-wider font-bold">Longest Present</div>
                {topLongest.map((v, i) => {
                  const cc = CLASS_COLORS[v.class] || { text: 'text-white/50', bar: 'bg-white/30' };
                  const dwellPct = Math.min(100, Math.round((v.dwell / Math.max(aggDwellMax, 1)) * 100));
                  return (
                    <div key={i} className="bg-black/30 border border-white/5 px-2 py-1.5">
                      <div className="flex justify-between items-center mb-1">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] font-bold text-white/70 font-mono">#{v.tid}</span>
                          <span className={`text-[9px] font-bold uppercase ${cc.text}`}>{v.class}</span>
                        </div>
                        <span className="text-[10px] text-orange-400 font-black">{v.dwell}s</span>
                      </div>
                      <div className="h-1 bg-white/5 rounded overflow-hidden">
                        <motion.div
                          className="h-full rounded bg-gradient-to-r from-amber-500/60 to-orange-500"
                          initial={{ width: 0 }}
                          animate={{ width: `${dwellPct}%` }}
                          transition={{ type: 'spring', stiffness: 50 }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* 2E: Attention Required */}
        {topAttention.length > 0 && (
          <div className="space-y-2 pt-3 border-t border-red-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-3.5 h-3.5 text-red-400" />
                <div className="text-[9px] text-red-400 uppercase tracking-widest font-black">Attention Required</div>
              </div>
              <span className="relative px-2 py-0.5 bg-red-500/20 text-red-400 text-[10px] font-black border border-red-500/30">
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-red-500 rounded-full animate-ping opacity-75" />
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-red-500 rounded-full" />
                {topAttention.length}
              </span>
            </div>
            <div className="space-y-1.5">
              {topAttention.map((v, i) => {
                const cc = CLASS_COLORS[v.class] || { text: 'text-white/60', bar: 'bg-white/30' };
                const impactPct = Math.min(100, Math.round((v.impact / 15) * 100));
                const impactColor = v.impact > 8 ? 'red' : v.impact > 4 ? 'orange' : 'yellow';
                const reasons = (v.reasons || [v.state]);
                return (
                  <div key={i} className={clsx(
                    'bg-black/50 border p-2.5 space-y-2 relative overflow-hidden',
                    i === 0 ? 'border-red-500/30' : 'border-white/10'
                  )}>
                    {/* Severity edge glow */}
                    <div className={`absolute left-0 top-0 bottom-0 w-0.5 bg-${impactColor}-500 shadow-[0_0_8px] shadow-${impactColor}-500/60`} />

                    {/* Header row */}
                    <div className="flex items-center justify-between pl-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-black text-white/90 font-mono">#{v.tid}</span>
                        <span className={`text-[9px] font-bold uppercase px-1.5 py-0.5 bg-black/40 border border-white/10 ${cc.text}`}>{v.class}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Clock className="w-2.5 h-2.5 text-white/20" />
                        <span className="text-[10px] text-white/50 font-mono font-bold">{v.dwell}s</span>
                      </div>
                    </div>

                    {/* Impact meter */}
                    <div className="pl-2">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[8px] text-white/30 uppercase tracking-wider font-bold">Impact</span>
                        <span className={`text-[11px] font-black text-${impactColor}-400`}>{v.impact}</span>
                      </div>
                      <div className="h-1.5 bg-white/5 rounded overflow-hidden">
                        <motion.div
                          className={`h-full rounded bg-gradient-to-r from-${impactColor}-500/50 to-${impactColor}-500 shadow-[0_0_6px] shadow-${impactColor}-500/40`}
                          initial={{ width: 0 }}
                          animate={{ width: `${impactPct}%` }}
                          transition={{ type: 'spring', stiffness: 50 }}
                        />
                      </div>
                    </div>

                    {/* Reason tags */}
                    <div className="flex flex-wrap gap-1 pl-2">
                      {reasons.map((r, ri) => (
                        <span key={ri} className={clsx(
                          'px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider',
                          r === 'stalled' || r === 'abnormal speed' ? 'bg-red-500/15 text-red-400 border border-red-500/20'
                            : r === 'erratic path' ? 'bg-orange-500/15 text-orange-400 border border-orange-500/20'
                            : r === 'high impact' ? 'bg-red-500/10 text-red-300 border border-red-500/15'
                            : r === 'long dwell' ? 'bg-amber-500/10 text-amber-300 border border-amber-500/15'
                            : 'bg-amber-500/15 text-amber-400 border border-amber-500/20'
                        )}>
                          {r}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

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
        {/* Confidence Control */}
        <ConfidenceControl selectedVideos={selectedVideos} accentColor="purple" />

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

/* ============================
   CONGESTION PANEL (HOT REGIONS)
============================ */
const CAUSE_LABELS = {
  lane_blockage: { label: 'Lane Blockage', icon: Ban, color: 'red' },
  vehicle_accumulation: { label: 'Vehicle Buildup', icon: Users, color: 'orange' },
  downstream_spillback: { label: 'Downstream Spillback', icon: ArrowDownRight, color: 'amber' },
  signal_choke: { label: 'Signal Choke', icon: Radio, color: 'yellow' },
};

const SPREAD_CONFIG = {
  expanding: { label: 'Expanding', icon: ArrowUpRight, color: 'text-red-400' },
  localized: { label: 'Localized', icon: Minus, color: 'text-amber-400' },
  clearing: { label: 'Clearing', icon: ArrowDownRight, color: 'text-emerald-400' },
};

/* ============================
   CROWD ANALYTICS PANEL
============================ */
const DENSITY_CLASS_COLORS = {
  sparse: 'teal',
  gathering: 'amber',
  dense: 'orange',
  critical: 'red',
};

const CrowdPanel = ({ selectedVideos = [] }) => {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (res.ok) setMetrics(await res.json());
      } catch (e) {}
    };
    poll();
    const t = setInterval(poll, 1500);
    return () => clearInterval(t);
  }, []);

  const sourceIds = selectedVideos.map(v => v.id);

  // Pick first source with crowd data (or aggregate)
  let crowdData = null;
  let totalPeople = 0;
  let maxRisk = 0;

  if (metrics) {
    for (const src of sourceIds) {
      const m = metrics[src];
      if (!m) continue;
      totalPeople += m.crowd_count || m.detection_count || 0;
      if (m.risk_score !== undefined) {
        maxRisk = Math.max(maxRisk, m.risk_score);
      }
      // Use first source with crowd analytics as primary
      if (!crowdData && m.mode === 'crowd') {
        crowdData = m;
      }
    }
  }

  // Fallback: use legacy metrics if no crowd-specific data
  const d = crowdData || {};
  const crowdCount = d.crowd_count ?? totalPeople;
  const riskScore = d.risk_score ?? maxRisk;
  const crowdDensity = d.crowd_density ?? (d.traffic_density || 0);
  const avgDensity = d.avg_density ?? 0;
  const densityClass = d.density_class ?? 'sparse';
  const crowdTrend = d.crowd_trend ?? 'stable';
  const trendRate = d.crowd_trend_rate ?? 0;
  const opStatus = d.operational_status ?? 'MONITOR';
  const timeToCritical = d.time_to_critical;
  const zones = d.zones ?? [];
  const zoneDist = d.zone_distribution ?? { sparse: 0, gathering: 0, dense: 0, critical: 0 };
  const hotspots = d.hotspots ?? [];
  const anomalies = d.anomalies ?? [];
  const peakCount = d.peak_count ?? 0;
  const peakDensity = d.peak_density ?? 0;
  const compressionTrend = d.compression_trend ?? 'stable';
  const flowSummary = d.flow_summary ?? { total_inflow: 0, total_outflow: 0, net_flow: 0 };

  // Status badge config
  const statusConfig = {
    'MONITOR': { color: 'teal', label: 'MONITOR' },
    'ALERT': { color: 'amber', label: 'ALERT' },
    'IMMEDIATE ACTION': { color: 'red', label: 'IMMEDIATE ACTION' },
  };
  const sc = statusConfig[opStatus] || statusConfig['MONITOR'];

  const riskColor = riskScore < 30 ? 'teal' : riskScore < 60 ? 'amber' : 'red';

  const TrendIcon = crowdTrend === 'increasing' ? TrendingUp : crowdTrend === 'decreasing' ? TrendingDown : Minus;
  const trendColor = crowdTrend === 'increasing' ? 'text-red-400' : crowdTrend === 'decreasing' ? 'text-emerald-400' : 'text-white/40';

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      {/* Header + operational status badge */}
      <div className="px-4 py-3 border-b border-white/10 bg-[#060a12]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Users className="w-4 h-4 text-teal-500/70" />
            <div className="text-xs text-teal-500 font-black uppercase tracking-[0.25em]">Crowd Analytics</div>
          </div>
          <div className={clsx(
            'flex items-center gap-1.5 px-2 py-0.5 border',
            `bg-${sc.color}-500/15 border-${sc.color}-500/30`,
            opStatus === 'IMMEDIATE ACTION' && 'animate-pulse'
          )}>
            <Shield className={`w-3 h-3 text-${sc.color}-400`} />
            <span className={`text-[9px] font-bold text-${sc.color}-400`}>{sc.label}</span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="px-4 py-3 space-y-4">

          {/* Confidence Control */}
          <ConfidenceControl selectedVideos={selectedVideos} accentColor="teal" />
          <OverlayToggles selectedVideos={selectedVideos} accentColor="teal" useCase="crowd" />

          {/* Primary Stats 2x2 Grid */}
          <div className="grid grid-cols-2 gap-2">
            {/* People count + trend */}
            <div className="bg-black/50 border border-teal-500/20 p-3">
              <div className="flex items-center justify-between mb-1">
                <div className="text-2xl font-black text-teal-400 leading-none">{crowdCount}</div>
                <div className={`flex items-center gap-0.5 ${trendColor}`}>
                  <TrendIcon className="w-3.5 h-3.5" />
                  {trendRate !== 0 && <span className="text-[9px] font-bold">{trendRate > 0 ? '+' : ''}{trendRate}/s</span>}
                </div>
              </div>
              <div className="text-[8px] text-teal-400/60 uppercase font-bold tracking-wider">People</div>
            </div>
            {/* Risk score */}
            <div className={`bg-black/50 border border-${riskColor}-500/20 p-3`}>
              <div className={`text-2xl font-black text-${riskColor}-400 leading-none mb-1`}>{riskScore}</div>
              <div className={`text-[8px] text-${riskColor}-400/60 uppercase font-bold tracking-wider`}>Risk Score</div>
            </div>
            {/* Crowd density */}
            <div className="bg-black/50 border border-teal-500/20 p-3">
              <div className="text-2xl font-black text-teal-400 leading-none mb-1">{Math.round(crowdDensity)}%</div>
              <div className="text-[8px] text-teal-400/60 uppercase font-bold tracking-wider">
                Density <span className={`text-${DENSITY_CLASS_COLORS[densityClass] || 'teal'}-400`}>({densityClass})</span>
              </div>
            </div>
            {/* Time to critical */}
            <div className="bg-black/50 border border-teal-500/20 p-3">
              <div className="text-2xl font-black leading-none mb-1">
                {timeToCritical != null
                  ? <span className="text-red-400">{timeToCritical}s</span>
                  : <span className="text-emerald-400">Safe</span>
                }
              </div>
              <div className="text-[8px] text-teal-400/60 uppercase font-bold tracking-wider">Time to Critical</div>
            </div>
          </div>

          {/* Density Meter */}
          <div className="bg-black/30 border border-white/10 p-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[9px] text-white/50 uppercase tracking-wider font-bold">Crowd Density</span>
              <span className={`text-xs font-black text-${DENSITY_CLASS_COLORS[densityClass] || 'teal'}-400`}>{Math.round(crowdDensity)}%</span>
            </div>
            <div className="relative h-2.5 bg-white/5 rounded overflow-hidden">
              {/* Gradient background */}
              <div className="absolute inset-0 bg-gradient-to-r from-teal-500/30 via-amber-500/30 via-orange-500/30 to-red-500/30 rounded" />
              {/* Marker */}
              <motion.div
                className="absolute top-0 bottom-0 w-1 bg-white rounded shadow-[0_0_6px_rgba(255,255,255,0.8)]"
                initial={{ left: '0%' }}
                animate={{ left: `${Math.min(100, crowdDensity)}%` }}
                transition={{ type: 'spring', stiffness: 50 }}
              />
            </div>
            <div className="flex justify-between text-[8px] text-white/30 uppercase">
              <span>Sparse</span>
              <span>Gathering</span>
              <span>Dense</span>
              <span>Critical</span>
            </div>
          </div>

          {/* Trend Indicators (3-col) */}
          <div className="grid grid-cols-3 gap-1.5">
            <div className="bg-black/40 border border-white/10 p-2 text-center">
              <div className={`text-[10px] font-black uppercase ${trendColor}`}>{crowdTrend}</div>
              <div className="text-[7px] text-white/30 uppercase mt-0.5">Count Trend</div>
            </div>
            <div className="bg-black/40 border border-white/10 p-2 text-center">
              <div className={`text-[10px] font-black uppercase ${
                compressionTrend === 'compressing' ? 'text-red-400' : compressionTrend === 'expanding' ? 'text-emerald-400' : 'text-white/40'
              }`}>{compressionTrend}</div>
              <div className="text-[7px] text-white/30 uppercase mt-0.5">Compression</div>
            </div>
            <div className="bg-black/40 border border-white/10 p-2 text-center">
              <div className={`text-[10px] font-black ${
                flowSummary.net_flow > 0 ? 'text-red-400' : flowSummary.net_flow < 0 ? 'text-emerald-400' : 'text-white/40'
              }`}>
                {flowSummary.net_flow > 0 ? '+' : ''}{flowSummary.net_flow}
              </div>
              <div className="text-[7px] text-white/30 uppercase mt-0.5">Net Flow</div>
            </div>
          </div>

          {/* Zone Distribution (4-col) */}
          <div className="space-y-2">
            <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Zone Distribution</div>
            <div className="grid grid-cols-4 gap-1">
              {[
                { label: 'Sparse', count: zoneDist.sparse, color: 'teal' },
                { label: 'Gather', count: zoneDist.gathering, color: 'amber' },
                { label: 'Dense', count: zoneDist.dense, color: 'orange' },
                { label: 'Critical', count: zoneDist.critical, color: 'red' },
              ].map(s => (
                <div key={s.label} className={`bg-${s.color}-500/10 border border-${s.color}-500/20 p-2 text-center`}>
                  <div className={`text-sm font-black text-${s.color}-400 leading-none`}>{s.count}</div>
                  <div className={`text-[7px] text-${s.color}-400/60 uppercase font-bold mt-0.5`}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Zone Grid Heatmap (4x6 CSS grid) */}
          {zones.length > 0 && (
            <div className="space-y-2">
              <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Zone Grid</div>
              <div className="grid grid-cols-6 gap-0.5">
                {zones.map((z) => {
                  const zc = DENSITY_CLASS_COLORS[z.density_class] || 'teal';
                  const isAnomaly = anomalies.some(a => a.type === 'stationary_buildup' && a.description.includes(`(${z.row},${z.col})`));
                  return (
                    <div
                      key={z.index}
                      className={clsx(
                        'aspect-square flex items-center justify-center text-[8px] font-black border',
                        `bg-${zc}-500/20 border-${zc}-500/30 text-${zc}-400`,
                        isAnomaly && 'animate-pulse ring-1 ring-red-500/50'
                      )}
                      title={`Zone ${z.row},${z.col}: ${z.density}% (${z.density_class})`}
                    >
                      {Math.round(z.density)}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Active Hotspots */}
          {hotspots.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[9px] text-white/40 uppercase tracking-widest font-black">Active Hotspots</span>
                <span className="text-[9px] font-bold text-orange-400">{hotspots.length} DETECTED</span>
              </div>
              <div className="space-y-1.5 max-h-36 overflow-y-auto">
                {hotspots.slice(0, 5).map((hs) => {
                  const hsColor = hs.severity === 'HIGH' ? 'red' : hs.severity === 'MODERATE' ? 'orange' : 'amber';
                  return (
                    <div key={hs.id} className={`bg-black/40 border border-${hsColor}-500/20 p-2 relative overflow-hidden`}>
                      <div className={`absolute left-0 top-0 bottom-0 w-0.5 bg-${hsColor}-500`} />
                      <div className="pl-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className={`text-[9px] font-black text-${hsColor}-400 uppercase`}>{hs.severity}</span>
                            <span className="text-[8px] text-white/40">{hs.zone_count} zones</span>
                          </div>
                          <span className={`text-[10px] font-black text-${hsColor}-400`}>{hs.avg_density}%</span>
                        </div>
                        <div className="flex items-center gap-3 mt-1 text-[8px] text-white/30">
                          <span>Area: {hs.size_pct}%</span>
                          <span>Duration: {hs.persistence}s</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Anomaly Alerts */}
          <AnimatePresence>
            {anomalies.length > 0 && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-2"
              >
                <div className="flex items-center gap-2">
                  <Zap className="w-3 h-3 text-red-400" />
                  <span className="text-[9px] text-red-400 uppercase tracking-widest font-black">Anomalies</span>
                </div>
                {anomalies.map((a, i) => {
                  const anomColor = a.severity === 'critical' ? 'red' : a.severity === 'high' ? 'orange' : 'amber';
                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: 10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={`bg-${anomColor}-500/10 border border-${anomColor}-500/30 p-2`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <AlertTriangle className={`w-3 h-3 text-${anomColor}-400`} />
                        <span className={`text-[9px] font-black uppercase text-${anomColor}-400`}>{a.type.replace(/_/g, ' ')}</span>
                      </div>
                      <div className="text-[9px] text-white/50">{a.description}</div>
                    </motion.div>
                  );
                })}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Peak Stats */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-black/30 border border-white/10 p-2.5 text-center">
              <div className="text-[8px] text-white/30 uppercase font-bold tracking-wider">Peak Count</div>
              <div className="text-base font-black text-teal-400 leading-tight mt-0.5">{peakCount}</div>
            </div>
            <div className="bg-black/30 border border-white/10 p-2.5 text-center">
              <div className="text-[8px] text-white/30 uppercase font-bold tracking-wider">Peak Density</div>
              <div className="text-base font-black text-teal-400 leading-tight mt-0.5">{peakDensity}%</div>
            </div>
          </div>

          {/* Per-source breakdown */}
          {sourceIds.length > 1 && (
            <div className="space-y-2 pt-2 border-t border-white/5">
              <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Per Source</div>
              {sourceIds.map(src => {
                const m = metrics?.[src];
                const cnt = m?.crowd_count || m?.detection_count || 0;
                const droneMatch = src.match(/bcpdrone(\d+)/i);
                const label = droneMatch ? `DRONE ${droneMatch[1]}` : src;
                return (
                  <div key={src} className="flex justify-between items-center">
                    <span className="text-[10px] text-white/50 font-mono uppercase">{label}</span>
                    <span className="text-[10px] font-bold text-teal-400">{cnt} people</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const ACTION_CONFIG = {
  clear_stalled_vehicle: 'Clear stalled vehicle',
  deploy_personnel: 'Deploy traffic personnel',
  redirect_upstream: 'Redirect upstream traffic',
  open_close_lanes: 'Open/close lanes',
  monitor: 'Monitor only',
  no_action: 'No action needed',
};

const PRIORITY_CONFIG = {
  immediate: { label: 'IMMEDIATE', color: 'red', bg: 'bg-red-500/20', border: 'border-red-500/40', text: 'text-red-400' },
  monitor: { label: 'MONITOR', color: 'amber', bg: 'bg-amber-500/15', border: 'border-amber-500/30', text: 'text-amber-400' },
  no_action: { label: 'NO ACTION', color: 'white', bg: 'bg-white/5', border: 'border-white/10', text: 'text-white/40' },
};

const CongestionPanel = ({ selectedVideos = [], alerts = [], onSelectAlert }) => {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (res.ok) setMetrics(await res.json());
      } catch (e) {}
    };
    poll();
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, []);

  const sourceIds = selectedVideos.map(v => v.id);

  // Aggregate hot regions across sources
  let allRegions = [];
  let sevCounts = { HIGH: 0, MODERATE: 0, LOW: 0 };
  let totalActive = 0;

  if (metrics) {
    for (const src of sourceIds) {
      const m = metrics[src];
      if (!m?.hot_regions) continue;
      const hr = m.hot_regions;
      totalActive += hr.active_count || 0;
      if (hr.severity_counts) {
        sevCounts.HIGH += hr.severity_counts.HIGH || 0;
        sevCounts.MODERATE += hr.severity_counts.MODERATE || 0;
        sevCounts.LOW += hr.severity_counts.LOW || 0;
      }
      if (hr.regions) {
        allRegions = allRegions.concat(hr.regions.map(r => ({ ...r, source: src })));
      }
    }
  }

  // Sort: HIGH first, then MODERATE, then LOW
  const sevOrder = { HIGH: 0, MODERATE: 1, LOW: 2 };
  allRegions.sort((a, b) => (sevOrder[a.severity] || 3) - (sevOrder[b.severity] || 3));
  const topRegions = allRegions.slice(0, 8);

  const hasRegions = totalActive > 0;

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/10 bg-[#060a12]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Flame className="w-4 h-4 text-red-500/70" />
            <div className="text-xs text-red-500 font-black uppercase tracking-[0.25em]">Congestion</div>
          </div>
          <div className="flex items-center gap-2">
            {sevCounts.HIGH > 0 && (
              <span className="relative px-1.5 py-0.5 bg-red-500/20 text-red-400 text-[9px] font-bold border border-red-500/30">
                <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-red-500 rounded-full animate-ping opacity-75" />
                <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-red-500 rounded-full" />
                {sevCounts.HIGH} HIGH
              </span>
            )}
            <span className="text-[10px] font-bold text-orange-400 bg-orange-500/15 px-2 py-0.5 border border-orange-500/20">
              {totalActive} HOT
            </span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="px-4 py-3 space-y-4">

          {/* Confidence Control */}
          <ConfidenceControl selectedVideos={selectedVideos} accentColor="cyan" />
          <OverlayToggles selectedVideos={selectedVideos} accentColor="red" />

          {/* Severity summary cards */}
          <div className="grid grid-cols-3 gap-1.5">
            {[
              { label: 'High', count: sevCounts.HIGH, color: 'red' },
              { label: 'Moderate', count: sevCounts.MODERATE, color: 'orange' },
              { label: 'Low', count: sevCounts.LOW, color: 'amber' },
            ].map(s => (
              <div key={s.label} className={`bg-black/50 border border-${s.color}-500/20 p-2.5 text-center relative overflow-hidden`}>
                <motion.div
                  className={`absolute bottom-0 left-0 right-0 bg-${s.color}-500/10`}
                  initial={{ height: 0 }}
                  animate={{ height: `${Math.min(100, s.count * 33)}%` }}
                  transition={{ type: 'spring', stiffness: 40, damping: 15 }}
                />
                <div className="relative">
                  <div className={`text-xl font-black text-${s.color}-400 leading-none`}>{s.count}</div>
                  <div className={`text-[8px] text-${s.color}-400/60 uppercase font-bold mt-1 tracking-wider`}>{s.label}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Hot Regions List */}
          {hasRegions ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <MapPin className="w-3 h-3 text-orange-400/60" />
                <div className="text-[9px] text-white/40 uppercase tracking-widest font-black">Active Hot Regions</div>
              </div>

              {topRegions.map((region, i) => {
                const sevColor = region.severity === 'HIGH' ? 'red' : region.severity === 'MODERATE' ? 'orange' : 'amber';
                const causeInfo = CAUSE_LABELS[region.cause] || CAUSE_LABELS.vehicle_accumulation;
                const CauseIcon = causeInfo.icon;
                const spreadInfo = SPREAD_CONFIG[region.spread] || SPREAD_CONFIG.localized;
                const SpreadIcon = spreadInfo.icon;
                const priorityInfo = PRIORITY_CONFIG[region.action_priority] || PRIORITY_CONFIG.no_action;
                const actionLabel = ACTION_CONFIG[region.suggested_action] || 'Monitor';

                return (
                  <div key={i} className={clsx(
                    'bg-black/50 border relative overflow-hidden',
                    region.severity === 'HIGH' ? 'border-red-500/30' : region.severity === 'MODERATE' ? 'border-orange-500/20' : 'border-white/10'
                  )}>
                    {/* Severity edge */}
                    <div className={`absolute left-0 top-0 bottom-0 w-1 bg-${sevColor}-500 shadow-[0_0_8px] shadow-${sevColor}-500/50`} />

                    <div className="pl-3.5 pr-2.5 py-2.5 space-y-2.5">
                      {/* Top row: severity + persistence + spread */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className={`text-[10px] font-black text-${sevColor}-400 uppercase tracking-wider`}>
                            {region.severity}
                          </span>
                          <span className={clsx(
                            'px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider',
                            region.persistence === 'recurring' ? 'bg-red-500/15 text-red-300 border border-red-500/20'
                              : region.persistence === 'ongoing' ? 'bg-orange-500/15 text-orange-300 border border-orange-500/20'
                              : 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/20'
                          )}>
                            {region.persistence}
                          </span>
                          {region.recurring && (
                            <span className="px-1 py-0.5 text-[7px] font-bold uppercase bg-purple-500/15 text-purple-300 border border-purple-500/20">
                              RECURRING
                            </span>
                          )}
                        </div>
                        <div className={clsx('flex items-center gap-1', spreadInfo.color)}>
                          <SpreadIcon className="w-3 h-3" />
                          <span className="text-[9px] font-bold">{spreadInfo.label}</span>
                        </div>
                      </div>

                      {/* Congestion meter */}
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[8px] text-white/30 uppercase tracking-wider font-bold">Congestion</span>
                          <span className={`text-[10px] font-black text-${sevColor}-400`}>{region.avg_congestion}%</span>
                        </div>
                        <div className="h-2 bg-white/5 rounded overflow-hidden">
                          <motion.div
                            className={clsx('h-full rounded',
                              region.avg_congestion > 80 ? 'bg-gradient-to-r from-red-600 to-red-400 shadow-[0_0_8px_rgba(239,68,68,0.5)]'
                                : region.avg_congestion > 60 ? 'bg-gradient-to-r from-orange-600 to-orange-400'
                                : 'bg-gradient-to-r from-amber-600 to-amber-400'
                            )}
                            initial={{ width: 0 }}
                            animate={{ width: `${region.avg_congestion}%` }}
                            transition={{ type: 'spring', stiffness: 50 }}
                          />
                        </div>
                      </div>

                      {/* Stats row: size + duration + vehicles */}
                      <div className="grid grid-cols-3 gap-1.5">
                        <div className="bg-black/40 border border-white/5 px-2 py-1 text-center">
                          <div className="text-[7px] text-white/25 uppercase font-bold">Area</div>
                          <div className="text-[10px] font-black text-white/70">{region.size_pct}%</div>
                        </div>
                        <div className="bg-black/40 border border-white/5 px-2 py-1 text-center">
                          <div className="text-[7px] text-white/25 uppercase font-bold">Duration</div>
                          <div className="text-[10px] font-black text-white/70">{region.duration}s</div>
                        </div>
                        <div className="bg-black/40 border border-white/5 px-2 py-1 text-center">
                          <div className="text-[7px] text-white/25 uppercase font-bold">Vehicles</div>
                          <div className="text-[10px] font-black text-white/70">
                            <span className="text-red-400">{region.stalled}</span>
                            <span className="text-white/20">/</span>
                            <span className="text-orange-400">{region.slow}</span>
                            <span className="text-white/20">/</span>
                            {region.total_vehicles}
                          </div>
                        </div>
                      </div>

                      {/* Cause */}
                      <div className="flex items-center gap-2">
                        <CauseIcon className={`w-3 h-3 text-${causeInfo.color}-400/60`} />
                        <span className="text-[9px] text-white/50 font-bold">{causeInfo.label}</span>
                        <div className="flex-1" />
                        <span className={clsx(
                          'text-[8px] font-bold uppercase px-1.5 py-0.5',
                          region.impact === 'blocking_junctions' ? 'text-red-300 bg-red-500/10 border border-red-500/15'
                            : region.impact === 'affecting_service_roads' ? 'text-orange-300 bg-orange-500/10 border border-orange-500/15'
                            : 'text-amber-300 bg-amber-500/10 border border-amber-500/15'
                        )}>
                          {region.impact.replace(/_/g, ' ')}
                        </span>
                      </div>

                      {/* Action */}
                      <div className="flex items-center justify-between bg-black/30 border border-white/5 px-2 py-1.5">
                        <div className="flex items-center gap-1.5">
                          <Siren className={`w-3 h-3 ${priorityInfo.text}`} />
                          <span className={`text-[8px] font-black uppercase tracking-wider ${priorityInfo.text}`}>
                            {priorityInfo.label}
                          </span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Wrench className="w-2.5 h-2.5 text-white/20" />
                          <span className="text-[8px] text-white/40 font-bold">{actionLabel}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="flex items-center justify-center py-8 border border-dashed border-white/5 bg-black/20">
              <div className="text-center">
                <Flame className="w-5 h-5 text-white/10 mx-auto mb-2" />
                <span className="text-[10px] text-white/20 font-mono uppercase block">No active hot regions</span>
                <span className="text-[9px] text-white/10 font-mono block mt-1">Monitoring road segments</span>
              </div>
            </div>
          )}
        </div>

        {/* Alerts section below hot regions */}
        <AlertsSection alerts={alerts} onSelectAlert={onSelectAlert} />
      </div>
    </div>
  );
};

/* ============================
   ALERTS SECTION (reusable)
============================ */
const AlertsSection = ({ alerts, onSelectAlert, fullHeight = false }) => (
  <div className={clsx('flex flex-col border-t border-white/10', fullHeight ? 'flex-1' : '')} style={fullHeight ? { flex: '1 1 0%', minHeight: 0 } : { flex: '1 1 0%', minHeight: 0 }}>
    <div className="px-3 py-2 border-b border-white/5 bg-[#060a12] flex items-center justify-between">
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
    <div className="flex-1 overflow-y-auto min-h-0">
      {alerts.length === 0 ? (
        <div className="p-4 text-center">
          <div className="text-white/15 text-[10px] font-mono">No active alerts</div>
        </div>
      ) : (
        alerts.map((alert) => {
          const config = severityConfig[alert.severity] || severityConfig.medium;
          const Icon = config.icon;
          const droneMatch = alert.source.match(/bcpdrone(\d+)/i);
          const droneLabel = droneMatch ? `DRONE ${droneMatch[1]}` : alert.source.toUpperCase();

          return (
            <div
              key={alert.id}
              onClick={() => onSelectAlert(alert)}
              className={clsx(
                'flex border-b border-white/5 hover:bg-white/[0.03] cursor-pointer transition-colors',
                config.bgColor
              )}
            >
              <div className={clsx('w-0.5', config.color.replace('text-', 'bg-'))} />
              <div className="px-3 py-1.5 flex gap-2 flex-1 font-mono items-center">
                <Icon className={clsx('w-3.5 h-3.5 shrink-0', config.color)} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-bold text-white truncate">{droneLabel}</span>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className={clsx('text-[8px] uppercase font-bold', config.color)}>{alert.congestion}%</span>
                    <span className="text-[8px] text-white/30">{formatTimeAgo(alert.timestamp)}</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })
      )}
    </div>
  </div>
);

const RightPanel = ({ useCase = 'congestion', selectedVideos = [] }) => {
  const [alerts, setAlerts] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState(null);

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
          setAlerts(data.alerts || []);
        }
      } catch (e) {
        console.error('Failed to fetch alerts:', e);
      }
    };
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 3000);
    return () => clearInterval(interval);
  }, [useCase]);

  const panelType = useCase === 'vehicle' ? 'vehicle'
    : useCase === 'flow' ? 'flow'
    : useCase === 'forensics' ? 'forensics'
    : useCase === 'crowd' ? 'crowd'
    : 'alerts';

  return (
    <>
      <motion.div
        initial={{ x: 40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        className="w-[320px] h-full bg-[#070b14]/80 border-l border-white/10 flex flex-col z-40"
      >
        {panelType === 'alerts' && (
          <CongestionPanel selectedVideos={selectedVideos} alerts={alerts} onSelectAlert={setSelectedAlert} />
        )}

        {panelType === 'vehicle' && (
          <VehicleStatsPanel selectedVideos={selectedVideos} />
        )}

        {panelType === 'flow' && (
          <FlowStatsPanel selectedVideos={selectedVideos} />
        )}

        {panelType === 'forensics' && (
          <ForensicsPanel selectedVideos={selectedVideos} />
        )}

        {panelType === 'crowd' && (
          <CrowdPanel selectedVideos={selectedVideos} />
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
