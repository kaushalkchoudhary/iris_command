import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, AlertCircle, XCircle, X, Clock, Activity, Users, Gauge, Search, Square, Play, Loader2, Crosshair, ScanSearch, BarChart3 } from 'lucide-react';
import clsx from 'clsx';

const API_BASE_URL = import.meta.env.DEV
  ? '/api'
  : `http://${window.location.hostname}:9010`;

/* ============================
   DRONE → LOCATION MAP
============================ */
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
const ForensicsPanel = ({ selectedVideos = [], onExpandResult }) => {
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
  const updateTimerRef = useRef(null);

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

  const sendUpdate = (updates) => {
    if (!isRunning || !source) return;
    if (updateTimerRef.current) clearTimeout(updateTimerRef.current);
    updateTimerRef.current = setTimeout(async () => {
      try {
        await fetch(`${API_BASE_URL}/sam/update`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source, ...updates }),
        });
      } catch (e) {
        console.error('SAM update failed:', e);
      }
    }, 150);
  };

  const handleConfidenceChange = (val) => {
    setConfidence(val);
    sendUpdate({ confidence: val });
  };

  const handleToggleBoxes = () => {
    const next = !showBoxes;
    setShowBoxes(next);
    sendUpdate({ show_boxes: next });
  };

  const handleToggleMasks = () => {
    const next = !showMasks;
    setShowMasks(next);
    sendUpdate({ show_masks: next });
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

        <div className="flex gap-2 pb-1">
          <button
            onClick={handleToggleBoxes}
            className={clsx(
              'flex-1 px-3 py-1.5 text-[9px] font-black uppercase tracking-widest border transition-all duration-200',
              showBoxes
                ? 'bg-cyan-500 text-black border-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.4)]'
                : 'bg-black/60 text-white/40 border-white/10 hover:border-cyan-500/40 hover:text-white/60'
            )}
          >
            BBOX
          </button>
          <button
            onClick={handleToggleMasks}
            className={clsx(
              'flex-1 px-3 py-1.5 text-[9px] font-black uppercase tracking-widest border transition-all duration-200',
              showMasks
                ? 'bg-cyan-500 text-black border-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.4)]'
                : 'bg-black/60 text-white/40 border-white/10 hover:border-cyan-500/40 hover:text-white/60'
            )}
          >
            SEGMENT
          </button>
        </div>
      </div>

      {/* Result Area - scrollable, clickable to expand */}
      <div className="flex-1 overflow-y-auto min-h-0 px-4 py-3">
        {isLoading && !result && (
          <div className="flex items-center justify-center h-36 border border-white/5 bg-black/30">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-6 h-6 text-cyan-500/60 animate-spin" />
              <span className="text-[10px] text-white/30 font-mono uppercase tracking-wider">Processing frame...</span>
            </div>
          </div>
        )}

        {result && result.annotated_frame && (
          <div
            className="space-y-2 cursor-pointer group"
            onClick={() => onExpandResult && onExpandResult({ ...result, source, prompt: result.prompt || prompt })}
          >
            {/* Annotated Image */}
            <div className="relative">
              <img
                src={`data:image/jpeg;base64,${result.annotated_frame}`}
                alt="IRIS Forensics Detection"
                className="w-full h-auto border border-cyan-500/20 object-contain group-hover:border-cyan-500/40 transition-colors"
              />
              <div className="absolute top-1 right-1 flex gap-1">
                <span className="px-1.5 py-0.5 bg-black/80 border border-cyan-500/30 text-[8px] text-cyan-400 font-mono font-bold">
                  LIVE
                </span>
              </div>
              <div className="absolute bottom-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="px-1.5 py-0.5 bg-cyan-500/90 text-[8px] text-black font-mono font-bold">
                  CLICK TO EXPAND
                </span>
              </div>
            </div>

            {/* Compact Output */}
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

const RightPanel = ({ useCase = 'traffic', selectedVideos = [] }) => {
  const [alerts, setAlerts] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [expandedForensic, setExpandedForensic] = useState(null);

  // Fetch alerts from backend (including stored)
  useEffect(() => {
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
  }, []);

  return (
    <>
      {/* ================= PANEL ================= */}
      <motion.div
        initial={{ x: 40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        className="w-[320px] h-full bg-[#070b14]/80 border-l border-white/10 flex flex-col z-40"
      >
        {/* ===== IRIS FORENSICS SECTION (TOP) ===== */}
        <div className="flex flex-col" style={{ flex: '1.2 1 0%', minHeight: 0 }}>
          <ForensicsPanel
            selectedVideos={selectedVideos}
            onExpandResult={setExpandedForensic}
          />
        </div>

        {/* ===== ALERTS SECTION (BOTTOM - scrollable) ===== */}
        <div className="flex flex-col border-t border-white/10" style={{ flex: '1 1 0%', minHeight: 0 }}>
          {/* Alerts Header */}
          <div className="px-3 py-2 border-b border-white/5 bg-[#060a12] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-3 h-3 text-red-400/60" />
              <div className="text-[9px] text-red-400/60 font-black uppercase tracking-[0.3em]">
                Alerts
              </div>
              {alerts.length > 0 && (
                <span className="px-1.5 py-0.5 bg-red-500/15 text-red-400 text-[9px] font-bold border border-red-500/20">
                  {alerts.length}
                </span>
              )}
            </div>
          </div>

          {/* Alert List - scrollable */}
          <div className="flex-1 overflow-y-auto min-h-0">
            {alerts.length === 0 ? (
              <div className="p-4 text-center">
                <div className="text-white/15 text-[10px] font-mono">No active alerts</div>
              </div>
            ) : (
              alerts.map((alert) => {
                const config = severityConfig[alert.severity] || severityConfig.medium;
                const Icon = config.icon;
                const location = DRONE_REGION_MAP[alert.source] || alert.source;
                const droneMatch = alert.source.match(/bcpdrone(\d+)/i);
                const droneLabel = droneMatch ? `D${droneMatch[1]}` : alert.source.toUpperCase();

                return (
                  <div
                    key={alert.id}
                    onClick={() => setSelectedAlert(alert)}
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
                          <span className="text-[10px] font-bold text-white truncate">
                            {location}
                          </span>
                          <span className="text-[8px] font-bold text-cyan-500/60 bg-cyan-500/10 px-1 py-0.5 shrink-0">
                            {droneLabel}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span className={clsx('text-[8px] uppercase font-bold', config.color)}>
                            {alert.congestion}%
                          </span>
                          <span className="text-[8px] text-white/30">
                            {formatTimeAgo(alert.timestamp)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
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
                              {DRONE_REGION_MAP[selectedAlert.source] || selectedAlert.source}
                            </span>
                            <span className="text-xs font-bold text-cyan-400 bg-cyan-500/20 px-2 py-0.5">
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

      {/* ================= FORENSICS DETAIL MODAL ================= */}
      <AnimatePresence>
        {expandedForensic && (
          <motion.div
            className="fixed inset-0 bg-black/90 z-[100] flex items-center justify-center p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setExpandedForensic(null)}
          >
            <motion.div
              onClick={(e) => e.stopPropagation()}
              className="bg-[#0a1120] max-w-5xl w-full border border-cyan-500/20 relative overflow-hidden"
              initial={{ scale: 0.96 }}
              animate={{ scale: 1 }}
            >
              <button
                onClick={() => setExpandedForensic(null)}
                className="absolute top-4 right-4 z-10 border border-white/10 p-2 bg-black/50 hover:bg-white/10 transition-colors"
              >
                <X className="w-4 h-4 text-white/70" />
              </button>

              {/* Header */}
              <div className="p-4 border-b border-cyan-500/20 bg-black/30">
                <div className="flex items-center gap-3">
                  <Crosshair className="w-6 h-6 text-cyan-500" />
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg font-black text-white uppercase tracking-wide">
                        IRIS Forensics
                      </span>
                      <span className="text-xs font-bold text-cyan-400 bg-cyan-500/20 px-2 py-0.5">
                        {expandedForensic.source?.toUpperCase()}
                      </span>
                      <span className="text-xs font-bold text-emerald-400 bg-emerald-500/15 px-2 py-0.5 border border-emerald-500/20">
                        {expandedForensic.count} DETECTED
                      </span>
                    </div>
                    <div className="text-sm font-bold text-cyan-400/70">
                      Prompt: "{expandedForensic.prompt}" &bull; Confidence: {((expandedForensic.confidence || 0.7) * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Image + Metrics side by side */}
              <div className="flex flex-col md:flex-row">
                {/* Image — uses object-cover to fill its box, no black bars */}
                <div className="flex-1 relative bg-[#050810] min-h-0 flex items-center justify-center overflow-hidden">
                  <img
                    src={`data:image/jpeg;base64,${expandedForensic.annotated_frame}`}
                    alt="Forensic Detection"
                    className="w-full h-full object-cover"
                    style={{ maxHeight: '60vh' }}
                  />
                  <div className="absolute top-2 left-2 px-2 py-1 bg-black/70 border border-cyan-500/30 text-xs text-cyan-400 font-mono font-bold">
                    LIVE ANALYSIS
                  </div>
                  <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/70 text-xs text-white/60 font-mono">
                    {new Date(expandedForensic.timestamp * 1000).toLocaleTimeString()}
                  </div>
                </div>

                {/* Metrics Panel — scrollable right side */}
                <div className="w-full md:w-80 bg-black/20 border-t md:border-t-0 md:border-l border-cyan-500/10 overflow-y-auto" style={{ maxHeight: '60vh' }}>
                  {/* Stats cards */}
                  <div className="p-4 space-y-3">
                    <div className="flex items-center gap-2 mb-1">
                      <BarChart3 className="w-4 h-4 text-cyan-500/60" />
                      <div className="text-xs text-cyan-500/60 font-black uppercase tracking-widest">
                        Detection Analytics
                      </div>
                    </div>

                    {/* 2x2 stat grid */}
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-black/40 border border-white/5 p-3 rounded">
                        <div className="flex items-center gap-1.5 mb-1.5">
                          <ScanSearch className="w-3.5 h-3.5 text-cyan-400" />
                          <span className="text-[9px] text-white/40 uppercase tracking-wide">Prompt</span>
                        </div>
                        <div className="text-sm font-bold text-cyan-400 break-words leading-tight">
                          "{expandedForensic.prompt}"
                        </div>
                      </div>

                      <div className="bg-black/40 border border-white/5 p-3 rounded">
                        <div className="flex items-center gap-1.5 mb-1.5">
                          <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-[9px] text-white/40 uppercase tracking-wide">Found</span>
                        </div>
                        <div className="text-3xl font-black text-emerald-400 leading-none">
                          {expandedForensic.count}
                        </div>
                      </div>

                      <div className="bg-black/40 border border-white/5 p-3 rounded">
                        <div className="flex items-center gap-1.5 mb-1.5">
                          <Gauge className="w-3.5 h-3.5 text-yellow-400" />
                          <span className="text-[9px] text-white/40 uppercase tracking-wide">Confidence</span>
                        </div>
                        <div className="text-3xl font-black text-yellow-400 leading-none">
                          {((expandedForensic.confidence || 0.7) * 100).toFixed(0)}%
                        </div>
                      </div>

                      <div className="bg-black/40 border border-white/5 p-3 rounded">
                        <div className="flex items-center gap-1.5 mb-1.5">
                          <Activity className="w-3.5 h-3.5 text-white/50" />
                          <span className="text-[9px] text-white/40 uppercase tracking-wide">Source</span>
                        </div>
                        <div className="text-sm font-bold text-white/80 leading-tight">
                          {DRONE_REGION_MAP[expandedForensic.source] || expandedForensic.source}
                        </div>
                      </div>
                    </div>

                    {/* Render config */}
                    <div className="flex gap-2 pt-1">
                      <span className={clsx(
                        'px-3 py-1.5 text-[9px] font-bold uppercase border rounded',
                        expandedForensic.show_boxes ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30' : 'bg-white/5 text-white/20 border-white/10'
                      )}>
                        BBOX {expandedForensic.show_boxes ? 'ON' : 'OFF'}
                      </span>
                      <span className={clsx(
                        'px-3 py-1.5 text-[9px] font-bold uppercase border rounded',
                        expandedForensic.show_masks ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30' : 'bg-white/5 text-white/20 border-white/10'
                      )}>
                        MASK {expandedForensic.show_masks ? 'ON' : 'OFF'}
                      </span>
                    </div>

                    {/* Detection scores */}
                    {expandedForensic.detections && expandedForensic.detections.length > 0 && (
                      <div className="pt-3 border-t border-white/10">
                        <div className="text-[10px] text-white/40 uppercase mb-3 tracking-wide">Detection Scores</div>
                        <div className="space-y-3">
                          {expandedForensic.detections.map((det, i) => (
                            <div key={i}>
                              <div className="flex justify-between text-xs mb-1.5">
                                <span className="text-white/50 font-mono">Object #{i + 1}</span>
                                <span className="text-cyan-400 font-bold text-sm">{(det.score * 100).toFixed(1)}%</span>
                              </div>
                              <div className="h-2.5 bg-white/5 rounded overflow-hidden">
                                <div
                                  className="h-full rounded transition-all duration-500"
                                  style={{
                                    width: `${det.score * 100}%`,
                                    background: det.score > 0.8 ? '#10b981' : det.score > 0.5 ? '#06b6d4' : '#eab308',
                                  }}
                                />
                              </div>
                              {det.box && (
                                <div className="text-[9px] text-white/20 font-mono mt-1">
                                  bbox: [{det.box.join(', ')}]
                                </div>
                              )}
                            </div>
                          ))}
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
