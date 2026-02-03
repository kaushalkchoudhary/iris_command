import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrowserRouter, Routes, Route, Navigate, useParams, useNavigate, useLocation } from 'react-router-dom';

import Header from './components/Dashboard/Header';
import RightPanel from './components/Dashboard/RightPanel';
import Footer from './components/Dashboard/Footer';
import IRISLoader from './components/Dashboard/IRISLoader';
import WelcomeScreen from './components/Dashboard/WelcomeScreen';
import Login from './Login';
import { API_BASE_URL, WEBRTC_BASE, HLS_BASE } from './config';

// ── MODE CONFIGURATION ──
// Overlays are hardcoded in the backend per mode. Frontend only knows theme.
const MODE_CONFIG = {
  congestion: {
    theme: { glow: 'rgba(6,182,212,0.15)', grid: 'rgba(0, 255, 255, 0.3)', accent: 'cyan' },
  },
  vehicle: {
    theme: { glow: 'rgba(16,185,129,0.15)', grid: 'rgba(16, 185, 129, 0.3)', accent: 'emerald' },
  },
  flow: {
    theme: { glow: 'rgba(168,85,247,0.15)', grid: 'rgba(168, 85, 247, 0.3)', accent: 'purple' },
  },
  forensics: {
    theme: { glow: 'rgba(245,158,11,0.15)', grid: 'rgba(245, 158, 11, 0.3)', accent: 'amber' },
  },
  crowd: {
    theme: { glow: 'rgba(20,184,166,0.15)', grid: 'rgba(20, 184, 166, 0.3)', accent: 'teal' },
  },
  safety: {
    theme: { glow: 'rgba(16,185,129,0.15)', grid: 'rgba(16, 185, 129, 0.3)', accent: 'emerald' },
  },
};

export { MODE_CONFIG };

// Stop all processing on the backend
const stopAllProcessing = async () => {
  try {
    await fetch(`${API_BASE_URL}/sources/stop_all`, { method: 'POST' });
  } catch (e) {
    console.error('Failed to stop all processing:', e);
  }
};

// Start processing a drone on the backend — backend handles overlay config per mode
const startDroneProcessing = async (droneIndex, useCase) => {
  try {
    await fetch(`${API_BASE_URL}/sources/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index: droneIndex, mode: useCase }),
    });
  } catch (e) {
    console.error('Failed to start drone processing:', e);
  }
};

// Stop processing a drone on the backend
const stopDroneProcessing = async (droneIndex) => {
  try {
    await fetch(`${API_BASE_URL}/sources/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index: droneIndex }),
    });
  } catch (e) {
    console.error('Failed to stop drone processing:', e);
  }
};

// ── Mode-specific metrics overlay — left & right panels on each video cell ──
/* Metric card with double corner accents */
const accentColors = {
  cyan: '#06b6d4', emerald: '#10b981', purple: '#a855f7',
  red: '#ef4444', amber: '#f59e0b', yellow: '#eab308',
};
const MetricCard = ({ borderColor = 'cyan', children, className = '' }) => {
  const accent = accentColors[borderColor] || accentColors.cyan;
  return (
    <div className={`relative bg-black/70 backdrop-blur-md px-3 py-2 text-center min-w-[72px] ${className}`}
         style={{ boxShadow: `0 0 10px ${accent}15, inset 0 0 20px rgba(0,0,0,0.3)` }}>
      {/* Outer corner accents */}
      <div className="absolute top-0 left-0 w-3 h-3 border-t border-l pointer-events-none" style={{ borderColor: accent, opacity: 0.7 }} />
      <div className="absolute top-0 right-0 w-3 h-3 border-t border-r pointer-events-none" style={{ borderColor: accent, opacity: 0.7 }} />
      <div className="absolute bottom-0 left-0 w-3 h-3 border-b border-l pointer-events-none" style={{ borderColor: accent, opacity: 0.7 }} />
      <div className="absolute bottom-0 right-0 w-3 h-3 border-b border-r pointer-events-none" style={{ borderColor: accent, opacity: 0.7 }} />
      {/* Top edge glow line */}
      <div className="absolute top-0 left-3 right-3 h-px pointer-events-none" style={{ background: `linear-gradient(90deg, transparent, ${accent}40, transparent)` }} />
      {children}
    </div>
  );
};

const MetricsHUD = ({ metrics, useCase, samResult }) => {
  if (useCase === 'forensics') {
    if (!samResult || !samResult.detections) return null;
    const detections = samResult.detections;
    const count = samResult.count ?? detections.length;
    const avgConf = detections.length > 0
      ? (detections.reduce((sum, d) => sum + (d.score || 0), 0) / detections.length * 100).toFixed(0)
      : 0;
    // Coverage: sum of bounding box areas / frame area (assume normalized 0-1 coords or use pixel ratios)
    let coverage = 0;
    if (detections.length > 0 && detections[0].box) {
      // box format: [x1, y1, x2, y2] — compute fraction of frame covered
      const frameW = samResult.frame_width || 1920;
      const frameH = samResult.frame_height || 1080;
      const frameArea = frameW * frameH;
      let totalBoxArea = 0;
      for (const d of detections) {
        const [x1, y1, x2, y2] = d.box;
        totalBoxArea += Math.abs(x2 - x1) * Math.abs(y2 - y1);
      }
      coverage = Math.min(100, (totalBoxArea / frameArea * 100)).toFixed(1);
    }
    const promptText = samResult.prompt || '';
    const truncatedPrompt = promptText.length > 14 ? promptText.slice(0, 14) + '…' : promptText;
    return (
      <>
        {/* Left — Detection Summary */}
        <div className="absolute left-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none flex flex-col gap-2">
          <MetricCard borderColor="amber">
            <div className="text-[9px] font-bold uppercase tracking-widest text-amber-500/60 mb-0.5">DETECTIONS</div>
            <div className="text-2xl font-black tabular-nums leading-none text-amber-300">{count}</div>
          </MetricCard>
          <MetricCard borderColor="amber">
            <div className="text-[9px] font-bold uppercase tracking-widest text-amber-500/60 mb-0.5">COVERAGE</div>
            <div className="text-xl font-black tabular-nums leading-none text-amber-300">{coverage}%</div>
          </MetricCard>
        </div>
        {/* Right — Confidence Breakdown */}
        <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none flex flex-col gap-2">
          <MetricCard borderColor="amber">
            <div className="text-[9px] font-bold uppercase tracking-widest text-amber-500/60 mb-0.5">AVG CONF</div>
            <div className="text-2xl font-black tabular-nums leading-none text-amber-300">{avgConf}%</div>
          </MetricCard>
          <MetricCard borderColor="amber">
            <div className="text-[9px] font-bold uppercase tracking-widest text-amber-500/60 mb-0.5">PROMPT</div>
            <div className="text-sm font-black leading-none text-amber-300 truncate max-w-[80px]">{truncatedPrompt || '—'}</div>
          </MetricCard>
        </div>
      </>
    );
  }

  if (!metrics) return null;

  const m = metrics;

  if (useCase === 'congestion') {
    const congestion = m.congestion_index ?? 0;
    const density = m.traffic_density ?? 0;
    const mobility = m.mobility_index ?? 0;
    const count = m.detection_count ?? 0;
    const congColor = congestion >= 60 ? 'text-red-400' : congestion >= 35 ? 'text-orange-400' : 'text-green-300';
    const mobColor = mobility >= 70 ? 'text-green-300' : mobility >= 40 ? 'text-yellow-300' : 'text-red-300';
    return (
      <>
        {/* Left panel */}
        <div className="absolute left-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none flex flex-col gap-2">
          <MetricCard borderColor="cyan">
            <div className="text-[9px] font-bold uppercase tracking-widest text-cyan-400/70 mb-0.5">CONGESTION</div>
            <div className={`text-2xl font-black tabular-nums leading-none ${congColor}`}>{congestion}%</div>
          </MetricCard>
          <MetricCard borderColor="cyan">
            <div className="text-[9px] font-bold uppercase tracking-widest text-cyan-400/70 mb-0.5">DENSITY</div>
            <div className="text-xl font-black tabular-nums leading-none text-cyan-200">{density}%</div>
          </MetricCard>
        </div>
        {/* Right panel */}
        <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none flex flex-col gap-2">
          <MetricCard borderColor="cyan">
            <div className="text-[9px] font-bold uppercase tracking-widest text-cyan-400/70 mb-0.5">MOBILITY</div>
            <div className={`text-2xl font-black tabular-nums leading-none ${mobColor}`}>{mobility}</div>
          </MetricCard>
          <MetricCard borderColor="cyan">
            <div className="text-[9px] font-bold uppercase tracking-widest text-cyan-400/70 mb-0.5">VEHICLES</div>
            <div className="text-xl font-black tabular-nums leading-none text-cyan-200">{count}</div>
          </MetricCard>
        </div>
      </>
    );
  }

  if (useCase === 'vehicle') {
    const cc = m.class_counts || {};
    const total = m.detection_count ?? 0;
    const topClasses = Object.entries(cc).sort((a, b) => b[1] - a[1]).slice(0, 6);
    const leftClasses = topClasses.slice(0, 3);
    const rightClasses = topClasses.slice(3, 6);
    return (
      <>
        {/* Left panel — total + top classes */}
        <div className="absolute left-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none flex flex-col gap-2">
          <MetricCard borderColor="emerald">
            <div className="text-[9px] font-bold uppercase tracking-widest text-emerald-400/70 mb-0.5">TOTAL</div>
            <div className="text-2xl font-black tabular-nums leading-none text-emerald-200">{total}</div>
          </MetricCard>
          {leftClasses.map(([cls, cnt]) => (
            <MetricCard key={cls} borderColor="emerald" className="py-1.5">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[8px] font-bold uppercase tracking-widest text-emerald-400/60">{cls.toUpperCase()}</span>
                <span className="text-lg font-black tabular-nums leading-none text-emerald-200">{cnt}</span>
              </div>
            </MetricCard>
          ))}
        </div>
        {/* Right panel — remaining classes */}
        {rightClasses.length > 0 && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none flex flex-col gap-2">
            {rightClasses.map(([cls, cnt]) => (
              <MetricCard key={cls} borderColor="emerald" className="py-1.5">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-[8px] font-bold uppercase tracking-widest text-emerald-400/60">{cls.toUpperCase()}</span>
                  <span className="text-lg font-black tabular-nums leading-none text-emerald-200">{cnt}</span>
                </div>
              </MetricCard>
            ))}
          </div>
        )}
      </>
    );
  }

  if (useCase === 'flow') {
    const mobility = m.mobility_index ?? 0;
    const count = m.detection_count ?? 0;
    const stalled = m.stalled_pct ?? 0;
    const slow = m.slow_pct ?? 0;
    const medium = m.medium_pct ?? 0;
    const fast = m.fast_pct ?? 0;
    const mobilityColor = mobility >= 70 ? 'text-green-300' : mobility >= 40 ? 'text-yellow-300' : 'text-red-300';
    return (
      <>
        {/* Left panel */}
        <div className="absolute left-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none flex flex-col gap-2">
          <MetricCard borderColor="purple">
            <div className="text-[9px] font-bold uppercase tracking-widest text-purple-400/70 mb-0.5">MOBILITY</div>
            <div className={`text-2xl font-black tabular-nums leading-none ${mobilityColor}`}>{mobility}</div>
          </MetricCard>
          <MetricCard borderColor="purple">
            <div className="text-[9px] font-bold uppercase tracking-widest text-purple-400/70 mb-0.5">VEHICLES</div>
            <div className="text-xl font-black tabular-nums leading-none text-purple-200">{count}</div>
          </MetricCard>
        </div>
        {/* Right panel — speed bands */}
        <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none flex flex-col gap-1">
          <MetricCard borderColor="purple" className="py-1">
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-[8px] font-bold uppercase tracking-widest text-red-400/80">STALLED</span>
              <span className="text-base font-black tabular-nums leading-none text-red-400">{stalled}%</span>
            </div>
          </MetricCard>
          <MetricCard borderColor="purple" className="py-1">
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-[8px] font-bold uppercase tracking-widest text-orange-400/80">SLOW</span>
              <span className="text-base font-black tabular-nums leading-none text-orange-300">{slow}%</span>
            </div>
          </MetricCard>
          <MetricCard borderColor="purple" className="py-1">
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-[8px] font-bold uppercase tracking-widest text-yellow-400/80">MEDIUM</span>
              <span className="text-base font-black tabular-nums leading-none text-yellow-300">{medium}%</span>
            </div>
          </MetricCard>
          <MetricCard borderColor="purple" className="py-1">
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-[8px] font-bold uppercase tracking-widest text-green-400/80">FAST</span>
              <span className="text-base font-black tabular-nums leading-none text-green-300">{fast}%</span>
            </div>
          </MetricCard>
        </div>
      </>
    );
  }

  if (useCase === 'crowd') {
    const count = m.crowd_count ?? m.detection_count ?? 0;
    const riskScore = m.risk_score ?? 0;
    const opStatus = m.operational_status ?? 'MONITOR';
    const density = m.crowd_density ?? m.traffic_density ?? 0;
    const trend = m.crowd_trend ?? 'stable';

    const riskColor = riskScore >= 60 ? 'text-red-400' : riskScore >= 30 ? 'text-amber-400' : 'text-teal-400';
    const statusColor = riskScore >= 60 ? 'text-red-400' : riskScore >= 30 ? 'text-amber-400' : 'text-teal-400';
    const trendArrow = trend === 'increasing' ? '\u2191' : trend === 'decreasing' ? '\u2193' : '\u2192';
    return (
      <>
        {/* Left panel — PEOPLE + RISK */}
        <div className="absolute left-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none flex flex-col gap-2">
          <MetricCard borderColor="cyan">
            <div className="text-[9px] font-bold uppercase tracking-widest text-teal-500/60 mb-0.5">PEOPLE</div>
            <div className="text-2xl font-black tabular-nums leading-none text-teal-300">
              {count} <span className="text-sm">{trendArrow}</span>
            </div>
          </MetricCard>
          <MetricCard borderColor="cyan">
            <div className="text-[9px] font-bold uppercase tracking-widest text-teal-500/60 mb-0.5">RISK</div>
            <div className={`text-xl font-black tabular-nums leading-none ${riskColor}`}>{riskScore}</div>
          </MetricCard>
        </div>
        {/* Right panel — STATUS + DENSITY */}
        <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none flex flex-col gap-2">
          <MetricCard borderColor="cyan">
            <div className="text-[9px] font-bold uppercase tracking-widest text-teal-500/60 mb-0.5">STATUS</div>
            <div className={`text-lg font-black leading-none ${statusColor}`}>{opStatus}</div>
          </MetricCard>
          <MetricCard borderColor="cyan">
            <div className="text-[9px] font-bold uppercase tracking-widest text-teal-500/60 mb-0.5">DENSITY</div>
            <div className="text-xl font-black tabular-nums leading-none text-teal-300">{Math.round(density)}%</div>
          </MetricCard>
        </div>
      </>
    );
  }

  return null;
};

// WebRTC WHEP player — connects to MediaMTX for RTSP streams
// WebRTC WHEP hook — used for raw RTSP feeds from MediaMTX (instant, low-latency)
const useWebRTC = (streamName, enabled = true, options = {}) => {
  const { initialDelayMs = 0, retryDelayMs = 3000, readyUrl = null } = options;
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!streamName || !enabled) return;

    let cancelled = false;
    let retryTimer = null;
    let startTimer = null;

    const waitForReady = async () => {
      if (!readyUrl) return true;
      const deadline = Date.now() + 15000;
      while (!cancelled && Date.now() < deadline) {
        try {
          const res = await fetch(readyUrl);
          if (res.ok) {
            const data = await res.json();
            if (data.ready) return true;
          }
        } catch (e) {}
        await new Promise(r => setTimeout(r, 800));
      }
      return false;
    };

    const connect = async () => {
      if (cancelled) return;
      if (readyUrl) {
        const ok = await waitForReady();
        if (!ok || cancelled) {
          retryTimer = setTimeout(connect, retryDelayMs);
          return;
        }
      }

      const pc = new RTCPeerConnection({ iceServers: [] });
      pcRef.current = pc;

      pc.addTransceiver('video', { direction: 'recvonly' });
      pc.addTransceiver('audio', { direction: 'recvonly' });

      pc.ontrack = (e) => {
        if (videoRef.current && e.streams[0]) {
          videoRef.current.srcObject = e.streams[0];
          if (!cancelled) setConnected(true);
        }
      };

      pc.oniceconnectionstatechange = () => {
        if (pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'disconnected') {
          if (!cancelled) {
            setConnected(false);
            pc.close();
            retryTimer = setTimeout(connect, retryDelayMs);
          }
        }
      };

      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const res = await fetch(`${WEBRTC_BASE}/${streamName}/whep`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/sdp' },
          body: offer.sdp,
        });

        if (!res.ok) {
          pc.close();
          if (!cancelled) retryTimer = setTimeout(connect, retryDelayMs);
          return;
        }

        const sdp = await res.text();
        if (!cancelled) {
          await pc.setRemoteDescription({ type: 'answer', sdp });
        }
      } catch (e) {
        pc.close();
        if (!cancelled) retryTimer = setTimeout(connect, retryDelayMs);
      }
    };

    startTimer = setTimeout(connect, initialDelayMs);

    return () => {
      cancelled = true;
      if (startTimer) clearTimeout(startTimer);
      if (retryTimer) clearTimeout(retryTimer);
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
      setConnected(false);
    };
  }, [streamName, enabled, initialDelayMs, retryDelayMs]);

  return { videoRef, connected };
};

const VideoCell = ({ video, index, total, getVideoClass, useCase, sourceMetrics }) => {
  const [isCellLoading, setIsCellLoading] = useState(true);
  const [rawPlaying, setRawPlaying] = useState(false);
  const [processedPlaying, setProcessedPlaying] = useState(false);
  const [samResult, setSamResult] = useState(null);

  const isUpload = video.type === 'upload';

  // Phase 1: instant raw RTSP via WebRTC (skip for uploads — no raw source exists)
  const { videoRef: rawVideoRef, connected: rawConnected } = useWebRTC(video.id, !isUpload);

  // Phase 2: processed feed via WebRTC (libx264 baseline → RTSP → MediaMTX → WebRTC)
  const processedStreamName = `processed_${video.id}`;
  const isForensics = useCase === 'forensics';
  const { videoRef: processedVideoRef, connected: processedConnected } = useWebRTC(
    processedStreamName,
    isUpload || !isForensics,  // Uploads always use processed; non-forensics drones too
    {
      initialDelayMs: 2000,
      retryDelayMs: 3000,
      readyUrl: `${API_BASE_URL}/processed/${video.id}/ready`,
    }
  );

  // Hide loader once video is rendering — brief delay to show "FEED LOCKED" state
  useEffect(() => {
    const isReady = isUpload ? processedPlaying : rawPlaying;
    if (!isReady) return;
    const timer = setTimeout(() => setIsCellLoading(false), 600);
    return () => clearTimeout(timer);
  }, [rawPlaying, processedPlaying, isUpload]);

  // Poll forensics result when in forensics mode
  useEffect(() => {
    if (!isForensics) {
      setSamResult(null);
      return;
    }
    const fetchResult = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/sam/result/${video.id}`);
        if (res.ok) {
          const data = await res.json();
          if (data.annotated_frame) {
            setSamResult(data);
          }
        }
      } catch (e) {}
    };
    fetchResult();
    const interval = setInterval(fetchResult, 1000);
    return () => clearInterval(interval);
  }, [video.id, isForensics]);

  const maskStyle = {
    maskImage: 'linear-gradient(to right, transparent 0%, black 12%, black 88%, transparent 100%), linear-gradient(to bottom, transparent 0%, black 12%, black 88%, transparent 100%)',
    maskComposite: 'intersect',
    WebkitMaskComposite: 'source-in'
  };

  const showForensicsResult = isForensics && !!samResult;
  const showProcessed = processedConnected && (isUpload || !isForensics) && !showForensicsResult;

  return (
    <div className={`relative overflow-hidden group ${getVideoClass(index, total)}`}>
      <AnimatePresence>
        {isCellLoading && (
          <IRISLoader
            connected={isUpload ? processedConnected : rawConnected}
            ready={isUpload ? processedPlaying : rawPlaying}
          />
        )}
      </AnimatePresence>

      {/* Tactical Cyberpunk Brackets */}
      <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-cyan-500/50 z-20 pointer-events-none" />
      <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-cyan-500/50 z-20 pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-cyan-500/50 z-20 pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-cyan-500/50 z-20 pointer-events-none" />

      {/* Inner corner accents */}
      <div className="absolute top-4 left-4 w-2 h-2 border-t border-l border-white/20 z-20 pointer-events-none" />
      <div className="absolute top-4 right-4 w-2 h-2 border-t border-r border-white/20 z-20 pointer-events-none" />
      <div className="absolute bottom-4 left-4 w-2 h-2 border-b border-l border-white/20 z-20 pointer-events-none" />
      <div className="absolute bottom-4 right-4 w-2 h-2 border-b border-r border-white/20 z-20 pointer-events-none" />

      {/* Drone label - inside corner accents, below top-left inner bracket */}
      <div className="absolute top-[25px] left-5 z-30 pointer-events-none">
        <div className="flex items-center gap-1.5 px-2 py-0.5 bg-black/70 border border-cyan-400/30 backdrop-blur-md shadow-[0_0_10px_rgba(6,182,212,0.12)]">
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-[10px] font-black uppercase tracking-[0.15em] text-cyan-300">
            {video.label || video.id}
          </span>
        </div>
      </div>

      {/* Layer 1: WebRTC raw RTSP — instant, always underneath (hidden for uploads) */}
      {!isUpload && (
        <video
          ref={rawVideoRef}
          autoPlay
          playsInline
          muted
          onPlaying={() => setRawPlaying(true)}
          className="absolute inset-0 w-full h-full object-cover"
          style={{
            ...maskStyle,
            opacity: isCellLoading ? 0 : (showProcessed || showForensicsResult) ? 0 : 1,
            transition: isCellLoading ? 'none' : 'opacity 0.5s ease',
          }}
        />
      )}

      {/* Layer 2: WebRTC processed stream — fades in once connected (primary layer for uploads) */}
      {(isUpload || !isForensics) && (
        <video
          ref={processedVideoRef}
          autoPlay
          playsInline
          muted
          onPlaying={() => setProcessedPlaying(true)}
          className="absolute inset-0 w-full h-full object-cover"
          style={{
            ...maskStyle,
            opacity: showProcessed ? 1 : 0,
            transition: 'opacity 0.5s ease',
          }}
        />
      )}

      {/* Layer 2 alt: Forensics SAM result */}
      {showForensicsResult && (
        <img
          src={`data:image/jpeg;base64,${samResult.annotated_frame}`}
          alt="IRIS Forensics Result"
          className="absolute inset-0 w-full h-full object-cover"
          style={maskStyle}
        />
      )}

      {/* Awaiting analysis prompt overlay */}
      {isForensics && !samResult && (isUpload ? processedConnected : rawConnected) && (
        <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
          <div className="px-4 py-2 bg-black/70 border border-amber-500/30 backdrop-blur-sm">
            <span className="text-[10px] font-black uppercase tracking-[0.3em] text-amber-400/80">
              AWAITING ANALYSIS PROMPT
            </span>
          </div>
        </div>
      )}

      {/* Per-video metrics HUD — left & right panels */}
      <MetricsHUD metrics={sourceMetrics} useCase={useCase} samResult={samResult} />
    </div>
  );
};

const Dashboard = ({ onLogout }) => {
  const { useCase } = useParams();
  const navigate = useNavigate();
  const prevUseCaseRef = useRef(useCase);

  const modeConfig = MODE_CONFIG[useCase];

  if (!modeConfig) {
    return <Navigate to="/" replace />;
  }

  const theme = modeConfig.theme;

  const ALL_DRONES = [
    { id: 'bcpdrone1', type: 'webrtc', stream: 'bcpdrone1', processedStream: 'processed_bcpdrone1', droneIndex: 1, label: 'DRONE 1' },
    { id: 'bcpdrone2', type: 'webrtc', stream: 'bcpdrone2', processedStream: 'processed_bcpdrone2', droneIndex: 2, label: 'DRONE 2' },
    { id: 'bcpdrone3', type: 'webrtc', stream: 'bcpdrone3', processedStream: 'processed_bcpdrone3', droneIndex: 3, label: 'DRONE 3' },
    { id: 'bcpdrone4', type: 'webrtc', stream: 'bcpdrone4', processedStream: 'processed_bcpdrone4', droneIndex: 4, label: 'DRONE 4' },
    { id: 'bcpdrone5', type: 'webrtc', stream: 'bcpdrone5', processedStream: 'processed_bcpdrone5', droneIndex: 5, label: 'DRONE 5' },
    { id: 'bcpdrone6', type: 'webrtc', stream: 'bcpdrone6', processedStream: 'processed_bcpdrone6', droneIndex: 6, label: 'DRONE 6' },
    { id: 'bcpdrone7', type: 'webrtc', stream: 'bcpdrone7', processedStream: 'processed_bcpdrone7', droneIndex: 7, label: 'DRONE 7' },
    { id: 'bcpdrone8', type: 'webrtc', stream: 'bcpdrone8', processedStream: 'processed_bcpdrone8', droneIndex: 8, label: 'DRONE 8' },
    { id: 'bcpdrone9', type: 'webrtc', stream: 'bcpdrone9', processedStream: 'processed_bcpdrone9', droneIndex: 9, label: 'DRONE 9' },
    { id: 'bcpdrone10', type: 'webrtc', stream: 'bcpdrone10', processedStream: 'processed_bcpdrone10', droneIndex: 10, label: 'DRONE 10' },
    { id: 'bcpdrone11', type: 'webrtc', stream: 'bcpdrone11', processedStream: 'processed_bcpdrone11', droneIndex: 11, label: 'DRONE 11' },
    { id: 'bcpdrone12', type: 'webrtc', stream: 'bcpdrone12', processedStream: 'processed_bcpdrone12', droneIndex: 12, label: 'DRONE 12' },
  ];

  // Filter drones based on mode: crowd gets drone 10 & 12 only, others exclude 10 & 12
  const CROWD_ONLY_DRONES = [10, 12];
  const filteredDrones = useMemo(() => {
    if (useCase === 'crowd') {
      return ALL_DRONES.filter(d => CROWD_ONLY_DRONES.includes(d.droneIndex));
    }
    return ALL_DRONES.filter(d => !CROWD_ONLY_DRONES.includes(d.droneIndex));
  }, [useCase]);

  const [allVideos, setAllVideos] = useState(filteredDrones);
  const [selectedVideos, setSelectedVideos] = useState([]);
  const [allMetrics, setAllMetrics] = useState({});

  // Sync allVideos when useCase changes (different drones for different modes)
  useEffect(() => {
    setAllVideos(filteredDrones);
    setSelectedVideos([]);  // Clear selection when switching modes
  }, [filteredDrones]);

  // Poll metrics from backend every 1.5s
  useEffect(() => {
    if (selectedVideos.length === 0) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (res.ok && !cancelled) {
          const data = await res.json();
          setAllMetrics(data);
        }
      } catch (e) {}
    };
    poll();
    const interval = setInterval(poll, 1500);
    return () => { cancelled = true; clearInterval(interval); };
  }, [selectedVideos]);

  // Stream lifecycle: start selected videos — backend handles overlays per mode
  useEffect(() => {
    let cancelled = false;

    const startStreams = async () => {
      for (const v of selectedVideos) {
        if (cancelled || v.type === 'upload') continue;  // uploads already processing via /api/upload
        await startDroneProcessing(v.droneIndex, useCase);
      }
    };

    startStreams();

    return () => {
      cancelled = true;
    };
  }, [selectedVideos, useCase]);

  // On unmount (navigate to /), stop all processing
  useEffect(() => {
    return () => {
      stopAllProcessing();
    };
  }, []);

  const getGridLayout = (count) => {
    switch (count) {
      case 1: return 'grid-cols-1 grid-rows-1';
      case 2: return 'grid-cols-2 grid-rows-1';
      case 3:
      case 4: return 'grid-cols-2 grid-rows-2';
      case 5:
      case 6: return 'grid-cols-3 grid-rows-2';
      default: return 'grid-cols-3 grid-rows-3';
    }
  };

  const getVideoClass = (index, total) => {
    if (index === 0) {
      if (total === 3) return 'col-span-2';
      if (total === 5) return 'col-span-2';
      if (total === 7) return 'col-span-3';
    }
    return '';
  };

  // Safety mode - Coming Soon page
  if (useCase === 'safety') {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 flex flex-col">
        <Header onReset={() => navigate('/')} useCase={useCase} onLogout={onLogout} />
        <div className="flex-1 flex items-center justify-center relative overflow-hidden">
          {/* Background grid */}
          <div className="absolute inset-0 z-0 opacity-40 pointer-events-none"
            style={{
              backgroundImage: `linear-gradient(${theme.grid} 1px, transparent 1px), linear-gradient(90deg, ${theme.grid} 1px, transparent 1px)`,
              backgroundSize: '40px 40px'
            }}>
          </div>
          <div className="absolute inset-0 z-0 pointer-events-none"
            style={{ background: `radial-gradient(circle_at_center, transparent 40%, ${theme.glow} 100%)` }}
          ></div>

          {/* Coming Soon content */}
          <div className="relative z-10 text-center space-y-6">
            <div className="w-24 h-24 mx-auto border-2 border-emerald-500/50 flex items-center justify-center">
              <div className="w-16 h-16 border border-emerald-500/30 flex items-center justify-center">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                  className="w-8 h-8 border-t-2 border-r-2 border-emerald-400"
                />
              </div>
            </div>
            <div className="space-y-2">
              <h1 className="text-4xl font-black uppercase tracking-[0.3em] text-emerald-400">
                Coming Soon
              </h1>
              <p className="text-sm text-white/40 tracking-widest uppercase">
                Safety Analytics Module
              </p>
            </div>
            <div className="pt-4 space-y-1">
              <div className="text-[10px] text-white/30 uppercase tracking-wider">Features in Development</div>
              <div className="flex flex-wrap justify-center gap-2 max-w-md mx-auto">
                {['PPE Detection', 'Hazard Zones', 'Incident Alerts', 'Compliance Tracking'].map((feature) => (
                  <span key={feature} className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 text-[10px] text-emerald-400/70 uppercase tracking-wider">
                    {feature}
                  </span>
                ))}
              </div>
            </div>
            <button
              onClick={() => navigate('/')}
              className="mt-8 px-6 py-2 border border-white/20 text-white/60 text-xs uppercase tracking-widest hover:bg-white/5 transition-colors"
            >
              Return to Dashboard
            </button>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 flex flex-col">
      <Header onReset={() => navigate('/')} useCase={useCase} onLogout={onLogout} />

      <div className="flex-1 flex min-h-0">
        <div className="flex-1 relative overflow-hidden">
          <div className="absolute inset-0 z-0 opacity-40 pointer-events-none"
            style={{
              backgroundImage: `linear-gradient(${theme.grid} 1px, transparent 1px), linear-gradient(90deg, ${theme.grid} 1px, transparent 1px)`,
              backgroundSize: '40px 40px'
            }}>
          </div>
          <div className="absolute inset-0 z-0 opacity-10 pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,#000_100%)]"></div>
          <div className="absolute inset-0 z-0 pointer-events-none"
            style={{ background: `radial-gradient(circle_at_center, transparent 40%, ${theme.glow} 100%)` }}
          ></div>

          <div className="absolute inset-0 z-0 flex items-center justify-center">
            <div className={`w-full h-full grid gap-2 p-4 ${getGridLayout(selectedVideos.length)}`}>
              {selectedVideos.map((video, index) => (
                <VideoCell
                  key={video.id}
                  video={video}
                  index={index}
                  total={selectedVideos.length}
                  getVideoClass={getVideoClass}
                  useCase={useCase}
                  sourceMetrics={allMetrics[video.id] || null}
                />
              ))}
            </div>
          </div>

          <div className="absolute inset-0 z-0 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.6)_100%)] pointer-events-none"></div>
        </div>

        <RightPanel useCase={useCase} sources={allVideos} selectedVideos={selectedVideos} />
      </div>

      <Footer selectedVideos={selectedVideos} onVideosChange={setSelectedVideos} videos={allVideos} useCase={useCase} />
    </motion.div>
  );
}

const AppContents = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return localStorage.getItem('iris_authenticated') === 'true';
  });
  const location = useLocation();

  const handleLogin = () => {
    localStorage.setItem('iris_authenticated', 'true');
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('iris_authenticated');
    localStorage.removeItem('iris_username');
    setIsAuthenticated(false);
  };

  return (
    <div className="relative w-screen h-screen bg-[#050a14] overflow-hidden selection:bg-cyan-500/30 font-mono flex">
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route
            path="/"
            element={
              isAuthenticated
                ? <WelcomeScreen key="welcome" />
                : <Login onLogin={handleLogin} />
            }
          />
          <Route
            path="/:useCase"
            element={
              isAuthenticated
                ? <Dashboard key="dashboard" onLogout={handleLogout} />
                : <Navigate to="/" replace />
            }
          />
        </Routes>
      </AnimatePresence>
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <AppContents />
    </BrowserRouter>
  );
}

export default App;
