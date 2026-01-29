import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrowserRouter, Routes, Route, useParams, useNavigate, useLocation } from 'react-router-dom';
import CountUp from 'react-countup';

import Header from './components/Dashboard/Header';
import RightPanel from './components/Dashboard/RightPanel';
import Footer from './components/Dashboard/Footer';
import IRISLoader from './components/Dashboard/IRISLoader';
import WelcomeScreen from './components/Dashboard/WelcomeScreen';
import HLSVideo from './components/UI/HLSVideo';
import WebRTCVideo from './components/UI/WebRTCVideo';

// MediaMTX stream configuration
// In development, Vite proxies /hls/* to MediaMTX :8888
// In production, configure nginx/reverse proxy accordingly
const HLS_BASE_URL = import.meta.env.DEV ? '/hls' : `http://${window.location.hostname}:8888`;
const WEBRTC_BASE_URL = import.meta.env.DEV ? '/webrtc' : `http://${window.location.hostname}:8889`;
const API_BASE_URL = import.meta.env.DEV ? '/api' : `http://${window.location.hostname}:9010`;

// Start processing a drone on the backend
const startDroneProcessing = async (droneIndex) => {
  try {
    await fetch(`${API_BASE_URL}/sources/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index: droneIndex }),
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

const CameraAnalytics = ({ scale = 1, camIndex = 0, useCase, videoId }) => {
  const [metrics, setMetrics] = useState(null);
  const [hasReceivedData, setHasReceivedData] = useState(false);

  // Use videoId directly as the source name (e.g., 'bcpdrone1', 'bcpdrone3')
  const sourceName = videoId;

  useEffect(() => {
    if (!sourceName) return;

    const fetchMetrics = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        if (response.ok) {
          const data = await response.json();
          // Check if metrics exist for this source
          if (data[sourceName]) {
            setMetrics(data[sourceName]);
            setHasReceivedData(true);
          }
        }
      } catch (e) {
        console.error('Failed to fetch metrics:', e);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 500);
    return () => clearInterval(interval);
  }, [sourceName]);

  // Reset when videoId changes
  useEffect(() => {
    setHasReceivedData(false);
    setMetrics(null);
  }, [videoId]);

  const congestion = metrics?.congestion_index || 0;
  const density = metrics?.traffic_density || 0;
  const speed = Math.round(metrics?.mobility_index || 0);
  // const fps = metrics?.fps || 0;
  const detections = metrics?.detection_count || 0;

  // Color based on congestion (higher = worse = red)
  const getCongestionColor = (val) => {
    if (val > 75) return '#ef4444'; // red
    if (val > 50) return '#eab308'; // yellow
    if (val > 25) return '#34d399'; // emerald-400
    return '#10b981'; // emerald-500
  };

  // Color based on mobility (higher = better = green)
  const getMobilityColor = (val) => {
    if (val > 75) return '#10b981'; // emerald-500
    if (val > 50) return '#34d399'; // emerald-400
    if (val > 25) return '#eab308'; // yellow
    return '#ef4444'; // red
  };

  // Color based on density (higher = worse = red)
  const getDensityColor = (val) => {
    if (val > 75) return '#ef4444'; // red
    if (val > 50) return '#eab308'; // yellow
    if (val > 25) return '#34d399'; // emerald-400
    return '#10b981'; // emerald-500
  };

  // Status based on real congestion index
  let status = 'SMOOTH';
  let color = 'text-emerald-500';
  let borderColor = 'border-emerald-500/50';
  if (congestion > 75) { status = 'HEAVY'; color = 'text-red-500'; borderColor = 'border-red-500/50'; }
  else if (congestion > 50) { status = 'SLOW'; color = 'text-yellow-500'; borderColor = 'border-yellow-500/50'; }
  else if (congestion > 25) { status = 'MODERATE'; color = 'text-emerald-400'; borderColor = 'border-emerald-400/50'; }

  // Labels based on Use Case
  const configs = {
    traffic: { primary: 'MOBILITY', secondary: 'CONGESTION', statusLabel: 'Traffic' },
    crowd: { primary: 'DENSITY', secondary: 'FLOW RATE', statusLabel: 'Crowd' },
    safety: { primary: 'RISK LVL', secondary: 'ALERT CONF', statusLabel: 'Safety' },
    perimeter: { primary: 'PROXIMITY', secondary: 'SIGNAL', statusLabel: 'Perimeter' }
  };

  const config = configs[useCase] || configs.traffic;

  // Only show overlay when receiving processed frames (metrics data exists for this source)
  if (!hasReceivedData) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
      className="absolute inset-0 pointer-events-none"
    >
      {/* Left-side Analytics */}
      <div
        className="absolute left-8 top-1/2 -translate-y-1/2 flex flex-col gap-5 pointer-events-none origin-left"
        style={{ transform: `scale(${scale})` }}
      >
        {/* Mobility Index */}
        <div className="bg-black/70 backdrop-blur-md rounded-lg p-5 border border-emerald-500/30">
          <div className="text-sm text-emerald-400 font-mono font-bold tracking-wider mb-2">{config.primary}</div>
          <div
            className="text-7xl font-black font-mono drop-shadow-[0_0_20px_currentColor]"
            style={{ color: getMobilityColor(speed) }}
          >
            <CountUp
              end={speed}
              duration={0.5}
              preserveValue={true}
            />
          </div>
        </div>

        {/* Real-time Stats */}
        <div className="font-mono flex flex-col gap-4 bg-black/70 backdrop-blur-md rounded-lg p-4 border border-white/10">
          <div className="flex items-center justify-between gap-6">
            <span className="text-emerald-400 font-black text-sm tracking-wider">DENSITY</span>
            <span
              className="font-black text-2xl drop-shadow-[0_0_10px_currentColor]"
              style={{ color: getDensityColor(density) }}
            >
              <CountUp
                end={density}
                duration={0.5}
                preserveValue={true}
              />
            </span>
          </div>
          {/* <div className="flex items-center justify-between gap-6">
            <span className="text-emerald-400 font-black text-sm tracking-wider">FPS</span>
            <span className="text-emerald-300 font-black text-2xl">{fps.toFixed(1)}</span>
          </div> */}
          <div className="flex items-center justify-between gap-6">
            <span className="text-purple-400 font-black text-sm tracking-wider">OBJECTS</span>
            <span className="text-purple-300 font-black text-2xl">
              <CountUp
                end={detections}
                duration={0.3}
                preserveValue={true}
              />
            </span>
          </div>
        </div>
      </div>

      {/* Right-side Analytics */}
<div
  className="absolute right-8 top-[42%] -translate-y-1/2 flex flex-col items-end gap-6 pointer-events-none origin-right"
  style={{ transform: `scale(${scale})` }}
>


        {/* Congestion Index */}
        <div className="bg-black/70 backdrop-blur-md rounded-lg p-5 border border-emerald-500/30 text-right">
          <div className="text-sm text-emerald-400 font-mono font-bold tracking-wider mb-2">{config.secondary}</div>
          <div
            className="text-7xl font-black font-mono drop-shadow-[0_0_20px_currentColor]"
            style={{ color: getCongestionColor(congestion) }}
          >
            <CountUp
              end={congestion}
              duration={0.5}
              preserveValue={true}
            />
          </div>
        </div>

        {/* Speed Distribution */}
        <div className="flex flex-col gap-3 bg-black/50 backdrop-blur-md rounded-lg p-4 border border-white/10">
          <div className="flex items-center justify-between gap-6">
            <span className="text-red-400 font-mono text-sm font-bold">STALLED</span>
            <span className="text-red-300 font-black text-xl">
              <CountUp end={metrics?.stalled_pct || 0} duration={0.3} preserveValue={true} />
            </span>
          </div>
          <div className="flex items-center justify-between gap-6">
            <span className="text-orange-400 font-mono text-sm font-bold">SLOW</span>
            <span className="text-orange-300 font-black text-xl">
              <CountUp end={metrics?.slow_pct || 0} duration={0.3} preserveValue={true} />
            </span>
          </div>
          <div className="flex items-center justify-between gap-6">
            <span className="text-yellow-400 font-mono text-sm font-bold">MEDIUM</span>
            <span className="text-yellow-300 font-black text-xl">
              <CountUp end={metrics?.medium_pct || 0} duration={0.3} preserveValue={true} />
            </span>
          </div>
          <div className="flex items-center justify-between gap-6">
            <span className="text-emerald-400 font-mono text-sm font-bold">FAST</span>
            <span className="text-emerald-300 font-black text-xl">
              <CountUp end={metrics?.fast_pct || 0} duration={0.3} preserveValue={true} />
            </span>
          </div>
        </div>
      </div>

      {/* Status Label */}
<div
  className="absolute bottom-2 right-8 pointer-events-none origin-right z-10"
  style={{ transform: `scale(${scale})` }}
>


        <div className={`px-6 py-3 border-r-[8px] ${borderColor} bg-black/80 backdrop-blur-md transform skew-x-[-12deg] rounded-l-lg`}>
          <div className={`text-xl font-black uppercase tracking-[0.3em] ${color} skew-x-[12deg] drop-shadow-[0_0_10px_currentColor]`}>
            {config.statusLabel}: {status}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

const VideoCell = ({
  video,
  index,
  total,
  getAnalyticsScale,
  useCase,
  position,
  size,
  onPositionChange,
  onSizeChange,
  onBringToFront,
  zIndex
}) => {
  const [isCellLoading, setIsCellLoading] = useState(true);
  const [overlayState, setOverlayState] = useState({
    heatmap: false,
    trails: false,
    bboxes: false,
  });

  // Zoom state for video content
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isZoomDragging, setIsZoomDragging] = useState(false);
  const [zoomDragStart, setZoomDragStart] = useState({ x: 0, y: 0 });

  // Cell drag state
  const [isCellDragging, setIsCellDragging] = useState(false);
  const [cellDragStart, setCellDragStart] = useState({ x: 0, y: 0 });

  // Resize state
  const [isResizing, setIsResizing] = useState(false);
  const [resizeHandle, setResizeHandle] = useState(null);
  const [resizeStart, setResizeStart] = useState({ x: 0, y: 0, width: 0, height: 0 });

  const containerRef = React.useRef(null);

  useEffect(() => {
    const timer = setTimeout(() => setIsCellLoading(false), 2500);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const overlayKey = video.overlayKey || video.stream;
    if (!overlayKey) return;
    fetch(`${API_BASE_URL}/overlays/${overlayKey}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) {
          setOverlayState({
            heatmap: !!data.heatmap,
            trails: !!data.trails,
            bboxes: !!data.bboxes,
          });
        }
      })
      .catch(() => {});
  }, [video.stream, video.overlayKey]);

  const updateOverlay = (next) => {
    const overlayKey = video.overlayKey || video.stream;
    if (!overlayKey) return;
    setOverlayState(next);
    fetch(`${API_BASE_URL}/overlays/${overlayKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(next),
    }).catch(() => {});
  };

  const toggleOverlay = (key) => {
    updateOverlay({ ...overlayState, [key]: !overlayState[key] });
  };

  // === Cell Drag Handlers ===
  const handleCellDragStart = (e) => {
    if (e.target.closest('.resize-handle') || e.target.closest('.no-drag')) return;
    e.preventDefault();
    e.stopPropagation();
    onBringToFront?.();
    setIsCellDragging(true);
    setCellDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleCellDragMove = (e) => {
    if (isCellDragging) {
      const newX = Math.max(0, e.clientX - cellDragStart.x);
      const newY = Math.max(0, e.clientY - cellDragStart.y);
      onPositionChange?.({ x: newX, y: newY });
    }
    if (isResizing && resizeHandle) {
      const deltaX = e.clientX - resizeStart.x;
      const deltaY = e.clientY - resizeStart.y;
      let newWidth = resizeStart.width;
      let newHeight = resizeStart.height;
      let newX = position.x;
      let newY = position.y;

      if (resizeHandle.includes('e')) newWidth = Math.max(300, resizeStart.width + deltaX);
      if (resizeHandle.includes('w')) {
        newWidth = Math.max(300, resizeStart.width - deltaX);
        newX = position.x + (resizeStart.width - newWidth);
      }
      if (resizeHandle.includes('s')) newHeight = Math.max(200, resizeStart.height + deltaY);
      if (resizeHandle.includes('n')) {
        newHeight = Math.max(200, resizeStart.height - deltaY);
        newY = position.y + (resizeStart.height - newHeight);
      }

      onSizeChange?.({ width: newWidth, height: newHeight });
      if (resizeHandle.includes('w') || resizeHandle.includes('n')) {
        onPositionChange?.({ x: newX, y: newY });
      }
    }
    // Zoom pan
    if (isZoomDragging && zoom > 1) {
      const newX = e.clientX - zoomDragStart.x;
      const newY = e.clientY - zoomDragStart.y;
      const maxPan = (zoom - 1) * 150;
      setPan({
        x: Math.max(-maxPan, Math.min(maxPan, newX)),
        y: Math.max(-maxPan, Math.min(maxPan, newY))
      });
    }
  };

  const handleCellDragEnd = () => {
    setIsCellDragging(false);
    setIsResizing(false);
    setResizeHandle(null);
    setIsZoomDragging(false);
  };

  // === Resize Handlers ===
  const handleResizeStart = (e, handle) => {
    e.preventDefault();
    e.stopPropagation();
    onBringToFront?.();
    setIsResizing(true);
    setResizeHandle(handle);
    setResizeStart({ x: e.clientX, y: e.clientY, width: size.width, height: size.height });
  };

  // === Zoom Handlers ===
  const handleZoomClick = (e) => {
    if (isCellDragging || isResizing || isZoomDragging) return;
    e.stopPropagation();
    setZoom(prev => prev >= 3 ? 1 : prev + 0.5);
    if (zoom >= 3) setPan({ x: 0, y: 0 });
  };

  const handleZoomPanStart = (e) => {
    if (zoom <= 1 || e.target.closest('.no-drag')) return;
    e.preventDefault();
    setIsZoomDragging(true);
    setZoomDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.25 : 0.25;
    setZoom(prev => Math.max(1, Math.min(4, prev + delta)));
    if (zoom + delta <= 1) setPan({ x: 0, y: 0 });
  };

  const resetView = (e) => {
    e.stopPropagation();
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Global mouse events for drag/resize
  useEffect(() => {
    if (isCellDragging || isResizing || isZoomDragging) {
      const handleGlobalMove = (e) => handleCellDragMove(e);
      const handleGlobalUp = () => handleCellDragEnd();
      window.addEventListener('mousemove', handleGlobalMove);
      window.addEventListener('mouseup', handleGlobalUp);
      return () => {
        window.removeEventListener('mousemove', handleGlobalMove);
        window.removeEventListener('mouseup', handleGlobalUp);
      };
    }
  }, [isCellDragging, isResizing, isZoomDragging, cellDragStart, resizeStart, zoomDragStart]);

  // Determine video source - use HLS stream for 'live' or fallback to static file
  const isLiveStream = video.type === 'hls';
  const isWebRTCStream = video.type === 'webrtc';
  const hlsSrc = isLiveStream ? `${HLS_BASE_URL}/${video.stream}/index.m3u8` : null;
  const processedStream = video.processedStream;
  const webrtcPrimaryStream = isWebRTCStream && processedStream ? processedStream : video.stream;
  const webrtcFallbackStream = isWebRTCStream && processedStream && processedStream !== video.stream
    ? video.stream
    : null;
  const webrtcSrc = isWebRTCStream ? `${WEBRTC_BASE_URL}/${webrtcPrimaryStream}/whep` : null;
  const webrtcFallbackSrc = webrtcFallbackStream
    ? `${WEBRTC_BASE_URL}/${webrtcFallbackStream}/whep`
    : null;
  const fallbackSrc = video.fallback || (video.type === 'static' ? `/${video.id}` : null);

  const videoStyle = {
    transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
    transition: isZoomDragging ? 'none' : 'transform 0.3s ease-out',
    cursor: zoom > 1 ? (isZoomDragging ? 'grabbing' : 'grab') : 'zoom-in',
  };

  return (
    <div
      ref={containerRef}
      className="absolute overflow-hidden group border border-emerald-500/30 rounded-lg shadow-lg shadow-emerald-500/10"
      style={{
        left: position.x,
        top: position.y,
        width: size.width,
        height: size.height,
        zIndex: zIndex,
      }}
      onWheel={handleWheel}
      onClick={() => onBringToFront?.()}
    >
      <AnimatePresence>
        {isCellLoading && <IRISLoader />}
      </AnimatePresence>

      {/* Drag Handle - Top Bar */}
      <div
        className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-black/80 to-transparent z-30 cursor-move flex items-center px-3 gap-2"
        onMouseDown={handleCellDragStart}
      >
        <div className="flex gap-1">
          <div className="w-2 h-2 rounded-full bg-emerald-500/60" />
          <div className="w-2 h-2 rounded-full bg-emerald-500/40" />
          <div className="w-2 h-2 rounded-full bg-emerald-500/20" />
        </div>
        <span className="text-[10px] font-mono text-emerald-400/80 font-bold tracking-wider">
          {video.label || `DRONE ${index + 1}`}
        </span>
      </div>

      {/* Tactical Emerald Brackets */}
      <div className="absolute top-0 left-0 w-10 h-10 border-t-2 border-l-2 border-emerald-500/60 z-20 pointer-events-none" />
      <div className="absolute top-0 right-0 w-10 h-10 border-t-2 border-r-2 border-emerald-500/60 z-20 pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-10 h-10 border-b-2 border-l-2 border-emerald-500/60 z-20 pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-10 h-10 border-b-2 border-r-2 border-emerald-500/60 z-20 pointer-events-none" />

      {/* Resize Handles */}
      <div className="resize-handle absolute top-0 left-0 w-4 h-4 cursor-nw-resize z-40 hover:bg-emerald-500/30" onMouseDown={(e) => handleResizeStart(e, 'nw')} />
      <div className="resize-handle absolute top-0 right-0 w-4 h-4 cursor-ne-resize z-40 hover:bg-emerald-500/30" onMouseDown={(e) => handleResizeStart(e, 'ne')} />
      <div className="resize-handle absolute bottom-0 left-0 w-4 h-4 cursor-sw-resize z-40 hover:bg-emerald-500/30" onMouseDown={(e) => handleResizeStart(e, 'sw')} />
      <div className="resize-handle absolute bottom-0 right-0 w-4 h-4 cursor-se-resize z-40 hover:bg-emerald-500/30" onMouseDown={(e) => handleResizeStart(e, 'se')} />
      <div className="resize-handle absolute top-0 left-4 right-4 h-2 cursor-n-resize z-40 hover:bg-emerald-500/20" onMouseDown={(e) => handleResizeStart(e, 'n')} />
      <div className="resize-handle absolute bottom-0 left-4 right-4 h-2 cursor-s-resize z-40 hover:bg-emerald-500/20" onMouseDown={(e) => handleResizeStart(e, 's')} />
      <div className="resize-handle absolute left-0 top-4 bottom-4 w-2 cursor-w-resize z-40 hover:bg-emerald-500/20" onMouseDown={(e) => handleResizeStart(e, 'w')} />
      <div className="resize-handle absolute right-0 top-4 bottom-4 w-2 cursor-e-resize z-40 hover:bg-emerald-500/20" onMouseDown={(e) => handleResizeStart(e, 'e')} />

      {/* Overlay toggles */}
      <div className="no-drag absolute top-1 right-3 z-30 flex items-center gap-2 bg-black/60 border border-emerald-500/30 rounded-md px-2 py-1 backdrop-blur-sm">
        <button
          onClick={() => toggleOverlay('heatmap')}
          className={`text-[10px] font-mono font-bold uppercase px-2 py-1 rounded transition-all ${
            overlayState.heatmap ? 'bg-emerald-500 text-black shadow-[0_0_10px_rgba(16,185,129,0.5)]' : 'text-white/50 hover:text-emerald-400 hover:bg-emerald-500/10'
          }`}
        >
          H
        </button>
        <button
          onClick={() => toggleOverlay('trails')}
          className={`text-[10px] font-mono font-bold uppercase px-2 py-1 rounded transition-all ${
            overlayState.trails ? 'bg-emerald-500 text-black shadow-[0_0_10px_rgba(16,185,129,0.5)]' : 'text-white/50 hover:text-emerald-400 hover:bg-emerald-500/10'
          }`}
        >
          T
        </button>
        <button
          onClick={() => toggleOverlay('bboxes')}
          className={`text-[10px] font-mono font-bold uppercase px-2 py-1 rounded transition-all ${
            overlayState.bboxes ? 'bg-emerald-500 text-black shadow-[0_0_10px_rgba(16,185,129,0.5)]' : 'text-white/50 hover:text-emerald-400 hover:bg-emerald-500/10'
          }`}
        >
          B
        </button>
        {zoom > 1 && (
          <button
            onClick={resetView}
            className="text-[10px] font-mono font-bold uppercase px-2 py-1 rounded text-white/50 hover:text-emerald-400 hover:bg-emerald-500/10 transition-all ml-1 border-l border-white/10 pl-2"
          >
            {zoom.toFixed(1)}x
          </button>
        )}
      </div>

      {/* Video container with zoom/pan */}
      <div
        className="w-full h-full"
        onClick={handleZoomClick}
        onMouseDown={handleZoomPanStart}
      >
        {isWebRTCStream ? (
          <WebRTCVideo
            src={webrtcSrc}
            fallbackWebrtcSrc={webrtcFallbackSrc}
            fallbackSrc={fallbackSrc}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover opacity-100"
            style={videoStyle}
          />
        ) : isLiveStream ? (
          <HLSVideo
            src={hlsSrc}
            fallbackSrc={fallbackSrc}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover opacity-100"
            style={videoStyle}
          />
        ) : (
          <video
            src={fallbackSrc}
            autoPlay
            loop
            muted
            playsInline
            className="w-full h-full object-cover opacity-100"
            style={videoStyle}
          />
        )}
      </div>

      {/* Vertical Camera Analytics Overlay */}
      <CameraAnalytics scale={getAnalyticsScale(total)} camIndex={index} useCase={useCase} videoId={video.id} />

      {/* Video Label - Bottom Positioned */}
      <div className="absolute bottom-6 left-6 font-mono text-sm text-white/90 uppercase tracking-[0.5em] font-black drop-shadow-lg z-10 transition-all duration-300">
        {isLiveStream || isWebRTCStream ? 'LIVE' : `CAM ${index + 1}`}
        <div className={`w-8 h-0.5 mt-1 opacity-50 ${isLiveStream || isWebRTCStream ? 'bg-red-500' : 'bg-emerald-500'}`}></div>
      </div>

      {/* Zoom indicator */}
      {zoom > 1 && (
        <div className="absolute bottom-6 right-6 z-20 bg-black/60 backdrop-blur-sm border border-emerald-500/30 rounded px-2 py-1">
          <span className="text-emerald-400 font-mono text-xs font-bold">{zoom.toFixed(1)}x ZOOM</span>
        </div>
      )}
    </div>
  );
};

const Dashboard = () => {
  const { useCase } = useParams();
  const navigate = useNavigate();

  const themes = {
    traffic: { glow: 'rgba(6,182,212,0.15)', grid: 'rgba(0, 255, 255, 0.3)' },
    crowd: { glow: 'rgba(168,85,247,0.15)', grid: 'rgba(168, 85, 247, 0.3)' },
    safety: { glow: 'rgba(16,185,129,0.15)', grid: 'rgba(16, 185, 129, 0.3)' },
    perimeter: { glow: 'rgba(244,63,94,0.15)', grid: 'rgba(244, 63, 94, 0.3)' }
  };

  const theme = themes[useCase] || themes.traffic;

  // Video sources - WebRTC streams from MediaMTX (12 drones)
  const allVideos = [
    { id: 'bcpdrone1', type: 'webrtc', stream: 'bcpdrone1', processedStream: 'processed_bcpdrone1', overlayKey: 'bcpdrone1', droneIndex: 1, label: 'Drone 1' },
    { id: 'bcpdrone2', type: 'webrtc', stream: 'bcpdrone2', processedStream: 'processed_bcpdrone2', overlayKey: 'bcpdrone2', droneIndex: 2, label: 'Drone 2' },
    { id: 'bcpdrone3', type: 'webrtc', stream: 'bcpdrone3', processedStream: 'processed_bcpdrone3', overlayKey: 'bcpdrone3', droneIndex: 3, label: 'Drone 3' },
    { id: 'bcpdrone4', type: 'webrtc', stream: 'bcpdrone4', processedStream: 'processed_bcpdrone4', overlayKey: 'bcpdrone4', droneIndex: 4, label: 'Drone 4' },
    { id: 'bcpdrone5', type: 'webrtc', stream: 'bcpdrone5', processedStream: 'processed_bcpdrone5', overlayKey: 'bcpdrone5', droneIndex: 5, label: 'Drone 5' },
    { id: 'bcpdrone6', type: 'webrtc', stream: 'bcpdrone6', processedStream: 'processed_bcpdrone6', overlayKey: 'bcpdrone6', droneIndex: 6, label: 'Drone 6' },
    { id: 'bcpdrone7', type: 'webrtc', stream: 'bcpdrone7', processedStream: 'processed_bcpdrone7', overlayKey: 'bcpdrone7', droneIndex: 7, label: 'Drone 7' },
    { id: 'bcpdrone8', type: 'webrtc', stream: 'bcpdrone8', processedStream: 'processed_bcpdrone8', overlayKey: 'bcpdrone8', droneIndex: 8, label: 'Drone 8' },
    { id: 'bcpdrone9', type: 'webrtc', stream: 'bcpdrone9', processedStream: 'processed_bcpdrone9', overlayKey: 'bcpdrone9', droneIndex: 9, label: 'Drone 9' },
    { id: 'bcpdrone10', type: 'webrtc', stream: 'bcpdrone10', processedStream: 'processed_bcpdrone10', overlayKey: 'bcpdrone10', droneIndex: 10, label: 'Drone 10' },
    { id: 'bcpdrone11', type: 'webrtc', stream: 'bcpdrone11', processedStream: 'processed_bcpdrone11', overlayKey: 'bcpdrone11', droneIndex: 11, label: 'Drone 11' },
    { id: 'bcpdrone12', type: 'webrtc', stream: 'bcpdrone12', processedStream: 'processed_bcpdrone12', overlayKey: 'bcpdrone12', droneIndex: 12, label: 'Drone 12' },
  ];

  const [selectedVideos, setSelectedVideos] = useState([]);

  // Video layout state - position and size for each video
  const [videoLayouts, setVideoLayouts] = useState({});
  const [maxZIndex, setMaxZIndex] = useState(1);
  const containerRef = React.useRef(null);

  // Calculate optimal grid layout based on video count
  const calculateGridLayout = (count, containerWidth, containerHeight) => {
    const padding = 12;
    const gap = 12;
    let cols, rows;

    // Match the original grid layout logic
    switch (count) {
      case 1:
        cols = 1; rows = 1;
        break;
      case 2:
        cols = 2; rows = 1;
        break;
      case 3:
      case 4:
        cols = 2; rows = 2;
        break;
      case 5:
      case 6:
        cols = 3; rows = 2;
        break;
      default:
        cols = 3; rows = Math.ceil(count / 3);
    }

    const cellWidth = Math.floor((containerWidth - padding * 2 - gap * (cols - 1)) / cols);
    const cellHeight = Math.floor((containerHeight - padding * 2 - gap * (rows - 1)) / rows);

    return { cols, rows, cellWidth, cellHeight, padding, gap };
  };

  // Function to apply grid layout
  const applyGridLayout = () => {
    if (!containerRef.current || selectedVideos.length === 0) return;

    const container = containerRef.current.getBoundingClientRect();
    if (container.width === 0 || container.height === 0) return;

    const { cols, cellWidth, cellHeight, padding, gap } = calculateGridLayout(
      selectedVideos.length,
      container.width,
      container.height
    );

    const newLayouts = {};
    selectedVideos.forEach((video, idx) => {
      const col = idx % cols;
      const row = Math.floor(idx / cols);
      newLayouts[video.id] = {
        x: padding + col * (cellWidth + gap),
        y: padding + row * (cellHeight + gap),
        width: cellWidth,
        height: cellHeight,
        zIndex: idx + 1
      };
    });

    setVideoLayouts(newLayouts);
    setMaxZIndex(selectedVideos.length);
  };

  // Initialize layout when videos change
  useEffect(() => {
    // Small delay to ensure container is rendered
    const timer = setTimeout(applyGridLayout, 100);
    return () => clearTimeout(timer);
  }, [selectedVideos]);

  // Handle window resize - auto arrange on resize
  useEffect(() => {
    let resizeTimer;
    const handleResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(applyGridLayout, 200);
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      clearTimeout(resizeTimer);
    };
  }, [selectedVideos]);

  // Handle video selection changes - start/stop backend processing
  const handleVideosChange = (newSelectedVideos) => {
    const oldIds = new Set(selectedVideos.map(v => v.id));
    const newIds = new Set(newSelectedVideos.map(v => v.id));

    // Start processing for newly selected drones
    newSelectedVideos.forEach(video => {
      if (!oldIds.has(video.id) && video.droneIndex) {
        startDroneProcessing(video.droneIndex);
      }
    });

    // Stop processing for deselected drones
    selectedVideos.forEach(video => {
      if (!newIds.has(video.id) && video.droneIndex) {
        stopDroneProcessing(video.droneIndex);
      }
    });

    setSelectedVideos(newSelectedVideos);
  };

  const handlePositionChange = (videoId, newPosition) => {
    setVideoLayouts(prev => ({
      ...prev,
      [videoId]: { ...prev[videoId], ...newPosition }
    }));
  };

  const handleSizeChange = (videoId, newSize) => {
    setVideoLayouts(prev => ({
      ...prev,
      [videoId]: { ...prev[videoId], ...newSize }
    }));
  };

  const handleBringToFront = (videoId) => {
    setMaxZIndex(prev => prev + 1);
    setVideoLayouts(prev => ({
      ...prev,
      [videoId]: { ...prev[videoId], zIndex: maxZIndex + 1 }
    }));
  };

  const getAnalyticsScale = (count) => {
    if (count <= 1) return 1.0;
    if (count === 2) return 0.8;
    if (count <= 4) return 0.6;
    return 0.5;
  };

  // Auto-arrange videos in grid (resets to default layout)
  const autoArrange = () => {
    if (!containerRef.current || selectedVideos.length === 0) return;
    const container = containerRef.current.getBoundingClientRect();

    const { cols, cellWidth, cellHeight, padding, gap } = calculateGridLayout(
      selectedVideos.length,
      container.width,
      container.height
    );

    const newLayouts = {};
    selectedVideos.forEach((video, idx) => {
      const col = idx % cols;
      const row = Math.floor(idx / cols);
      newLayouts[video.id] = {
        x: padding + col * (cellWidth + gap),
        y: padding + row * (cellHeight + gap),
        width: cellWidth,
        height: cellHeight,
        zIndex: idx + 1
      };
    });
    setVideoLayouts(newLayouts);
    setMaxZIndex(selectedVideos.length);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex-1 flex"
    >
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col relative overflow-hidden">
        {/* Header */}
        <Header onReset={() => navigate('/')} useCase={useCase} />

        {/* Main Video Area - Free-form Layout */}
        <div ref={containerRef} className="flex-1 relative overflow-hidden">
          <div className="absolute inset-0 z-0 opacity-40 pointer-events-none"
            style={{
              backgroundImage: `linear-gradient(${theme.grid} 1px, transparent 1px), linear-gradient(90deg, ${theme.grid} 1px, transparent 1px)`,
              backgroundSize: '40px 40px'
            }}>
          </div>
          <div className="absolute inset-0 z-0 opacity-10 pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,#000_100%)]"></div>
          <div className="absolute inset-0 z-0 pointer-events-none"
            style={{
              background: `radial-gradient(circle_at_center, transparent 40%, ${theme.glow} 100%)`
            }}
          ></div>

          {/* Auto-arrange button */}
          {selectedVideos.length > 1 && (
            <button
              onClick={autoArrange}
              className="absolute top-4 left-4 z-50 px-3 py-2 bg-black/60 border border-emerald-500/30 rounded-md backdrop-blur-sm text-emerald-400 text-xs font-mono font-bold hover:bg-emerald-500/20 transition-all"
            >
              AUTO ARRANGE
            </button>
          )}

          {/* Video cells with absolute positioning */}
          <div className="absolute inset-0">
            {selectedVideos.map((video, index) => {
              const layout = videoLayouts[video.id] || { x: 50 + index * 50, y: 50 + index * 50, width: 500, height: 350, zIndex: index + 1 };
              return (
                <VideoCell
                  key={video.id}
                  video={video}
                  index={index}
                  total={selectedVideos.length}
                  getAnalyticsScale={getAnalyticsScale}
                  useCase={useCase}
                  position={{ x: layout.x, y: layout.y }}
                  size={{ width: layout.width, height: layout.height }}
                  zIndex={layout.zIndex}
                  onPositionChange={(pos) => handlePositionChange(video.id, pos)}
                  onSizeChange={(size) => handleSizeChange(video.id, size)}
                  onBringToFront={() => handleBringToFront(video.id)}
                />
              );
            })}
          </div>

          <div className="absolute inset-0 z-0 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.6)_100%)] pointer-events-none"></div>
        </div>

        {/* Footer */}
        <Footer
          selectedVideos={selectedVideos}
          onVideosChange={handleVideosChange}
          videos={allVideos}
        />
      </div>

      <RightPanel useCase={useCase} />
    </motion.div>
  );
}

const AppContents = () => {
  const location = useLocation();
  return (
    <div className="relative w-screen h-screen bg-[#050a14] overflow-hidden selection:bg-emerald-500/30 font-mono flex">
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<WelcomeScreen key="welcome" />} />
          <Route path="/:useCase" element={<Dashboard key="dashboard" />} />
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
