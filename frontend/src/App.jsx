import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrowserRouter, Routes, Route, useParams, useNavigate, useLocation } from 'react-router-dom';

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

const CameraAnalytics = ({ scale = 1, camIndex = 0, useCase }) => {
  const [congestion, setCongestion] = useState(Math.random() * 80 + 10);

  useEffect(() => {
    const interval = setInterval(() => {
      setCongestion(c => Math.min(100, Math.max(0, c + (Math.random() * 10 - 5))));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const speed = Math.round(100 - congestion);
  let status = 'SMOOTH';
  let color = 'text-emerald-500';
  let borderColor = 'border-emerald-500/50';
  if (congestion > 75) { status = 'HEAVY'; color = 'text-red-500'; borderColor = 'border-red-500/50'; }
  else if (congestion > 50) { status = 'SLOW'; color = 'text-yellow-500'; borderColor = 'border-yellow-500/50'; }
  else if (congestion > 25) { status = 'MODERATE'; color = 'text-cyan-400'; borderColor = 'border-cyan-400/50'; }

  // Technical Metadata
  const nodeNames = ['IRIS-ZONE-NORTH', 'IRIS-ZONE-SOUTH', 'IRIS-ZONE-EAST', 'IRIS-ZONE-WEST'];
  const nodeName = nodeNames[camIndex % nodeNames.length] || 'IRIS-ZONE-01';
  const lat = (12.9716 + (Math.random() * 0.01 - 0.005)).toFixed(4);
  const lng = (77.5946 + (Math.random() * 0.01 - 0.005)).toFixed(4);

  // Labels based on Use Case
  const configs = {
    traffic: {
      primary: 'VELOCITY',
      secondary: 'CONGESTION',
      unit: 'KM/H',
      statusLabel: 'Traffic'
    },
    crowd: {
      primary: 'DENSITY',
      secondary: 'FLOW RATE',
      unit: 'PPL/M²',
      statusLabel: 'Crowd'
    },
    safety: {
      primary: 'RISK LVL',
      secondary: 'ALERT CONF',
      unit: '%',
      statusLabel: 'Safety'
    },
    perimeter: {
      primary: 'PROXIMITY',
      secondary: 'SIGNAL',
      unit: 'M',
      statusLabel: 'Perimeter'
    }
  };

  const config = configs[useCase] || configs.traffic;

  return (
    <>
      {/* Left-side Analytics Stack */}
      <div
        className="absolute left-6 top-1/2 -translate-y-1/2 flex flex-col gap-8 pointer-events-none origin-left"
        style={{ transform: `scale(${scale})` }}
      >
        {/* Jet-style Airspeed Tape Indicator */}
        <div className="flex items-center gap-4">
          <div className="relative h-64 w-20 overflow-hidden border-l-2 border-white/10 bg-black/20 backdrop-blur-sm">
            <div className="absolute top-1/2 left-0 w-full h-10 -translate-y-1/2 bg-white border-l-4 border-cyan-500 z-20 flex items-center pl-2 shadow-[0_0_20px_rgba(255,255,255,0.2)]">
              <span className="text-2xl font-black text-black font-mono leading-none tracking-tighter">
                {speed}
              </span>
              <div className="absolute right-[-4px] w-0 h-0 border-t-[5px] border-t-transparent border-b-[5px] border-b-transparent border-l-[8px] border-l-white"></div>
            </div>

            <motion.div
              animate={{ y: speed * 4 }}
              transition={{ type: "spring", stiffness: 100, damping: 20 }}
              className="absolute bottom-1/2 left-0 w-full flex flex-col-reverse items-end pr-3"
            >
              {[...Array(25)].map((_, i) => {
                const val = i * 5;
                const isMajor = val % 10 === 0;
                return (
                  <div key={val} className="h-[20px] flex items-center justify-end gap-2 shrink-0">
                    {isMajor && (
                      <span className="text-[10px] font-mono font-bold text-white/40">
                        {val.toString().padStart(3, '0')}
                      </span>
                    )}
                    <div className={`${isMajor ? 'w-4 h-[2px] bg-white/40' : 'w-2 h-[1px] bg-white/20'}`} />
                  </div>
                );
              })}
            </motion.div>

            <div className="absolute top-0 left-0 w-full h-12 bg-gradient-to-b from-[#050a14] to-transparent z-10 opacity-80"></div>
            <div className="absolute bottom-0 left-0 w-full h-12 bg-gradient-to-t from-[#050a14] to-transparent z-10 opacity-80"></div>
          </div>

          <div className="flex flex-col gap-1">
            <div className="text-lg text-white/90 uppercase tracking-[0.6em] font-black">{config.primary}</div>
            <div className="text-[10px] text-white/40 font-mono font-bold">{config.primary}_TAPE_ACTV</div>
            <div className="text-xl text-white/40 font-black mt-1">{config.unit}</div>
          </div>
        </div>

        <div className="mt-4 font-mono text-[10px] text-cyan-500/40 flex flex-col gap-1 tracking-tighter">
          <div className="flex gap-4">
            <span className="text-cyan-500/60 font-black">NODE:</span>
            <span className="text-white/60">{nodeName}</span>
          </div>
          <div className="flex gap-4">
            <span className="text-cyan-500/60 font-black">COORD:</span>
            <span className="text-white/60">{lat}°N / {lng}°E</span>
          </div>
          <div className="flex gap-4">
            <span className="text-cyan-500/60 font-black">METRICS:</span>
            <span className="text-white/60">RT_PROC_ACTIVE_10b</span>
          </div>
        </div>
      </div>

      <div
        className="absolute right-6 top-1/2 -translate-y-1/2 flex items-center gap-4 pointer-events-none origin-right"
        style={{ transform: `scale(${scale})` }}
      >
        <div className="flex flex-col items-end text-right gap-1 order-1">
          <div className="text-lg text-white/90 uppercase tracking-[0.6em] font-black">{config.secondary}</div>
          <div className="text-[10px] text-white/40 font-mono font-bold uppercase">Dynamic_Grid_LVL</div>
          <div className="text-7xl font-black font-mono leading-none text-white/90 mt-2 drop-shadow-[0_0_20px_rgba(255,255,255,0.2)]">
            {Math.round(congestion)}%
          </div>
        </div>

        <div className="relative h-64 w-3 bg-white/10 overflow-hidden border-r-2 border-white/10 flex flex-col order-2">
          <div className="flex-1 w-full relative bg-red-500/10 border-b border-white/5">
            <motion.div
              className="absolute bottom-0 w-full bg-red-500 shadow-[0_0_15px_rgba(239,68,68,0.5)]"
              animate={{ height: `${Math.min(100, Math.max(0, (congestion - 75) * 4))}%` }}
            />
          </div>
          <div className="flex-1 w-full relative bg-orange-500/10 border-b border-white/5">
            <motion.div
              className="absolute bottom-0 w-full bg-orange-500 shadow-[0_0_15px_rgba(249,115,22,0.5)]"
              animate={{ height: `${Math.min(100, Math.max(0, (congestion - 50) * 4))}%` }}
            />
          </div>
          <div className="flex-1 w-full relative bg-yellow-500/10 border-b border-white/5">
            <motion.div
              className="absolute bottom-0 w-full bg-yellow-500 shadow-[0_0_15px_rgba(234,179,8,0.5)]"
              animate={{ height: `${Math.min(100, Math.max(0, (congestion - 25) * 4))}%` }}
            />
          </div>
          <div className="flex-1 w-full relative bg-emerald-500/10">
            <motion.div
              className="absolute bottom-0 w-full bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)]"
              animate={{ height: `${Math.min(100, Math.max(0, (congestion - 0) * 4))}%` }}
            />
          </div>
        </div>
      </div>

      <div
        className="absolute bottom-6 right-6 pointer-events-none origin-right z-10"
        style={{ transform: `scale(${scale})` }}
      >
        <div className={`px-4 py-2 border-r-[6px] ${borderColor} bg-black/50 backdrop-blur-md transform skew-x-[-12deg]`}>
          <div className={`text-sm font-black uppercase tracking-[0.2em] ${color} skew-x-[12deg]`}>
            {config.statusLabel}: {status}
          </div>
        </div>
      </div>
    </>
  );
};

const VideoCell = ({ video, index, total, getAnalyticsScale, getVideoClass, useCase }) => {
  const [isCellLoading, setIsCellLoading] = useState(true);
  const [overlayState, setOverlayState] = useState({
    heatmap: false,
    trails: false,
    bboxes: false,
  });

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

  return (
    <div
      className={`relative overflow-hidden group border border-white/5 ${getVideoClass(index, total)}`}
    >
      <AnimatePresence>
        {isCellLoading && <IRISLoader />}
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

      {/* Overlay toggles */}
      <div className="absolute top-4 right-4 z-30 flex items-center gap-2 bg-black/40 border border-white/10 rounded-md px-2 py-1">
        <button
          onClick={() => toggleOverlay('heatmap')}
          className={`text-[10px] font-mono font-bold uppercase px-2 py-1 rounded ${
            overlayState.heatmap ? 'bg-cyan-500 text-black' : 'text-white/50 hover:text-white'
          }`}
        >
          H
        </button>
        <button
          onClick={() => toggleOverlay('trails')}
          className={`text-[10px] font-mono font-bold uppercase px-2 py-1 rounded ${
            overlayState.trails ? 'bg-cyan-500 text-black' : 'text-white/50 hover:text-white'
          }`}
        >
          T
        </button>
        <button
          onClick={() => toggleOverlay('bboxes')}
          className={`text-[10px] font-mono font-bold uppercase px-2 py-1 rounded ${
            overlayState.bboxes ? 'bg-cyan-500 text-black' : 'text-white/50 hover:text-white'
          }`}
        >
          B
        </button>
      </div>

      {isWebRTCStream ? (
        <WebRTCVideo
          src={webrtcSrc}
          fallbackWebrtcSrc={webrtcFallbackSrc}
          fallbackSrc={fallbackSrc}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover opacity-100"
          style={{
            maskImage: 'linear-gradient(to right, transparent 0%, black 15%, black 85%, transparent 100%), linear-gradient(to bottom, transparent 0%, black 15%, black 85%, transparent 100%)',
            maskComposite: 'intersect',
            WebkitMaskComposite: 'source-in'
          }}
        />
      ) : isLiveStream ? (
        <HLSVideo
          src={hlsSrc}
          fallbackSrc={fallbackSrc}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover opacity-100"
          style={{
            maskImage: 'linear-gradient(to right, transparent 0%, black 15%, black 85%, transparent 100%), linear-gradient(to bottom, transparent 0%, black 15%, black 85%, transparent 100%)',
            maskComposite: 'intersect',
            WebkitMaskComposite: 'source-in'
          }}
        />
      ) : (
        <video
          src={fallbackSrc}
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover opacity-100"
          style={{
            maskImage: 'linear-gradient(to right, transparent 0%, black 15%, black 85%, transparent 100%), linear-gradient(to bottom, transparent 0%, black 15%, black 85%, transparent 100%)',
            maskComposite: 'intersect',
            WebkitMaskComposite: 'source-in'
          }}
        />
      )}

      {/* Vertical Camera Analytics Overlay */}
      <CameraAnalytics scale={getAnalyticsScale(total)} camIndex={index} useCase={useCase} />

      {/* Video Label - Bottom Positioned */}
      <div className="absolute bottom-6 left-6 font-mono text-sm text-white/90 uppercase tracking-[0.5em] font-black drop-shadow-lg z-10 transition-all duration-300">
        {isLiveStream || isWebRTCStream ? 'LIVE' : `CAM ${index + 1}`}
        <div className={`w-8 h-0.5 mt-1 opacity-50 ${isLiveStream || isWebRTCStream ? 'bg-red-500' : 'bg-cyan-500'}`}></div>
      </div>
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

  const getAnalyticsScale = (count) => {
    if (count <= 1) return 1.2;
    if (count === 2) return 0.85;
    if (count <= 4) return 0.65;
    return 0.5;
  };

  const getVideoClass = (index, total) => {
    if (index === 0) {
      if (total === 3) return 'col-span-2';
      if (total === 5) return 'col-span-2';
      if (total === 7) return 'col-span-3';
    }
    return '';
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
        <Header onReset={() => navigate('/')} useCase={useCase} />

        <div className="flex-1 relative overflow-hidden">
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

          <div className="absolute inset-0 z-0 flex items-center justify-center p-4">
            <div className={`w-full h-full grid gap-2 ${getGridLayout(selectedVideos.length)}`}>
              {selectedVideos.map((video, index) => (
                <VideoCell
                  key={`${video.id}-${index}`}
                  video={video}
                  index={index}
                  total={selectedVideos.length}
                  getAnalyticsScale={getAnalyticsScale}
                  getVideoClass={getVideoClass}
                  useCase={useCase}
                />
              ))}
            </div>
          </div>

          <div className="absolute inset-0 z-0 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.6)_100%)] pointer-events-none"></div>
        </div>

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
    <div className="relative w-screen h-screen bg-[#050a14] overflow-hidden selection:bg-cyan-500/30 font-mono flex">
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
