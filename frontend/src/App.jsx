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
import Login from './Login';
import LeftPanel from './components/Dashboard/LeftPanel';

// MediaMTX stream configuration
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

const CameraAnalytics = ({ scale = 1, camIndex = 0, useCase, videoId }) => {
  const [metrics, setMetrics] = useState(null);
  const [hasReceivedData, setHasReceivedData] = useState(false);

  const sourceName = videoId;

  useEffect(() => {
    if (!sourceName) return;

    const fetchMetrics = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        if (response.ok) {
          const data = await response.json();
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

  useEffect(() => {
    setHasReceivedData(false);
    setMetrics(null);
  }, [videoId]);

  const congestion = metrics?.congestion_index || 0;
  const speed = Math.round(metrics?.mobility_index || 0);
  const density = metrics?.traffic_density || 0;
  const detections = metrics?.detection_count || 0;
  const stalled = metrics?.stalled_pct || 0;
  const slow = metrics?.slow_pct || 0;
  const medium = metrics?.medium_pct || 0;
  const fast = metrics?.fast_pct || 0;
  const fps = metrics?.fps || 0;

  let status = 'SMOOTH';
  let color = 'text-cyan-400';
  let borderColor = 'border-cyan-400/50';
  if (congestion > 75) { status = 'HEAVY'; color = 'text-red-500'; borderColor = 'border-red-500/50'; }
  else if (congestion > 50) { status = 'SLOW'; color = 'text-yellow-500'; borderColor = 'border-yellow-500/50'; }
  else if (congestion > 25) { status = 'MODERATE'; color = 'text-cyan-400'; borderColor = 'border-cyan-400/50'; }

  const nodeNames = ['IRIS-ZONE-NORTH', 'IRIS-ZONE-SOUTH', 'IRIS-ZONE-EAST', 'IRIS-ZONE-WEST'];
  const nodeName = nodeNames[camIndex % nodeNames.length] || 'IRIS-ZONE-01';
  const lat = (12.9716 + (Math.random() * 0.01 - 0.005)).toFixed(4);
  const lng = (77.5946 + (Math.random() * 0.01 - 0.005)).toFixed(4);

  const configs = {
    traffic: { primary: 'MOBILITY', secondary: 'CONGESTION', unit: '%', statusLabel: 'Traffic' },
    crowd: { primary: 'DENSITY', secondary: 'FLOW RATE', unit: 'PPL/M²', statusLabel: 'Crowd' },
    safety: { primary: 'RISK LVL', secondary: 'ALERT CONF', unit: '%', statusLabel: 'Safety' },
    perimeter: { primary: 'PROXIMITY', secondary: 'SIGNAL', unit: 'M', statusLabel: 'Perimeter' }
  };

  const config = configs[useCase] || configs.traffic;

  if (!hasReceivedData && !metrics) return null;

  return (
    <>
      <div
        className="absolute left-4 top-1/2 -translate-y-1/2 flex flex-col gap-4 pointer-events-none origin-left"
        style={{ transform: `scale(${scale})` }}
      >
        <div className="flex items-center gap-4">
          <div className="relative h-64 w-20 overflow-hidden border-l-2 border-white/10 bg-black/20 backdrop-blur-sm">
            <div className="absolute top-1/2 left-0 w-full h-10 -translate-y-1/2 bg-white border-l-4 border-cyan-500 z-20 flex items-center pl-2 shadow-[0_0_20px_rgba(255,255,255,0.2)]">
              <span className="text-2xl font-black text-black font-mono leading-none tracking-tighter">
                <CountUp end={speed} duration={0.4} preserveValue={true} />
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
            <div className="text-[10px] text-white/40 font-mono font-bold tracking-widest leading-none">{config.primary}_DATA_LINK</div>
            <div className="text-xl text-white/40 font-black mt-1">{config.unit}</div>
          </div>
        </div>

        {/* Real-time Sub-Metrics Block */}
        <div className="flex flex-col gap-4 font-mono text-[12px] tracking-[0.2em]">
          <div className="flex items-center">
            <div className="w-1.5 h-12 bg-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.6)]" />
            <div className="flex justify-between items-center w-64 bg-black/60 backdrop-blur-md px-4 py-2 border-r border-white/10">
              <span className="text-white/40 uppercase font-black">Vehicles</span>
              <span className="text-cyan-400 font-bold text-2xl">
                <CountUp end={detections} duration={0.4} preserveValue={true} />
              </span>
            </div>
          </div>
          <div className="flex items-center">
            <div className="w-1.5 h-12 bg-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.6)]" />
            <div className="flex justify-between items-center w-64 bg-black/60 backdrop-blur-md px-4 py-2 border-r border-white/10">
              <span className="text-white/40 uppercase font-black">Density</span>
              <span className="text-cyan-400 font-bold text-2xl">
                <CountUp end={density} duration={0.4} preserveValue={true} />%
              </span>
            </div>
          </div>
          <div className="flex items-center">
            <div className="w-1.5 h-12 bg-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.6)]" />
            <div className="flex justify-between items-center w-64 bg-black/60 backdrop-blur-md px-4 py-2 border-r border-white/10">
              <span className="text-white/40 uppercase font-black">FPS Output</span>
              <span className="text-cyan-400 font-bold text-2xl">
                <CountUp end={fps} duration={0.4} preserveValue={true} />
              </span>
            </div>
          </div>
        </div>
      </div>

      <div
        className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-3 pointer-events-none origin-right"
        style={{ transform: `scale(${scale})` }}
      >
        <div className="flex flex-col items-end text-right gap-1 order-1">
          <div className="text-lg text-white/90 uppercase tracking-[0.6em] font-black">{config.secondary}</div>
          <div className="text-[10px] text-white/40 font-mono font-bold uppercase">Dynamic_Grid_LVL</div>
          <div className="text-7xl font-black font-mono leading-none text-white/90 mt-2 drop-shadow-[0_0_20px_rgba(255,255,255,0.2)]">
            <CountUp end={congestion} duration={0.5} preserveValue={true} />%
          </div>
        </div>

        <div className="relative h-64 w-3 bg-white/5 overflow-hidden border-r border-white/10 flex flex-col order-2">
          <div className="flex-1 w-full relative bg-red-500/5 border-b border-white/5">
            <motion.div
              className="absolute bottom-0 w-full bg-red-500 shadow-[0_0_15px_rgba(239,68,68,0.4)]"
              animate={{ height: `${stalled}%` }}
              transition={{ type: "spring", stiffness: 50 }}
            />
          </div>
          <div className="flex-1 w-full relative bg-orange-500/5 border-b border-white/5">
            <motion.div
              className="absolute bottom-0 w-full bg-orange-500 shadow-[0_0_15px_rgba(249,115,22,0.4)]"
              animate={{ height: `${slow}%` }}
              transition={{ type: "spring", stiffness: 50 }}
            />
          </div>
          <div className="flex-1 w-full relative bg-yellow-500/5 border-b border-white/5">
            <motion.div
              className="absolute bottom-0 w-full bg-yellow-500 shadow-[0_0_15px_rgba(234,179,8,0.4)]"
              animate={{ height: `${medium}%` }}
              transition={{ type: "spring", stiffness: 50 }}
            />
          </div>
          <div className="flex-1 w-full relative bg-emerald-500/5">
            <motion.div
              className="absolute bottom-0 w-full bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.4)]"
              animate={{ height: `${fast}%` }}
              transition={{ type: "spring", stiffness: 50 }}
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
  const [overlays, setOverlays] = useState({ heatmap: true, trails: true, bboxes: true });
  const [isToggling, setIsToggling] = useState(null);

  useEffect(() => {
    const timer = setTimeout(() => setIsCellLoading(false), 2500);
    return () => clearTimeout(timer);
  }, []);

  // Fetch overlay state on mount and periodically sync
  useEffect(() => {
    const fetchOverlay = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/overlays/${video.id}`);
        if (res.ok) {
          const data = await res.json();
          setOverlays(data);
        }
      } catch (e) { }
    };
    fetchOverlay();
    // Sync every 2 seconds to stay in sync with backend
    const interval = setInterval(fetchOverlay, 2000);
    return () => clearInterval(interval);
  }, [video.id]);

  const toggleOverlay = async (type) => {
    const nextValue = !overlays[type];
    const updated = { ...overlays, [type]: nextValue };
    setOverlays(updated);
    setIsToggling(type);

    try {
      const res = await fetch(`${API_BASE_URL}/overlays/${video.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [type]: nextValue }),
      });
      if (res.ok) {
        const data = await res.json();
        setOverlays(data); // Use server response as source of truth
      }
    } catch (e) {
      console.error('Failed to update overlay:', e);
      // Revert on error
      setOverlays(overlays);
    } finally {
      setTimeout(() => setIsToggling(null), 200);
    }
  };

  const isLiveStream = video.type === 'hls';
  const isWebRTCStream = video.type === 'webrtc';
  const isUploadStream = video.type === 'upload';
  const hlsSrc = isLiveStream ? `${HLS_BASE_URL}/${video.stream}/index.m3u8` : null;
  const webrtcSrc = isWebRTCStream ? `${WEBRTC_BASE_URL}/${video.processedStream || video.stream}/whep` : null;
  const fallbackSrc = video.fallback || (video.type === 'static' ? `/${video.id}` : null);

  const maskStyle = {
    maskImage: 'linear-gradient(to right, transparent 0%, black 12%, black 88%, transparent 100%), linear-gradient(to bottom, transparent 0%, black 12%, black 88%, transparent 100%)',
    maskComposite: 'intersect',
    WebkitMaskComposite: 'source-in'
  };

  return (
    <div className={`relative overflow-hidden group ${getVideoClass(index, total)}`}>
      <AnimatePresence>{isCellLoading && <IRISLoader />}</AnimatePresence>

      {/* Tactical Cyberpunk Brackets */}
      <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-cyan-500/50 z-20 pointer-events-none" />
      <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-cyan-500/50 z-20 pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-cyan-500/50 z-20 pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-cyan-500/50 z-20 pointer-events-none" />

      {/* Overlay Toggles (Top Right) */}
      <div className="absolute top-6 right-8 z-30 flex gap-2 pointer-events-auto">
        {[
          { id: 'heatmap', label: 'HEAT' },
          { id: 'trails', label: 'TRAIL' },
          { id: 'bboxes', label: 'BOX' }
        ].map((btn) => (
          <button
            key={btn.id}
            onClick={() => toggleOverlay(btn.id)}
            disabled={isToggling === btn.id}
            className={`px-3 py-1 text-[9px] font-black uppercase tracking-widest border transition-all duration-200 ${
              isToggling === btn.id
                ? 'bg-white/20 text-white/50 border-white/30 animate-pulse'
                : overlays[btn.id]
                  ? 'bg-cyan-500 text-black border-cyan-500 shadow-[0_0_12px_rgba(6,182,212,0.5)] hover:bg-cyan-400'
                  : 'bg-black/70 text-white/40 border-white/20 hover:border-cyan-500/50 hover:text-white/60'
            }`}
          >
            {btn.label}
          </button>
        ))}
      </div>

      {/* Inner corner accents */}
      <div className="absolute top-4 left-4 w-2 h-2 border-t border-l border-white/20 z-20 pointer-events-none" />
      <div className="absolute top-4 right-4 w-2 h-2 border-t border-r border-white/20 z-20 pointer-events-none" />
      <div className="absolute bottom-4 left-4 w-2 h-2 border-b border-l border-white/20 z-20 pointer-events-none" />
      <div className="absolute bottom-4 right-4 w-2 h-2 border-b border-r border-white/20 z-20 pointer-events-none" />

      {(isWebRTCStream || isUploadStream) ? (
        <img
          src={`${API_BASE_URL}/stream/${video.id}`}
          className="w-full h-full object-cover"
          style={maskStyle}
          onError={(e) => {
            if (fallbackSrc) e.target.src = fallbackSrc;
          }}
        />
      ) : isLiveStream ? (
        <HLSVideo src={hlsSrc} fallbackSrc={fallbackSrc} className="w-full h-full object-cover" autoPlay muted playsInline style={maskStyle} />
      ) : (
        <video src={fallbackSrc} autoPlay loop muted playsInline className="w-full h-full object-cover opacity-100" style={maskStyle} />
      )}

      {/* Vertical Camera Analytics Overlay */}
      <CameraAnalytics scale={getAnalyticsScale(total)} camIndex={index} useCase={useCase} videoId={video.id} />


    </div>
  );
};

const Dashboard = ({ onLogout }) => {
  const { useCase } = useParams();
  const navigate = useNavigate();

  const themes = {
    traffic: { glow: 'rgba(6,182,212,0.15)', grid: 'rgba(0, 255, 255, 0.3)' },
    crowd: { glow: 'rgba(168,85,247,0.15)', grid: 'rgba(168, 85, 247, 0.3)' },
    safety: { glow: 'rgba(16,185,129,0.15)', grid: 'rgba(16, 185, 129, 0.3)' },
    perimeter: { glow: 'rgba(244,63,94,0.15)', grid: 'rgba(244, 63, 94, 0.3)' }
  };

  const theme = themes[useCase] || themes.traffic;

  const STATIC_DRONES = [
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

  const [allVideos, setAllVideos] = useState(STATIC_DRONES);
  const [selectedVideos, setSelectedVideos] = useState([]);

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
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 flex">
      <div className="flex-1 flex flex-col relative overflow-hidden">
        <Header onReset={() => navigate('/')} useCase={useCase} onLogout={onLogout} />
        <LeftPanel />

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
            <div className={`w-full h-full grid gap-6 p-8 ${getGridLayout(selectedVideos.length)}`}>
              {selectedVideos.map((video, index) => (
                <VideoCell
                  key={video.id}
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

        <Footer selectedVideos={selectedVideos} onVideosChange={setSelectedVideos} videos={allVideos} />
      </div>

      <RightPanel useCase={useCase} sources={allVideos} />
    </motion.div>
  );
}

const AppContents = () => {
  // Check localStorage for existing session
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

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="relative w-screen h-screen bg-[#050a14] overflow-hidden selection:bg-cyan-500/30 font-mono flex">
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<WelcomeScreen key="welcome" />} />
          <Route path="/:useCase" element={<Dashboard key="dashboard" onLogout={handleLogout} />} />
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
