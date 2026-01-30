import React, { useState, useEffect, useMemo } from 'react';
import clsx from 'clsx';
import { Activity, Globe, Wifi } from 'lucide-react';

const API_BASE_URL = import.meta.env.DEV
  ? '/api'
  : `http://${window.location.hostname}:9010`;

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

const Footer = ({ selectedVideos, onVideosChange, videos = [], onRefresh }) => {
  /* ============================
     CLOCK
  ============================ */
  const [currentTime, setCurrentTime] = useState(
    new Date().toLocaleTimeString()
  );

  useEffect(() => {
    const t = setInterval(
      () => setCurrentTime(new Date().toLocaleTimeString()),
      1000
    );
    return () => clearInterval(t);
  }, []);

  /* ============================
     METRICS (REAL API)
  ============================ */
  const [fps, setFps] = useState(null);
  const [latency, setLatency] = useState(null);
  const [bandwidth, setBandwidth] = useState(null);

  useEffect(() => {
    let timer;

    const pollMetrics = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (!res.ok) return;

        const data = await res.json();
        const ids = selectedVideos.map(v => v.id);

        let fpsSum = 0;
        let fpsCount = 0;
        let latencySum = 0;
        let bwSum = 0;
        let bwCount = 0;

        Object.entries(data).forEach(([id, m]) => {
          if (!ids.includes(id)) return;

          if (typeof m.fps === 'number') {
            fpsSum += m.fps;
            fpsCount++;
          }
          if (typeof m.latency_ms === 'number') {
            latencySum += m.latency_ms;
          }
          if (typeof m.bandwidth_mb === 'number') {
            bwSum += m.bandwidth_mb;
            bwCount++;
          }
        });

        setFps(fpsCount ? fpsSum / fpsCount : null);
        setLatency(fpsCount ? latencySum / fpsCount : null);
        setBandwidth(bwCount ? bwSum : null);
      } catch (e) {
        console.error('metrics fetch failed', e);
      }
    };

    pollMetrics();
    timer = setInterval(pollMetrics, 2000);
    return () => clearInterval(timer);
  }, [selectedVideos]);

  /* ============================
     AUTO REFRESH SOURCES
  ============================ */
  useEffect(() => {
    if (!onRefresh) return;
    const i = setInterval(onRefresh, 5000);
    return () => clearInterval(i);
  }, [onRefresh]);

  /* ============================
     CAMERA LABELS (OLD LOGIC)
  ============================ */
  const uploadedMap = useMemo(() => {
    const map = {};
    let c = 0;
    videos.forEach(v => {
      if (v.label === 'UPLOADED') {
        c++;
        map[v.id] = `UP ${c}`;
      }
    });
    return map;
  }, [videos]);

  const toggleVideo = async (video) => {
    const active = selectedVideos.some(v => v.id === video.id);
    if (active) {
      if (selectedVideos.length === 1) return;
      onVideosChange(selectedVideos.filter(v => v.id !== video.id));

      // Stop inference on backend
      try {
        await fetch(`${API_BASE_URL}/sources/stop`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ index: video.droneIndex }),
        });
      } catch (e) { console.error('Stop failed', e); }

    } else {
      onVideosChange([...selectedVideos, video]);

      // Start inference on backend
      try {
        await fetch(`${API_BASE_URL}/sources/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ index: video.droneIndex }),
        });
      } catch (e) { console.error('Start failed', e); }
    }
  };

  const isSelected = (id) =>
    selectedVideos.some(v => v.id === id);

  /* ============================
     UPLOAD LOGIC
  ============================ */
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadMessage, setUploadMessage] = useState(null);

  const uploadFile = async (file) => {
    if (!file) return;
    setIsUploading(true);
    setUploadProgress(0);
    setUploadMessage(null);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/upload?filename=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        setUploadProgress((e.loaded / e.total) * 100);
      }
    };

    xhr.onload = () => {
      setIsUploading(false);
      if (xhr.status === 200) {
        setUploadMessage({ type: 'success', text: 'UPLOADED' });
        setTimeout(() => setUploadMessage(null), 3000);
      } else {
        setUploadMessage({ type: 'error', text: 'FAILED' });
        setTimeout(() => setUploadMessage(null), 3000);
      }
    };

    xhr.onerror = () => {
      setIsUploading(false);
      setUploadMessage({ type: 'error', text: 'ERROR' });
      setTimeout(() => setUploadMessage(null), 3000);
    };

    xhr.send(file);
  };

  /* ============================
     RENDER
  ============================ */
  return (
    <div className="w-full h-12 bg-black/60 border-t border-cyan-500/30 flex items-center justify-between px-6 z-50 font-mono">
      {/* ===== LEFT: SYSTEM STATUS ===== */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
          <span className="text-[10px] text-emerald-500 font-bold uppercase tracking-[0.2em]">
            Network: Online
          </span>
        </div>

        <div className="flex items-center gap-2 border-l border-white/10 pl-6">
          <Globe className="w-3 h-3 text-cyan-500/60" />
          <span className="text-[10px] text-cyan-500/60 uppercase tracking-widest">
            Nodes: {selectedVideos.length}/{videos.length.toString().padStart(2, '0')}
          </span>
        </div>

        <div className="flex items-center gap-2 border-l border-white/10 pl-6">
          <Activity className="w-3 h-3 text-cyan-500/60" />
          <span className="text-[10px] text-cyan-500/60 uppercase tracking-widest">
            Feed: Active
          </span>
        </div>
      </div>

      {/* ===== CENTER: CAMERA SELECTOR & UPLOADER ===== */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1 border border-white/10 px-1 py-0.5">
          {videos.map((vid, idx) => {
            const label =
              vid.label === 'UPLOADED'
                ? uploadedMap[vid.id] || 'UP'
                : vid.label || `CAM ${idx + 1}`;

            return (
              <button
                key={vid.id}
                onClick={() => toggleVideo(vid)}
                className={clsx(
                  'px-3 py-1 text-[10px] uppercase tracking-wider font-bold transition-colors',
                  isSelected(vid.id)
                    ? 'bg-cyan-500 text-black'
                    : 'text-white/40 hover:text-white/70'
                )}
                title={DRONE_REGION_MAP[vid.id] || label}
              >
                {label}
              </button>
            );
          })}
        </div>

        {/* Tactical Uploader Small UI */}
        <div className="relative group">
          <input
            type="file"
            accept=".mp4,.mkv,.asf"
            className="absolute inset-0 opacity-0 cursor-pointer z-10"
            onChange={(e) => uploadFile(e.target.files[0])}
            disabled={isUploading}
          />
          <div className={clsx(
            "px-4 py-1 border border-cyan-500/30 text-[10px] font-black uppercase tracking-[0.2em] transition-all duration-300 flex items-center gap-2",
            isUploading ? "bg-cyan-500/20" : "hover:bg-cyan-500 hover:text-black cursor-pointer shadow-[0_0_10px_rgba(6,182,212,0.1)] hover:shadow-[0_0_15px_rgba(6,182,212,0.3)]"
          )}>
            {isUploading ? (
              <div className="flex items-center gap-2">
                <div className="w-12 h-1 bg-white/10 overflow-hidden relative">
                  <div className="absolute inset-0 bg-cyan-400" style={{ width: `${uploadProgress}%` }} />
                </div>
                <span className="text-cyan-400">{Math.round(uploadProgress)}%</span>
              </div>
            ) : uploadMessage ? (
              <span className={uploadMessage.type === 'success' ? 'text-emerald-500' : 'text-red-500'}>
                {uploadMessage.text}
              </span>
            ) : (
              <span>Upload_Feed</span>
            )}
          </div>
        </div>
      </div>

      {/* ===== RIGHT: DIAGNOSTICS ===== */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <Activity className="w-3 h-3 text-cyan-500/60" />
          <span className="text-[10px] text-cyan-500/80">
            {fps !== null ? fps.toFixed(0) : '--'} FPS
          </span>
        </div>

        <div className="flex items-center gap-2 border-l border-white/10 pl-6">
          <span className="text-[10px] text-white/30 uppercase tracking-widest">
            Latency: {latency !== null ? latency.toFixed(0) : '--'}ms
          </span>
        </div>

        <div className="flex items-center gap-2 border-l border-white/10 pl-6">
          <span className="text-[10px] text-cyan-500/80">
            {currentTime}
          </span>
        </div>

        <div className="flex items-center gap-2 border-l border-white/10 pl-6">
          <Wifi className="w-3 h-3 text-white/20" />
          <span className="text-[10px] text-white/20">
            {bandwidth !== null ? bandwidth.toFixed(1) : '--'} MB/s
          </span>
        </div>
      </div>
    </div>
  );
};

export default Footer;
