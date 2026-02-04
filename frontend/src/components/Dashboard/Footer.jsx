import React, { useState, useEffect, useMemo } from 'react';
import clsx from 'clsx';
import { Activity, Globe, Upload } from 'lucide-react';
import { API_BASE_URL } from '../../config';

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

const Footer = ({ selectedVideos, onVideosChange, videos = [], onRefresh, useCase }) => {
  /* ── METRICS ── */
  const [fps, setFps] = useState(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (!res.ok) return;
        const data = await res.json();
        const ids = selectedVideos.map(v => v.id);
        let sum = 0, cnt = 0;
        Object.entries(data).forEach(([id, m]) => {
          if (!ids.includes(id)) return;
          if (typeof m.fps === 'number') { sum += m.fps; cnt++; }
        });
        setFps(cnt ? sum / cnt : null);
      } catch (e) { /* silent */ }
    };
    poll();
    const t = setInterval(poll, 1500);
    return () => clearInterval(t);
  }, [selectedVideos]);

  /* ── AUTO REFRESH ── */
  useEffect(() => {
    if (!onRefresh) return;
    const i = setInterval(onRefresh, 5000);
    return () => clearInterval(i);
  }, [onRefresh]);

  /* ── UPLOADED MAP ── */
  const uploadedMap = useMemo(() => {
    const map = {};
    let c = 0;
    videos.forEach(v => {
      if (v.label === 'UPLOADED') { c++; map[v.id] = `UP ${c}`; }
    });
    return map;
  }, [videos]);

  /* ── TOGGLE CAMERA ── */
  const toggleVideo = async (video) => {
    const active = selectedVideos.some(v => v.id === video.id);
    if (active) {
      if (selectedVideos.length === 1) return;
      onVideosChange(selectedVideos.filter(v => v.id !== video.id));
      try {
        await fetch(`${API_BASE_URL}/sources/stop`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ index: video.droneIndex }),
        });
      } catch (e) { /* silent */ }
    } else {
      onVideosChange([...selectedVideos, video]);
      try {
        await fetch(`${API_BASE_URL}/sources/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ index: video.droneIndex, mode: useCase }),
        });
      } catch (e) { /* silent */ }
    }
  };

  const isSelected = (id) => selectedVideos.some(v => v.id === id);

  /* ── UPLOADS ── */
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadMessage, setUploadMessage] = useState(null);
  const [uploadedVideos, setUploadedVideos] = useState([]);

  useEffect(() => {
    const fetchUploads = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/uploads`);
        if (res.ok) {
          const data = await res.json();
          setUploadedVideos(data.uploads || []);
        }
      } catch (e) { /* silent */ }
    };
    fetchUploads();
    const i = setInterval(fetchUploads, 5000);
    return () => clearInterval(i);
  }, []);

  const uploadFile = async (file) => {
    if (!file) return;
    setIsUploading(true);
    setUploadProgress(0);
    setUploadMessage(null);

    const formData = new FormData();
    formData.append('file', file);
    if (useCase) {
      formData.append('mode', useCase);
    }

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/upload`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setUploadProgress((e.loaded / e.total) * 100);
    };

    xhr.onload = () => {
      setIsUploading(false);
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const response = JSON.parse(xhr.responseText);
          setUploadMessage({ type: 'success', text: 'STARTED' });
          const newVideo = {
            id: response.name,
            type: 'upload',
            stream: response.name,
            label: file.name.replace(/\.[^/.]+$/, '').toUpperCase().slice(0, 8),
          };
          setUploadedVideos(prev => [...prev, { name: response.name, original_name: file.name }]);
          onVideosChange([...selectedVideos, newVideo]);
          setTimeout(() => setUploadMessage(null), 2000);
        } catch (e) {
          setUploadMessage({ type: 'error', text: 'PARSE ERROR' });
          setTimeout(() => setUploadMessage(null), 3000);
        }
      } else {
        setUploadMessage({ type: 'error', text: `FAILED (${xhr.status})` });
        setTimeout(() => setUploadMessage(null), 3000);
      }
    };

    xhr.onerror = () => {
      setIsUploading(false);
      setUploadMessage({ type: 'error', text: 'ERROR' });
      setTimeout(() => setUploadMessage(null), 3000);
    };

    xhr.send(formData);
  };

  const toggleUploadedVideo = (upload) => {
    const video = {
      id: upload.name,
      type: 'upload',
      stream: upload.name,
      label: (upload.original_name || upload.name).replace(/\.[^/.]+$/, '').toUpperCase().slice(0, 8),
    };
    const active = selectedVideos.some(v => v.id === upload.name);
    if (active) {
      if (selectedVideos.length === 1) return;
      onVideosChange(selectedVideos.filter(v => v.id !== upload.name));
    } else {
      onVideosChange([...selectedVideos, video]);
    }
  };

  /* ── RENDER ── */
  return (
    <div className="w-full min-h-[2.25rem] max-h-[2.75rem] bg-black/60 backdrop-blur-md border-t border-white/10 flex items-center justify-between px-2 sm:px-3 md:px-5 z-50 font-mono shrink-0 gap-2">

      {/* LEFT — Status */}
      <div className="flex items-center gap-2 sm:gap-3 md:gap-4 shrink-0">
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
          <span className="text-[9px] sm:text-[10px] md:text-[11px] text-emerald-400/90 font-bold tracking-wider">ONLINE</span>
        </div>

        <span className="text-white/10 hidden sm:inline">|</span>

        <div className="hidden sm:flex items-center gap-1.5">
          <Globe className="w-3 h-3 text-white/30" />
          <span className="text-[9px] sm:text-[10px] md:text-[11px] text-white/60 tracking-wider">
            {selectedVideos.length}<span className="text-white/30">/{videos.length + uploadedVideos.length}</span> FEEDS
          </span>
        </div>

        <span className="text-white/10 hidden md:inline">|</span>

        <div className="hidden md:flex items-center gap-1.5">
          <Activity className="w-3 h-3 text-white/30" />
          <span className="text-[9px] sm:text-[10px] md:text-[11px] text-white/60 tabular-nums">
            {fps !== null ? fps.toFixed(0) : '--'} <span className="text-white/30">FPS</span>
          </span>
        </div>
      </div>

      {/* CENTER — Camera selector + Upload */}
      <div className="flex items-center gap-1.5 sm:gap-2 md:gap-3 min-w-0 flex-1 justify-center">
        <div className="flex items-center gap-px bg-white/[0.03] border border-white/[0.06] rounded-sm overflow-x-auto min-w-0"
             style={{ scrollbarWidth: 'none' }}>
          {videos.map((vid, idx) => {
            const label = vid.label === 'UPLOADED'
              ? uploadedMap[vid.id] || 'UP'
              : vid.label || `CAM ${idx + 1}`;
            return (
              <button
                key={vid.id}
                onClick={() => toggleVideo(vid)}
                className={clsx(
                  'px-1.5 sm:px-2 md:px-2.5 py-0.5 sm:py-1 text-[8px] sm:text-[9px] md:text-[10px] uppercase tracking-wider font-bold transition-all whitespace-nowrap',
                  isSelected(vid.id)
                    ? 'bg-cyan-500/90 text-black'
                    : 'text-white/30 hover:text-white/60 hover:bg-white/[0.04]'
                )}
                title={DRONE_REGION_MAP[vid.id] || label}
              >
                {label}
              </button>
            );
          })}

          {uploadedVideos.length > 0 && (
            <div className="w-px h-3 sm:h-4 bg-white/10 mx-0.5 shrink-0" />
          )}

          {uploadedVideos.map((upload, idx) => {
            const label = `UP${idx + 1} ↑`;
            const isActive = selectedVideos.some(v => v.id === upload.name);
            return (
              <button
                key={upload.name}
                onClick={() => toggleUploadedVideo(upload)}
                className={clsx(
                  'px-1.5 sm:px-2 md:px-2.5 py-0.5 sm:py-1 text-[8px] sm:text-[9px] md:text-[10px] uppercase tracking-wider font-bold transition-all whitespace-nowrap',
                  isActive
                    ? 'bg-emerald-500/90 text-black'
                    : 'text-emerald-400/30 hover:text-emerald-400/60 hover:bg-white/[0.04]'
                )}
                title={`Uploaded: ${upload.original_name || upload.name}`}
              >
                {label}
              </button>
            );
          })}
        </div>

        {/* Upload button */}
        <div className="relative shrink-0">
          <input
            type="file"
            accept=".mp4,.mkv,.avi,.mov"
            className="absolute inset-0 opacity-0 cursor-pointer z-10"
            onChange={(e) => uploadFile(e.target.files[0])}
            disabled={isUploading}
          />
          <div className={clsx(
            'flex items-center gap-1 sm:gap-1.5 px-1.5 sm:px-2 md:px-3 py-0.5 sm:py-1 border text-[8px] sm:text-[9px] md:text-[10px] font-bold uppercase tracking-wider transition-all',
            isUploading
              ? 'border-cyan-500/20 bg-cyan-500/10 text-cyan-400'
              : 'border-white/10 text-white/40 hover:border-cyan-500/40 hover:text-cyan-400 hover:bg-cyan-500/10 cursor-pointer'
          )}>
            {isUploading ? (
              <>
                <div className="w-6 sm:w-8 md:w-10 h-1 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-400 transition-all" style={{ width: `${uploadProgress}%` }} />
                </div>
                <span>{Math.round(uploadProgress)}%</span>
              </>
            ) : uploadMessage ? (
              <span className={uploadMessage.type === 'success' ? 'text-emerald-400' : 'text-red-400'}>
                {uploadMessage.text}
              </span>
            ) : (
              <>
                <Upload className="w-2.5 h-2.5 sm:w-3 sm:h-3" />
                <span className="hidden sm:inline">UPLOAD</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* RIGHT — Branding */}
      <div className="flex items-center shrink-0">
        <span className="text-[8px] sm:text-[9px] md:text-[10px] text-white/20 tracking-widest whitespace-nowrap">IRIS <span className="hidden sm:inline">COMMAND </span>v1.0</span>
      </div>
    </div>
  );
};

export default Footer;
