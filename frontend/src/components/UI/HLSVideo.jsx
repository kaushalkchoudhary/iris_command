import React, { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';

const HLSVideo = ({ src, fallbackSrc, className = '', ...props }) => {
  const videoRef = useRef(null);
  const hlsRef = useRef(null);

  const [useFallback, setUseFallback] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;

    setUseFallback(false);
    setError(null);

    video.pause();
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();

    const isHLS = src.endsWith('.m3u8') || src.includes('/hls/');

    if (isHLS) {
      if (Hls.isSupported()) {
        const hls = new Hls({
          enableWorker: true,
          lowLatencyMode: false,

          backBufferLength: 90,
          maxBufferLength: 30,
          maxMaxBufferLength: 60,

          liveSyncDurationCount: 3,
          liveMaxLatencyDurationCount: 8,

          capLevelToPlayerSize: false,
          abrEwmaFastLive: 3,
          abrEwmaSlowLive: 9,

          maxStarvationDelay: 4,
          maxLoadingDelay: 4,
        });

        hlsRef.current = hls;

        hls.loadSource(src);
        hls.attachMedia(video);

        hls.on(Hls.Events.MANIFEST_PARSED, () => {
          video.play().catch(() => {});
        });

        hls.on(Hls.Events.ERROR, (_, data) => {
          if (!data.fatal) return;

          if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
            hls.startLoad();
            return;
          }

          if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
            hls.recoverMediaError();
            return;
          }

          hls.destroy();
          hlsRef.current = null;

          if (fallbackSrc) {
            setUseFallback(true);
          } else {
            setError('STREAM UNAVAILABLE');
          }
        });

        return () => {
          hls.destroy();
          hlsRef.current = null;
        };
      }

      if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = src;
        video.addEventListener(
          'loadedmetadata',
          () => video.play().catch(() => {}),
          { once: true }
        );
        return;
      }

      if (fallbackSrc) {
        setUseFallback(true);
      } else {
        setError('HLS NOT SUPPORTED');
      }
    } else {
      video.src = src;
      video.play().catch(() => {});
    }
  }, [src, fallbackSrc]);

  useEffect(() => {
    if (!useFallback || !fallbackSrc || !videoRef.current) return;

    const video = videoRef.current;
    video.srcObject = null;
    video.src = fallbackSrc;
    video.play().catch(() => {});
  }, [useFallback, fallbackSrc]);

  if (error && !fallbackSrc) {
    return (
      <div className={`flex items-center justify-center bg-black ${className}`}>
        <div className="text-center font-mono">
          <div className="text-emerald-400 text-sm tracking-widest">{error}</div>
          <div className="text-white/40 text-xs mt-1">Awaiting signal…</div>
        </div>
      </div>
    );
  }

  return (
    <video
      ref={videoRef}
      className={`w-full h-full object-cover iris-video ${className}`}
      playsInline
      muted
      autoPlay
      {...props}
    />
  );
};

export default HLSVideo;
