import React, { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';

/* === OLD IRIS VIDEO EDGE FADE (CRITICAL) === */
const irisVideoMaskStyle = {
  maskImage: `
    linear-gradient(to right,
      transparent 0%,
      black 12%,
      black 88%,
      transparent 100%),
    linear-gradient(to bottom,
      transparent 0%,
      black 12%,
      black 88%,
      transparent 100%)
  `,
  WebkitMaskImage: `
    linear-gradient(to right,
      transparent 0%,
      black 12%,
      black 88%,
      transparent 100%),
    linear-gradient(to bottom,
      transparent 0%,
      black 12%,
      black 88%,
      transparent 100%)
  `,
  maskComposite: 'intersect',
  WebkitMaskComposite: 'source-in',
};

const HLSVideo = ({ src, fallbackSrc, className = '', ...props }) => {
  const videoRef = useRef(null);
  const hlsRef = useRef(null);

  const [useFallback, setUseFallback] = useState(false);
  const [error, setError] = useState(null);

  /* === FULL TEARDOWN === */
  const destroyHls = () => {
    if (hlsRef.current) {
      try {
        hlsRef.current.destroy();
      } catch { }
      hlsRef.current = null;
    }
  };

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;

    setUseFallback(false);
    setError(null);

    destroyHls();

    video.pause();
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();

    const isHlsSource =
      src.endsWith('.m3u8') ||
      src.includes('/hls/') ||
      src.includes('index.m3u8');

    /* === HLS.JS PATH === */
    if (isHlsSource && Hls.isSupported()) {
      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,

        backBufferLength: 20,
        maxBufferLength: 10,
        maxMaxBufferLength: 20,

        // Slightly relaxed LL-HLS edge to avoid micro-stalls/rebuffers.
        liveSyncDurationCount: 2,
        liveMaxLatencyDurationCount: 5,

        abrEwmaFastLive: 2,
        abrEwmaSlowLive: 6,

        maxStarvationDelay: 2,
        maxLoadingDelay: 2,
      });

      hlsRef.current = hls;
      hls.loadSource(src);
      hls.attachMedia(video);

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(() => { });
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

        destroyHls();

        if (fallbackSrc) {
          setUseFallback(true);
        } else {
          setError('STREAM UNAVAILABLE');
        }
      });

      return () => destroyHls();
    }

    /* === SAFARI NATIVE HLS === */
    if (isHlsSource && video.canPlayType('application/vnd.apple.mpegurl')) {
      const onLoaded = () => video.play().catch(() => { });
      video.src = src;
      video.addEventListener('loadedmetadata', onLoaded, { once: true });

      return () => {
        video.removeEventListener('loadedmetadata', onLoaded);
      };
    }

    /* === FALLBACK OR DIRECT VIDEO === */
    if (fallbackSrc) {
      setUseFallback(true);
    } else if (!isHlsSource) {
      video.src = src;
      video.play().catch(() => { });
    } else {
      setError('HLS NOT SUPPORTED');
    }
  }, [src, fallbackSrc]);

  /* === FALLBACK VIDEO === */
  useEffect(() => {
    if (!useFallback || !fallbackSrc || !videoRef.current) return;

    destroyHls();

    const video = videoRef.current;
    video.pause();
    video.srcObject = null;
    video.src = fallbackSrc;
    video.load();
    video.play().catch(() => { });
  }, [useFallback, fallbackSrc]);

  /* === ERROR STATE === */
  if (error && !fallbackSrc) {
    return (
      <div className={`flex items-center justify-center bg-transparent ${className}`}>
        <div className="text-center font-mono">
          <div className="text-emerald-400 text-sm tracking-widest">
            {error}
          </div>
          <div className="text-white/40 text-xs mt-1">
            Awaiting signal…
          </div>
        </div>
      </div>
    );
  }

  return (
    <video
      ref={videoRef}
      className={`w-full h-full object-cover ${className}`}
      style={irisVideoMaskStyle}
      playsInline
      muted
      autoPlay
      {...props}
    />
  );
};

export default HLSVideo;
