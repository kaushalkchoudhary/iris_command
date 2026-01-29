import React, { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';

const HLSVideo = ({
  src,
  fallbackSrc,
  className = '',
  ...props
}) => {
  const videoRef = useRef(null);
  const hlsRef = useRef(null);

  const [useFallback, setUseFallback] = useState(false);
  const [error, setError] = useState(null);

  /* ============================
     MAIN STREAM SETUP
  ============================ */
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;

    // Reset state on source change
    setUseFallback(false);
    setError(null);

    // Hard reset video element
    video.pause();
    video.removeAttribute('src');
    video.load();

    const isHLS = src.endsWith('.m3u8') || src.includes('/hls/');

    if (isHLS) {
      if (Hls.isSupported()) {
        const hls = new Hls({
          enableWorker: true,
          lowLatencyMode: true,

          // Less aggressive buffering → better clarity
          backBufferLength: 60,
          maxBufferLength: 20,
          liveSyncDurationCount: 2,
          liveMaxLatencyDurationCount: 6,

          // Avoid excessive level switching artifacts
          capLevelToPlayerSize: true,
        });

        hlsRef.current = hls;

        hls.loadSource(src);
        hls.attachMedia(video);

        hls.on(Hls.Events.MANIFEST_PARSED, () => {
          video.play().catch(() => {});
        });

        hls.on(Hls.Events.ERROR, (_, data) => {
          if (!data.fatal) return;

          switch (data.type) {
            case Hls.ErrorTypes.NETWORK_ERROR:
              console.warn('[HLS] Network error → retrying');
              hls.startLoad();
              break;

            case Hls.ErrorTypes.MEDIA_ERROR:
              console.warn('[HLS] Media error → recovering');
              hls.recoverMediaError();
              break;

            default:
              console.error('[HLS] Fatal error', data);
              hls.destroy();
              hlsRef.current = null;

              if (fallbackSrc) {
                setUseFallback(true);
              } else {
                setError('STREAM UNAVAILABLE');
              }
          }
        });

        return () => {
          hls.destroy();
          hlsRef.current = null;
        };
      }

      // Native HLS (Safari)
      if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = src;
        video.addEventListener(
          'loadedmetadata',
          () => video.play().catch(() => {}),
          { once: true }
        );
        return;
      }

      // No HLS support at all
      if (fallbackSrc) {
        setUseFallback(true);
      } else {
        setError('HLS NOT SUPPORTED');
      }
    } else {
      // Regular MP4 / file stream
      video.src = src;
      video.play().catch(() => {});
    }
  }, [src, fallbackSrc]);

  /* ============================
     FALLBACK HANDLER
  ============================ */
  useEffect(() => {
    if (!useFallback || !fallbackSrc || !videoRef.current) return;

    const video = videoRef.current;
    video.src = fallbackSrc;
    video.play().catch(() => {});
  }, [useFallback, fallbackSrc]);

  /* ============================
     ERROR STATE (MINIMAL)
  ============================ */
  if (error && !fallbackSrc) {
    return (
      <div
        className={`flex items-center justify-center bg-black ${className}`}
      >
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

  /* ============================
     VIDEO RENDER (REFINED)
  ============================ */
  return (
    <video
      ref={videoRef}
      className={`
        w-full h-full object-cover
        iris-video
        ${className}
      `}
      playsInline
      muted
      autoPlay
      {...props}
    />
  );
};

export default HLSVideo;
