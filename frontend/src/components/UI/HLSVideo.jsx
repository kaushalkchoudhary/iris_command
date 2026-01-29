import React, { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';

const HLSVideo = ({ src, fallbackSrc, className, ...props }) => {
  const videoRef = useRef(null);
  const hlsRef = useRef(null);
  const [useFallback, setUseFallback] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;

    // Reset state on src change
    setUseFallback(false);
    setError(null);

    // Check if it's an HLS stream
    const isHLS = src.endsWith('.m3u8') || src.includes('/hls/');

    if (isHLS) {
      if (Hls.isSupported()) {
        const hls = new Hls({
          enableWorker: true,
          lowLatencyMode: true,
          backBufferLength: 90,
          maxBufferLength: 30,
          maxMaxBufferLength: 600,
          liveSyncDurationCount: 3,
          liveMaxLatencyDurationCount: 10,
        });

        hlsRef.current = hls;

        hls.loadSource(src);
        hls.attachMedia(video);

        hls.on(Hls.Events.MANIFEST_PARSED, () => {
          video.play().catch(() => {});
        });

        hls.on(Hls.Events.ERROR, (event, data) => {
          if (data.fatal) {
            switch (data.type) {
              case Hls.ErrorTypes.NETWORK_ERROR:
                console.warn('HLS network error, attempting recovery...');
                hls.startLoad();
                break;
              case Hls.ErrorTypes.MEDIA_ERROR:
                console.warn('HLS media error, attempting recovery...');
                hls.recoverMediaError();
                break;
              default:
                console.error('Fatal HLS error:', data);
                if (fallbackSrc) {
                  setUseFallback(true);
                } else {
                  setError('Stream unavailable');
                }
                hls.destroy();
                break;
            }
          }
        });

        return () => {
          hls.destroy();
          hlsRef.current = null;
        };
      } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        // Native HLS support (Safari)
        video.src = src;
        video.addEventListener('loadedmetadata', () => {
          video.play().catch(() => {});
        });
      } else {
        // No HLS support, use fallback
        if (fallbackSrc) {
          setUseFallback(true);
        } else {
          setError('HLS not supported');
        }
      }
    } else {
      // Regular video file
      video.src = src;
    }
  }, [src, fallbackSrc]);

  // Handle fallback video
  useEffect(() => {
    if (useFallback && fallbackSrc && videoRef.current) {
      videoRef.current.src = fallbackSrc;
      videoRef.current.play().catch(() => {});
    }
  }, [useFallback, fallbackSrc]);

  if (error && !fallbackSrc) {
    return (
      <div className={`flex items-center justify-center bg-black/50 ${className}`}>
        <div className="text-center">
          <div className="text-red-500 text-sm font-mono">{error}</div>
          <div className="text-white/50 text-xs mt-1">Waiting for stream...</div>
        </div>
      </div>
    );
  }

  return (
    <video
      ref={videoRef}
      className={className}
      {...props}
    />
  );
};

export default HLSVideo;
