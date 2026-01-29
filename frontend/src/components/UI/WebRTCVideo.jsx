import React, { useEffect, useRef, useState } from 'react';

/* ============================
   ICE GATHER HELPER
============================ */
const waitForIceGatheringComplete = (pc) =>
  new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') {
      resolve();
      return;
    }
    const onStateChange = () => {
      if (pc.iceGatheringState === 'complete') {
        pc.removeEventListener('icegatheringstatechange', onStateChange);
        resolve();
      }
    };
    pc.addEventListener('icegatheringstatechange', onStateChange);
  });

/* ============================
   WEBRTC VIDEO
============================ */
const WebRTCVideo = ({
  src,
  fallbackWebrtcSrc,
  fallbackSrc,
  className = '',
  ...props
}) => {
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const sessionUrlRef = useRef(null);

  const [activeSrc, setActiveSrc] = useState(src);
  const [useFallback, setUseFallback] = useState(false);
  const [error, setError] = useState(null);

  const [retryCount, setRetryCount] = useState(0);
  const [usedWebrtcFallback, setUsedWebrtcFallback] = useState(false);

  const MAX_RETRIES = 5;
  const RETRY_DELAY = 2000;

  /* ============================
     RESET ON SRC CHANGE
  ============================ */
  useEffect(() => {
    setActiveSrc(src);
    setRetryCount(0);
    setUsedWebrtcFallback(false);
    setUseFallback(false);
    setError(null);
  }, [src]);

  /* ============================
     MAIN WEBRTC SETUP
  ============================ */
  useEffect(() => {
    let cancelled = false;
    const video = videoRef.current;
    if (!video || !activeSrc) return;

    // Hard reset video element
    video.pause();
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();

    const pc = new RTCPeerConnection();
    pcRef.current = pc;

    pc.addTransceiver('video', { direction: 'recvonly' });

    pc.ontrack = (event) => {
      if (cancelled) return;
      if (video.srcObject !== event.streams[0]) {
        video.srcObject = event.streams[0];
      }
      video.play().catch(() => {});
    };

    pc.onconnectionstatechange = () => {
      if (pc.connectionState !== 'failed' || cancelled) return;

      const baseSrc = activeSrc.split('?')[0];

      // Retry primary feed first
      if (retryCount < MAX_RETRIES && baseSrc === src) {
        setTimeout(() => {
          if (!cancelled) {
            setRetryCount((c) => c + 1);
            setActiveSrc(`${src}?retry=${Date.now()}`);
          }
        }, RETRY_DELAY);
        return;
      }

      // WebRTC fallback (raw feed)
      if (!usedWebrtcFallback && fallbackWebrtcSrc) {
        setUsedWebrtcFallback(true);
        setActiveSrc(fallbackWebrtcSrc);
        return;
      }

      // Final fallback (HLS / MP4)
      if (fallbackSrc) {
        setUseFallback(true);
      } else {
        setError('STREAM UNAVAILABLE');
      }
    };

    const setup = async () => {
      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitForIceGatheringComplete(pc);

        const fetchUrl = activeSrc.split('?')[0];
        const response = await fetch(fetchUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/sdp' },
          body: pc.localDescription?.sdp || '',
        });

        if (!response.ok) {
          throw new Error(`Signaling failed (${response.status})`);
        }

        const answerSdp = await response.text();
        const sessionUrl = response.headers.get('Location');
        if (sessionUrl) sessionUrlRef.current = sessionUrl;

        await pc.setRemoteDescription({
          type: 'answer',
          sdp: answerSdp,
        });
      } catch (err) {
        if (cancelled) return;

        const baseSrc = activeSrc.split('?')[0];

        if (retryCount < MAX_RETRIES && baseSrc === src) {
          setTimeout(() => {
            if (!cancelled) {
              setRetryCount((c) => c + 1);
              setActiveSrc(`${src}?retry=${Date.now()}`);
            }
          }, RETRY_DELAY);
          return;
        }

        if (!usedWebrtcFallback && fallbackWebrtcSrc) {
          setUsedWebrtcFallback(true);
          setActiveSrc(fallbackWebrtcSrc);
          return;
        }

        if (fallbackSrc) {
          setUseFallback(true);
        } else {
          setError(err?.message || 'WEBRTC ERROR');
        }
      }
    };

    setup();

    return () => {
      cancelled = true;

      if (sessionUrlRef.current) {
        fetch(sessionUrlRef.current, { method: 'DELETE' }).catch(() => {});
        sessionUrlRef.current = null;
      }

      pc.close();
      pcRef.current = null;
    };
  }, [activeSrc, src, fallbackWebrtcSrc, fallbackSrc, retryCount, usedWebrtcFallback]);

  /* ============================
     FINAL FALLBACK (FILE / HLS)
  ============================ */
  useEffect(() => {
    if (!useFallback || !fallbackSrc || !videoRef.current) return;
    const video = videoRef.current;
    video.srcObject = null;
    video.src = fallbackSrc;
    video.play().catch(() => {});
  }, [useFallback, fallbackSrc]);

  /* ============================
     ERROR STATE (MINIMAL)
  ============================ */
  if (error && !fallbackSrc) {
    return (
      <div className={`flex items-center justify-center bg-black ${className}`}>
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
     VIDEO RENDER (IRIS STYLE)
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

export default WebRTCVideo;
