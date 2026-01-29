import React, { useEffect, useRef, useState } from 'react';

const waitForIceGatheringComplete = (pc) =>
  new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') return resolve();
    const handler = () => {
      if (pc.iceGatheringState === 'complete') {
        pc.removeEventListener('icegatheringstatechange', handler);
        resolve();
      }
    };
    pc.addEventListener('icegatheringstatechange', handler);
  });

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
  const primaryRetryRef = useRef(null);

  const MAX_RETRIES = 5;
  const RETRY_DELAY = 2000;
  const PRIMARY_RETRY_INTERVAL = 5000; // Retry primary stream every 5s when on fallback

  useEffect(() => {
    setActiveSrc(src);
    setRetryCount(0);
    setUsedWebrtcFallback(false);
    setUseFallback(false);
    setError(null);
  }, [src]);

  useEffect(() => {
    let cancelled = false;
    const video = videoRef.current;
    if (!video || !activeSrc) return;

    video.pause();
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();

    const pc = new RTCPeerConnection({
      bundlePolicy: 'max-bundle',
      rtcpMuxPolicy: 'require',
      iceTransportPolicy: 'all'
    });

    pcRef.current = pc;

    pc.addTransceiver('video', {
      direction: 'recvonly',
      sendEncodings: [{ maxBitrate: 6_000_000 }]
    });

    pc.ontrack = (event) => {
      if (cancelled) return;
      if (video.srcObject !== event.streams[0]) {
        video.srcObject = event.streams[0];
        video.play().catch(() => {});
      }
    };

    pc.onconnectionstatechange = () => {
      if (cancelled) return;
      if (pc.connectionState === 'connected') return;

      if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        const base = activeSrc.split('?')[0];

        if (retryCount < MAX_RETRIES && base === src) {
          setTimeout(() => {
            if (!cancelled) {
              setRetryCount(c => c + 1);
              setActiveSrc(`${src}?r=${Date.now()}`);
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
          setError('STREAM UNAVAILABLE');
        }
      }
    };

    const setup = async () => {
      try {
        const offer = await pc.createOffer({
          offerToReceiveVideo: true
        });
        await pc.setLocalDescription(offer);
        await waitForIceGatheringComplete(pc);

        const res = await fetch(activeSrc.split('?')[0], {
          method: 'POST',
          headers: { 'Content-Type': 'application/sdp' },
          body: pc.localDescription.sdp
        });

        if (!res.ok) throw new Error();

        const answer = await res.text();
        const sessionUrl = res.headers.get('Location');
        if (sessionUrl) sessionUrlRef.current = sessionUrl;

        await pc.setRemoteDescription({ type: 'answer', sdp: answer });
      } catch {
        if (cancelled) return;

        if (retryCount < MAX_RETRIES) {
          setTimeout(() => {
            if (!cancelled) {
              setRetryCount(c => c + 1);
              setActiveSrc(`${src}?r=${Date.now()}`);
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
          setError('WEBRTC ERROR');
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

  useEffect(() => {
    if (!useFallback || !fallbackSrc || !videoRef.current) return;
    const video = videoRef.current;
    video.srcObject = null;
    video.src = fallbackSrc;
    video.play().catch(() => {});
  }, [useFallback, fallbackSrc]);

  // Periodically retry primary (processed) stream when on fallback
  useEffect(() => {
    if (!usedWebrtcFallback || !src) return;

    const tryPrimary = async () => {
      try {
        // Quick check if the processed stream responds
        const res = await fetch(src.split('?')[0], {
          method: 'OPTIONS',
        });
        if (res.ok) {
          // Stream is available, switch back to it
          setUsedWebrtcFallback(false);
          setRetryCount(0);
          setActiveSrc(`${src}?r=${Date.now()}`);
        }
      } catch {
        // Stream not ready yet, will retry
      }
    };

    primaryRetryRef.current = setInterval(tryPrimary, PRIMARY_RETRY_INTERVAL);
    // Try immediately once
    tryPrimary();

    return () => {
      if (primaryRetryRef.current) {
        clearInterval(primaryRetryRef.current);
      }
    };
  }, [usedWebrtcFallback, src]);

  if (error && !fallbackSrc) {
    return (
      <div className={`flex items-center justify-center bg-black/90 ${className}`}>
        <div className="text-center font-mono">
          <div className="text-white/20 text-xs tracking-[0.3em] uppercase">No Signal</div>
          <div className="w-8 h-[1px] bg-white/10 mx-auto mt-2" />
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

export default WebRTCVideo;
