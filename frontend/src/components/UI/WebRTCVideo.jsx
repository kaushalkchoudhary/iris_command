import React, { useEffect, useRef, useState } from 'react';

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

const WebRTCVideo = ({ src, fallbackWebrtcSrc, fallbackSrc, className, ...props }) => {
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const sessionUrlRef = useRef(null);
  const [useFallback, setUseFallback] = useState(false);
  const [error, setError] = useState(null);
  const [activeSrc, setActiveSrc] = useState(src);
  const [triedWebrtcFallback, setTriedWebrtcFallback] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const maxRetries = 5;
  const retryDelay = 2000; // 2 seconds between retries

  useEffect(() => {
    setActiveSrc(src);
    setTriedWebrtcFallback(false);
    setRetryCount(0);
  }, [src]);

  useEffect(() => {
    let cancelled = false;
    const video = videoRef.current;
    if (!video || !activeSrc) return;

    setUseFallback(false);
    setError(null);

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
      if (pc.connectionState === 'failed' && !cancelled) {
        // Retry primary stream a few times before falling back (gives backend time to start)
        const activeSrcBase = activeSrc.split('?')[0];
        if (retryCount < maxRetries && activeSrcBase === src) {
          console.log(`WebRTC: Retrying primary stream (${retryCount + 1}/${maxRetries})...`);
          setTimeout(() => {
            if (!cancelled) {
              setRetryCount(r => r + 1);
              setActiveSrc(src + '?retry=' + Date.now()); // Force re-render
            }
          }, retryDelay);
          return;
        }
        if (!triedWebrtcFallback && fallbackWebrtcSrc) {
          console.log('WebRTC: Falling back to raw stream');
          setTriedWebrtcFallback(true);
          setActiveSrc(fallbackWebrtcSrc);
          return;
        }
        if (fallbackSrc) {
          setUseFallback(true);
        } else {
          setError('Stream unavailable');
        }
      }
    };

    const setup = async () => {
      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitForIceGatheringComplete(pc);

        const fetchUrl = activeSrc.split('?')[0]; // Strip retry query param
        const response = await fetch(fetchUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/sdp',
          },
          body: pc.localDescription?.sdp || '',
        });

        if (!response.ok) {
          throw new Error(`WebRTC signaling failed (${response.status})`);
        }

        const answerSdp = await response.text();
        const sessionUrl = response.headers.get('Location');
        if (sessionUrl) {
          sessionUrlRef.current = sessionUrl;
        }

        await pc.setRemoteDescription({
          type: 'answer',
          sdp: answerSdp,
        });
      } catch (err) {
        if (cancelled) return;
        // Retry primary stream a few times before falling back
        const activeSrcBase = activeSrc.split('?')[0];
        if (retryCount < maxRetries && activeSrcBase === src) {
          console.log(`WebRTC: Retrying primary stream after error (${retryCount + 1}/${maxRetries})...`);
          setTimeout(() => {
            if (!cancelled) {
              setRetryCount(r => r + 1);
              setActiveSrc(src + '?retry=' + Date.now());
            }
          }, retryDelay);
          return;
        }
        if (!triedWebrtcFallback && fallbackWebrtcSrc) {
          console.log('WebRTC: Falling back to raw stream after error');
          setTriedWebrtcFallback(true);
          setActiveSrc(fallbackWebrtcSrc);
          return;
        }
        if (fallbackSrc) {
          setUseFallback(true);
        } else {
          setError(err?.message || 'WebRTC error');
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
  }, [activeSrc, fallbackSrc, fallbackWebrtcSrc, triedWebrtcFallback, retryCount, src]);

  useEffect(() => {
    if (useFallback && fallbackSrc && videoRef.current) {
      videoRef.current.srcObject = null;
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

export default WebRTCVideo;
