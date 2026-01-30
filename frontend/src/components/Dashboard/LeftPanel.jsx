import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = import.meta.env.DEV
  ? '/api'
  : `http://${window.location.hostname}:9010`;

const LeftPanel = () => {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    let timer;
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (res.ok) {
          const data = await res.json();
          setMetrics(data.global || data);
        }
      } catch (e) {
        console.error('metrics poll failed', e);
      }
    };

    poll();
    timer = setInterval(poll, 2000);
    return () => clearInterval(timer);
  }, []);

  const congestion = Math.round(metrics?.congestion_index ?? 0);
  const speed = Math.round(metrics?.mobility_index ?? 0);

  let status = 'SMOOTH';
  let color = 'text-cyan-400';
  let borderColor = 'border-cyan-400/50';

  if (congestion > 75) {
    status = 'HEAVY';
    color = 'text-red-500';
    borderColor = 'border-red-500/50';
  } else if (congestion > 50) {
    status = 'SLOW';
    color = 'text-yellow-500';
    borderColor = 'border-yellow-500/50';
  }

  return (
    <>
      {/* ===== OLD IRIS METRICS HUD ===== */}
      <motion.div
        initial={{ x: -40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        className="absolute left-6 top-1/2 -translate-y-1/2 flex flex-col gap-12 pointer-events-none z-40"
      >

      </motion.div>
    </>
  );
};

export default LeftPanel;
