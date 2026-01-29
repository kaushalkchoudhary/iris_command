import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

/* ============================
   UTIL: DAMPED VALUE UPDATE
============================ */
const damp = (current, target, factor = 0.08) =>
  current + (target - current) * factor;

/* ============================
   LEFT TELEMETRY PANEL
============================ */
const LeftPanel = () => {
  // Internal “true” load (simulates backend metric)
  const [loadTarget, setLoadTarget] = useState(42);

  // Displayed values (smoothed)
  const [systemLoad, setSystemLoad] = useState(42);

  /* --------------------------------
     Simulate backend updates (slow)
  ---------------------------------*/
  useEffect(() => {
    const interval = setInterval(() => {
      setLoadTarget((prev) => {
        const drift = (Math.random() - 0.5) * 6;
        return Math.min(85, Math.max(15, prev + drift));
      });
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  /* --------------------------------
     Smooth toward target (fast)
  ---------------------------------*/
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemLoad((v) => damp(v, loadTarget));
    }, 80);
    return () => clearInterval(interval);
  }, [loadTarget]);

  /* --------------------------------
     Derived metrics
  ---------------------------------*/
  const throughput = Math.round(100 - systemLoad);
  const loadRounded = Math.round(systemLoad);

  let state = 'NOMINAL';
  let stateColor = 'text-emerald-400';
  let borderColor = 'border-emerald-400/50';

  if (loadRounded > 70) {
    state = 'DEGRADED';
    stateColor = 'text-emerald-300';
    borderColor = 'border-emerald-300/40';
  }
  if (loadRounded > 82) {
    state = 'CONSTRAINED';
    stateColor = 'text-emerald-200';
    borderColor = 'border-emerald-200/40';
  }

  return (
    <motion.div
      initial={{ x: -40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      className="
        absolute left-6 top-1/2 -translate-y-1/2
        flex flex-col gap-10
        pointer-events-none z-40
        font-mono
      "
    >
      {/* THROUGHPUT */}
      <div>
        <div className="text-[10px] uppercase tracking-[0.35em] text-emerald-400/60 font-bold">
          NETWORK THROUGHPUT
        </div>

        <div className="flex items-end gap-2 mt-1">
          <span className="text-[96px] font-black text-white leading-none tracking-tighter">
            {throughput}
          </span>
          <span className="mb-3 text-sm text-emerald-400/50 font-bold">
            %
          </span>
        </div>
      </div>

      {/* SYSTEM LOAD */}
      <div>
        <div className="text-[10px] uppercase tracking-[0.35em] text-emerald-400/60 font-bold">
          SYSTEM LOAD
        </div>

        <div className="flex items-end gap-2 mt-1">
          <span
            className={`text-[64px] font-black leading-none ${stateColor}`}
          >
            {loadRounded}
          </span>
          <span className="mb-2 text-sm text-emerald-400/40 font-bold">
            %
          </span>
        </div>
      </div>

      {/* STATUS STRIP */}
      <div
        className={`
          px-4 py-2 border-l-4
          ${borderColor}
          bg-black/40 backdrop-blur
          self-start
        `}
      >
        <div
          className={`text-[11px] font-black uppercase tracking-[0.3em] ${stateColor}`}
        >
          STATE: {state}
        </div>
      </div>

      {/* NODE METADATA */}
      <div className="pt-4 border-t border-white/5 text-[10px] text-emerald-400/50 space-y-1">
        <div className="flex gap-3">
          <span className="text-emerald-400/70">NODE</span>
          <span>IRIS-BLR-MAIN</span>
        </div>
        <div className="flex gap-3">
          <span className="text-emerald-400/70">LAT</span>
          <span>12.9716° N</span>
        </div>
        <div className="flex gap-3">
          <span className="text-emerald-400/70">LNG</span>
          <span>77.5946° E</span>
        </div>
      </div>
    </motion.div>
  );
};

export default LeftPanel;
