import React, { useEffect, useState } from 'react';
// Map drone id → region name (shared with RightPanel)

/* ============================
   TYPEWRITER HOOK
============================ */
const useTypewriter = (text, speed = 60, resetKey = 0) => {
  const [output, setOutput] = useState('');
  const [index, setIndex] = useState(0);

  useEffect(() => {
    setOutput('');
    setIndex(0);
  }, [resetKey, text]);

  useEffect(() => {
    if (index >= text.length) return;
    const t = setTimeout(() => {
      setOutput((prev) => prev + text[index]);
      setIndex((i) => i + 1);
    }, speed);
    return () => clearTimeout(t);
  }, [index, text, speed]);

  return output;
};

/* ============================
   TECH SENSOR ICON
============================ */
const SensorCore = () => {
  return (
    <div className="relative w-5 h-5 flex items-center justify-center">
      <div className="absolute inset-0 rounded-full border border-emerald-400/25 animate-[spin_12s_linear_infinite]" />
      <div className="absolute inset-[3px] rounded-full border border-dashed border-emerald-400/30 animate-[spin_6s_linear_reverse_infinite]" />
      <div className="relative w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]">
        <div className="absolute inset-0 rounded-full bg-emerald-400/40 animate-ping" />
      </div>
    </div>
  );
};

/* ============================
   HEADER
============================ */
const Header = ({ onReset }) => {
  const CALLSIGN = 'IRIS';
  const FULL_FORM = 'INTELLIGENT RECONNAISSANCE & INSIGHT SYSTEM';

  const [cycle, setCycle] = useState(0);
  const [showFullForm, setShowFullForm] = useState(false);

  // Toggle every 3 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setShowFullForm((v) => !v);
      setCycle((c) => c + 1);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const irisText = useTypewriter(CALLSIGN, 120, cycle);
  const fullFormText = useTypewriter(FULL_FORM, 28, cycle);

  return (
    <header
      className="
        w-full z-50 px-5 py-2
        flex items-center justify-between
        bg-[#050a14]/90 backdrop-blur-xl
        border-b border-emerald-500/15
        select-none font-mono
      "
    >
      {/* BRAND */}
      <button
        onClick={onReset}
        className="flex items-center gap-3 pointer-events-auto
                   hover:opacity-85 transition-opacity"
      >
        {/* TECH SENSOR ICON */}
        <SensorCore />

        {/* SINGLE LINE BRAND */}
        <div className="flex items-center gap-3 h-5 overflow-hidden">
          {/* IRIS */}
          <span className="text-lg font-black tracking-[0.45em] text-emerald-400 whitespace-nowrap">
            {irisText}
            <span
              className="inline-block w-[1px] h-4 ml-1
                         bg-emerald-400/70
                         animate-pulse align-middle"
            />
          </span>

          {/* FULL FORM — APPEARS BESIDE */}
          <span
            className={`
              text-[10px] tracking-widest whitespace-nowrap
              text-emerald-400/40
              transition-all duration-700
              ${showFullForm ? 'opacity-100 max-w-[420px]' : 'opacity-0 max-w-0'}
            `}
          >
            {fullFormText}
          </span>
        </div>
      </button>

      {/* RIGHT — intentionally empty */}
      <div className="w-6" />
    </header>
  );
};

export default Header;
