import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Cpu, Activity, Wifi, Radar } from 'lucide-react';

/* ============================
   BOOT SEQUENCE
============================ */
const BOOT_STEPS = [
  { label: 'LINK LAYER', detail: 'Encrypted channel established', target: 20 },
  { label: 'CORE INIT', detail: 'Compute cores online', target: 45 },
  { label: 'STREAM BUS', detail: 'Video pipelines synchronized', target: 70 },
  { label: 'EDGE AI', detail: 'Inference models loaded', target: 90 },
  { label: 'READY', detail: 'All systems nominal', target: 100 },
];

const IRISLoader = () => {
  const [progress, setProgress] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    if (stepIndex >= BOOT_STEPS.length) return;

    const { target } = BOOT_STEPS[stepIndex];

    const interval = setInterval(() => {
      setProgress((p) => {
        if (p >= target) {
          clearInterval(interval);
          setStepIndex((i) => i + 1);
          return p;
        }
        return Math.min(p + 1, target);
      });
    }, 40);

    return () => clearInterval(interval);
  }, [stepIndex]);

  const currentStep = BOOT_STEPS[Math.min(stepIndex, BOOT_STEPS.length - 1)];

  return (
    <div className="absolute inset-0 z-[100] bg-[#050a14] flex items-center justify-center font-mono">

      {/* BACKGROUND GRID */}
      <div
        className="absolute inset-0 opacity-[0.08] pointer-events-none"
        style={{
          backgroundImage:
            'linear-gradient(rgba(16,185,129,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(16,185,129,0.15) 1px, transparent 1px)',
          backgroundSize: '36px 36px',
        }}
      />

      {/* CENTER STACK */}
      <div className="relative flex flex-col items-center gap-8">

        {/* RADAR CORE */}
        <div className="relative w-40 h-40 flex items-center justify-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
            className="absolute inset-0 rounded-full border border-emerald-400/20"
          />

          <motion.div
            animate={{ rotate: -360 }}
            transition={{ duration: 14, repeat: Infinity, ease: 'linear' }}
            className="absolute inset-[10px] rounded-full border border-dashed border-emerald-400/20"
          />

          <Radar className="w-10 h-10 text-emerald-400" strokeWidth={1.4} />

          <div className="absolute bottom-[-2.5rem] text-lg font-black tracking-[0.45em] text-emerald-400">
            IRIS
          </div>
        </div>

        {/* STATUS BLOCK */}
        <div className="w-[340px]">

          {/* LABEL */}
          <div className="flex justify-between items-end mb-2">
            <div>
              <div className="text-[10px] text-emerald-400/70 uppercase tracking-widest font-bold">
                SYSTEM BOOT
              </div>
              <div className="text-[11px] text-white/50 mt-0.5">
                {currentStep.label} — {currentStep.detail}
              </div>
            </div>

            <div className="text-xl font-bold text-emerald-400 tabular-nums">
              {progress}%
            </div>
          </div>

          {/* PROGRESS BAR */}
          <div className="h-1 w-full bg-white/5 rounded overflow-hidden">
            <motion.div
              className="h-full bg-emerald-400"
              animate={{ width: `${progress}%` }}
              transition={{ ease: 'linear' }}
            />
          </div>

          {/* SUBSYSTEMS */}
          <div className="grid grid-cols-3 gap-3 mt-4 text-[9px] uppercase tracking-widest text-white/50">
            <div className="flex items-center gap-2">
              <Cpu className="w-3 h-3 text-emerald-400/70" />
              CORES
            </div>
            <div className="flex items-center gap-2">
              <Activity className="w-3 h-3 text-emerald-400/70" />
              STREAMS
            </div>
            <div className="flex items-center gap-2 justify-end">
              <Wifi className="w-3 h-3 text-emerald-400/70" />
              LINK
            </div>
          </div>
        </div>
      </div>

      {/* FOOT NOTE */}
      <motion.div
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 2, repeat: Infinity }}
        className="absolute bottom-6 text-[10px] tracking-[0.45em] uppercase text-emerald-400/70"
      >
        SYNCHRONIZING SURVEILLANCE NODES
      </motion.div>
    </div>
  );
};

export default IRISLoader;
