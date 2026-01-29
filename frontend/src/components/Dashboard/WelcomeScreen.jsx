import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  Car,
  Users,
  ShieldAlert,
  Fence,
  ChevronRight,
  Power,
  Settings,
} from 'lucide-react';

/* ============================
   REAL MISSION DEFINITIONS
============================ */
const MISSIONS = [
  {
    id: 'traffic',
    title: 'TRAFFIC INTELLIGENCE',
    description: 'Aerial vehicle tracking and flow analysis across arterial roads.',
    icon: Car,
    drones: 12,
    status: 'ACTIVE',
  },
  {
    id: 'crowd',
    title: 'CROWD ANALYTICS',
    description: 'Density estimation and movement analysis in public zones.',
    icon: Users,
    drones: 3,
    status: 'STANDBY',
  },
  {
    id: 'safety',
    title: 'PUBLIC SAFETY',
    description: 'Threat detection and anomaly monitoring.',
    icon: ShieldAlert,
    drones: 2,
    status: 'ACTIVE',
  },
  {
    id: 'perimeter',
    title: 'PERIMETER SECURITY',
    description: 'Restricted zone surveillance and intrusion detection.',
    icon: Fence,
    drones: 1,
    status: 'MAINT',
  },
];

const WelcomeScreen = () => {
  const navigate = useNavigate();

  return (
    <div className="fixed inset-0 bg-[#050a14] font-mono text-white">

      {/* BACKGROUND GRID */}
      <div
        className="absolute inset-0 opacity-[0.06] pointer-events-none"
        style={{
          backgroundImage:
            'linear-gradient(rgba(16,185,129,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(16,185,129,0.2) 1px, transparent 1px)',
          backgroundSize: '48px 48px',
        }}
      />

      {/* TOP BAR */}
      <header className="h-14 border-b border-white/5 bg-black/40 backdrop-blur flex items-center justify-between px-6">
        <div className="text-sm tracking-[0.35em] text-emerald-400 font-black">
          IRIS COMMAND
        </div>

        <div className="flex items-center gap-4 text-white/40">
          <button className="hover:text-white">
            <Settings className="w-4 h-4" />
          </button>
          <button className="hover:text-emerald-400">
            <Power className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* CENTER */}
      <main className="relative z-10 h-[calc(100%-7rem)] flex flex-col items-center justify-center px-8">

        {/* TITLE */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-16 text-center"
        >
          <div className="text-[10px] text-white/40 tracking-[0.35em] uppercase">
            Select Operational Mission
          </div>
          <div className="mt-2 text-5xl font-black tracking-[0.25em] text-emerald-400">
            IRIS
          </div>
        </motion.div>

        {/* MISSION GRID */}
        <motion.div
          initial="hidden"
          animate="visible"
          variants={{
            hidden: {},
            visible: {
              transition: { staggerChildren: 0.08 },
            },
          }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-6xl"
        >
          {MISSIONS.map((m) => {
            const Icon = m.icon;
            const active = m.status === 'ACTIVE';

            return (
              <motion.button
                key={m.id}
                variants={{
                  hidden: { opacity: 0, y: 20 },
                  visible: { opacity: 1, y: 0 },
                }}
                onClick={() => navigate(`/${m.id}`)}
                className="
                  relative p-5 text-left
                  bg-black/40 border border-white/10
                  hover:border-emerald-400/50
                  transition-all duration-300
                  group
                "
              >
                {/* ACTIVE SCAN LINE */}
                {active && (
                  <motion.div
                    animate={{ opacity: [0.2, 0.6, 0.2] }}
                    transition={{ duration: 2.5, repeat: Infinity }}
                    className="absolute inset-x-0 top-0 h-[2px] bg-emerald-400"
                  />
                )}

                {/* HEADER */}
                <div className="flex items-center justify-between mb-4">
                  <Icon className="w-6 h-6 text-emerald-400" />
                  <span
                    className={`text-[9px] tracking-widest font-bold ${
                      active ? 'text-emerald-400' : 'text-white/30'
                    }`}
                  >
                    {m.status}
                  </span>
                </div>

                {/* TITLE */}
                <div className="text-sm font-black tracking-widest mb-2">
                  {m.title}
                </div>

                {/* DESC */}
                <div className="text-xs text-white/50 leading-relaxed h-10">
                  {m.description}
                </div>

                {/* FOOT */}
                <div className="mt-4 pt-3 border-t border-white/5 flex items-center justify-between">
                  <span className="text-[9px] text-white/40 tracking-widest">
                    DRONES: {m.drones}
                  </span>
                  <ChevronRight className="w-4 h-4 text-white/20 group-hover:text-white transition-colors" />
                </div>
              </motion.button>
            );
          })}
        </motion.div>
      </main>

      {/* FOOTER */}
      <footer className="h-12 border-t border-white/5 bg-black/30 flex items-center justify-between px-6 text-[9px] text-white/30 tracking-widest">
        <div>LOCATION: BLR-HUB</div>
        <div>IRIS CORE v4.0.2</div>
      </footer>
    </div>
  );
};

export default WelcomeScreen;
