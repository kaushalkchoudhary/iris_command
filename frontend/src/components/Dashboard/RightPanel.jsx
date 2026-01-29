import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import WebRTCVideo from '../UI/WebRTCVideo';

/* ============================
   STREAM CONFIG
============================ */
const WEBRTC_BASE_URL = import.meta.env.DEV
  ? '/webrtc'
  : `http://${window.location.hostname}:8889`;

/* ============================
   DRONE FEEDS
============================ */
const DRONE_REGION_MAP = {
  bcpdrone1: 'MG Road Junction',
  bcpdrone2: 'Outer Ring Road',
  bcpdrone3: 'Whitefield Main Road',
  bcpdrone4: 'Silk Board Signal',
  bcpdrone5: 'Marathahalli Bridge',
  bcpdrone6: 'Electronic City Flyover',
  bcpdrone7: 'Hebbal Flyover',
  bcpdrone8: 'KR Puram Junction',
  bcpdrone9: 'Bellandur Lake Road',
  bcpdrone10: 'HSR Layout Sector 7',
  bcpdrone11: 'Yelahanka New Town',
  bcpdrone12: 'JP Nagar Phase 6',
};

const RightPanel = ({ sources = [] }) => {
  const [selectedFeed, setSelectedFeed] = useState(null);

  return (
    <>
      {/* RIGHT PANEL */}
      <motion.div
        initial={{ x: 32, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="
          w-[360px] h-full
          bg-[#070b14]/90 backdrop-blur-xl
          border-l border-emerald-500/15
          flex flex-col z-40 font-mono
        "
      >
        {/* HEADER */}
        <div className="px-6 py-4 border-b border-emerald-500/15">
          <span className="text-[10px] text-emerald-400/60 uppercase tracking-[0.35em] font-bold">
            Sensor Stream
          </span>
          <h1 className="mt-1 text-2xl font-black text-emerald-400 tracking-widest">
            LIVE FEEDS
          </h1>
        </div>

        {/* FEED LIST */}
        <div className="flex-1 overflow-y-auto no-scrollbar">
          {sources.map((feed, index) => {
            const region = DRONE_REGION_MAP[feed.id] || (feed.label === 'UPLOADED' ? 'Local Video Feed' : 'Remote Stream');

            // For modal compatibility
            const feedWithRegion = { ...feed, region };

            return (
              <motion.div
                key={feed.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.03 }}
                onClick={() => setSelectedFeed(feedWithRegion)}
                className="
                  px-6 py-4
                  border-b border-white/5
                  cursor-pointer
                  hover:bg-white/[0.03]
                  transition-colors
                "
              >
                <div className="flex items-center justify-between gap-4">
                  {/* REGION */}
                  <div className="flex-1">
                    <div className="text-base font-black text-white uppercase tracking-wide">
                      {region}
                    </div>

                    {/* LIVE INDICATOR */}
                    <div className="mt-2 flex items-center gap-2 text-[11px] text-emerald-400 uppercase tracking-widest font-bold">
                      <span className="relative">
                        <span className="absolute inset-0 rounded-full bg-emerald-400/30 animate-ping" />
                        <span className="relative block w-2 h-2 rounded-full bg-emerald-400" />
                      </span>
                      LIVE FEED
                    </div>
                  </div>

                  {/* PASSIVE CARET */}
                  <span className="text-white/20 text-sm">›</span>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* FOOTER */}
        <div className="px-6 py-3 border-t border-emerald-500/15 bg-black/50">
          <span className="text-[10px] text-white/40 tracking-widest">
            SELECT FEED TO INSPECT
          </span>
        </div>
      </motion.div>

      {/* CENTER MODAL */}
      <AnimatePresence>
        {selectedFeed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedFeed(null)}
            className="
              fixed inset-0 bg-black/95 backdrop-blur-md
              z-[100] flex items-center justify-center p-8
            "
          >
            <motion.div
              initial={{ scale: 0.97, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.97, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="
                relative max-w-6xl w-full h-[72vh]
                bg-[#0a1120] border border-emerald-500/15
                flex overflow-hidden font-mono
              "
            >
              {/* CLOSE */}
              <button
                onClick={() => setSelectedFeed(null)}
                className="
                  absolute top-6 right-6 z-20
                  bg-black/60 border border-white/10
                  p-2 hover:border-emerald-500/50
                "
              >
                <X className="w-5 h-5 text-white/60 hover:text-emerald-400" />
              </button>

              {/* VIDEO */}
              <div className="flex-1 relative bg-black">
                <WebRTCVideo
                  src={`${WEBRTC_BASE_URL}/${selectedFeed.stream}/whep`}
                  autoPlay
                  muted
                  playsInline
                  className="w-full h-full object-cover"
                />

                {/* HUD */}
                <div className="absolute top-6 left-6 space-y-3">
                  <div className="px-4 py-1.5 bg-black/70 border-l-4 border-emerald-400 text-sm text-emerald-400 font-black tracking-widest">
                    LIVE FEED
                  </div>
                  <div className="px-4 py-1.5 bg-black/60 border-l-2 border-white/20 text-xs text-white/70">
                    {new Date().toISOString().replace('T', ' ').slice(0, 19)}
                  </div>
                </div>
              </div>

              {/* INFO PANEL */}
              <div className="w-[420px] bg-[#070b14] border-l border-emerald-500/15 p-10">
                <span className="text-xs text-emerald-400/60 uppercase tracking-[0.35em]">
                  Active Feed
                </span>

                <h2 className="mt-2 text-4xl font-black text-white uppercase tracking-tight">
                  {selectedFeed.region}
                </h2>

                {/* DRONE LABEL — ONLY HERE */}
                <div className="mt-2 text-sm text-white/50 uppercase tracking-widest">
                  {selectedFeed.label}
                </div>

                <p className="mt-6 text-base text-white/70 leading-relaxed">
                  Real-time aerial surveillance feed from <strong>{selectedFeed.label}</strong>.
                  Edge analytics and tracking overlays are currently active.
                </p>

                <button
                  onClick={() => setSelectedFeed(null)}
                  className="
                    mt-10 w-full py-5
                    bg-white/5 hover:bg-white/10
                    border border-white/10 hover:border-emerald-500/50
                    text-white font-black uppercase tracking-[0.35em] text-sm
                  "
                >
                  CLOSE VIEW
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default RightPanel;
