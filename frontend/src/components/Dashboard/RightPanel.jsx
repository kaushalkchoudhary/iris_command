import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, AlertCircle, Info, XCircle, X } from 'lucide-react';
import clsx from 'clsx';
import WebRTCVideo from '../UI/WebRTCVideo';

/* ============================
   STREAM CONFIG
============================ */
const WEBRTC_BASE_URL = import.meta.env.DEV
  ? '/webrtc'
  : `http://${window.location.hostname}:8889`;

/* ============================
   DRONE → LOCATION MAP
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

/* ============================
   ALERT TEMPLATES (NEW LOGIC)
============================ */
const ALERT_TEMPLATES = {
  traffic: [
    { severity: 'critical', reason: 'Major accident detected', icon: XCircle },
    { severity: 'high', reason: 'Road obstruction reported', icon: AlertTriangle },
    { severity: 'medium', reason: 'Congestion building up', icon: AlertCircle },
    { severity: 'low', reason: 'Minor slowdown observed', icon: Info },
  ],
  crowd: [
    { severity: 'critical', reason: 'Density exceeded safety limit', icon: XCircle },
    { severity: 'high', reason: 'Crowd bottleneck forming', icon: AlertTriangle },
    { severity: 'medium', reason: 'Rapid density increase', icon: AlertCircle },
  ],
  safety: [
    { severity: 'critical', reason: 'Unattended object detected', icon: XCircle },
    { severity: 'high', reason: 'Restricted zone intrusion', icon: AlertTriangle },
  ],
  perimeter: [
    { severity: 'critical', reason: 'Fence breach detected', icon: XCircle },
    { severity: 'high', reason: 'Unauthorized movement detected', icon: AlertTriangle },
  ],
};

const severityColor = (severity) => {
  switch (severity) {
    case 'critical': return 'text-red-400';
    case 'high': return 'text-orange-400';
    case 'medium': return 'text-yellow-400';
    default: return 'text-cyan-400';
  }
};

const RightPanel = ({ useCase = 'traffic', sources = [] }) => {
  const [selectedAlert, setSelectedAlert] = useState(null);

  /* ============================
     BUILD ALERTS FROM SOURCES
  ============================ */
  const alerts = useMemo(() => {
    const templates = ALERT_TEMPLATES[useCase] || ALERT_TEMPLATES.traffic;

    return sources.map((src, i) => {
      const template = templates[i % templates.length];
      return {
        id: src.id,
        stream: src.stream,
        label: src.label,
        location: DRONE_REGION_MAP[src.id] || 'Remote Stream',
        severity: template.severity,
        reason: template.reason,
        icon: template.icon,
        time: `${Math.floor(Math.random() * 20) + 1}m ago`,
      };
    });
  }, [sources, useCase]);

  return (
    <>
      {/* ================= PANEL ================= */}
      <motion.div
        initial={{ x: 40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        className="w-[320px] h-full bg-[#070b14]/80 border-l border-white/10 flex flex-col z-40"
      >
        {/* HEADER — OLD STYLE */}
        <div className="p-4 border-b border-white/10">
          <div className="text-[10px] text-cyan-500/60 font-black uppercase tracking-[0.4em]">
            Intelligence Stream
          </div>
          <div className="text-xl font-black font-mono tracking-widest text-white">
            LIVE_ALERTS
          </div>
        </div>

        {/* ALERT LIST — OLD STYLE */}
        <div className="flex-1 overflow-y-auto">
          {alerts.map((alert) => {
            const Icon = alert.icon;
            return (
              <div
                key={alert.id}
                onClick={() => setSelectedAlert(alert)}
                className="flex border-b border-white/5 hover:bg-white/[0.03] cursor-pointer"
              >
                {/* side accent */}
                <div className="w-1 bg-cyan-500/30" />

                <div className="p-3 flex gap-3 flex-1 font-mono">
                  <div className="flex-1">
                    <div className="text-[11px] font-black uppercase text-white">
                      {alert.location}
                    </div>
                    <div className={clsx(
                      'text-[9px] uppercase font-bold',
                      severityColor(alert.severity)
                    )}>
                      {alert.severity}
                    </div>
                    <div className="text-[9px] text-white/40 mt-1">
                      {alert.time}
                    </div>
                  </div>

                  <Icon className={clsx(
                    'w-4 h-4 mt-1',
                    severityColor(alert.severity)
                  )} />
                </div>
              </div>
            );
          })}
        </div>
      </motion.div>

      {/* ================= MODAL (FROM NEW) ================= */}
      <AnimatePresence>
        {selectedAlert && (
          <motion.div
            className="fixed inset-0 bg-black/90 z-[100] flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedAlert(null)}
          >
            <motion.div
              onClick={(e) => e.stopPropagation()}
              className="bg-[#0a1120] w-[80vw] h-[70vh] border border-white/10 relative"
              initial={{ scale: 0.96 }}
              animate={{ scale: 1 }}
            >
              <button
                onClick={() => setSelectedAlert(null)}
                className="absolute top-4 right-4 border border-white/10 p-2"
              >
                <X className="w-4 h-4 text-white/70" />
              </button>

              <WebRTCVideo
                src={`${WEBRTC_BASE_URL}/${selectedAlert.stream}/whep`}
                autoPlay
                muted
                playsInline
                className="w-full h-full object-cover"
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default RightPanel;
