import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, AlertCircle, XCircle, X, Clock, Activity, Users, Gauge } from 'lucide-react';
import clsx from 'clsx';

const API_BASE_URL = import.meta.env.DEV
  ? '/api'
  : `http://${window.location.hostname}:9010`;

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

const severityConfig = {
  critical: {
    color: 'text-red-400',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/30',
    icon: XCircle,
    label: 'CRITICAL'
  },
  high: {
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30',
    icon: AlertTriangle,
    label: 'HIGH'
  },
  medium: {
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500/30',
    icon: AlertCircle,
    label: 'MEDIUM'
  },
};

const formatTimeAgo = (timestamp) => {
  const seconds = Math.floor(Date.now() / 1000 - timestamp);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
};

const RightPanel = ({ useCase = 'traffic' }) => {
  const [alerts, setAlerts] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState(null);

  // Fetch alerts from backend
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/alerts?limit=30`);
        if (res.ok) {
          const data = await res.json();
          setAlerts(data.alerts || []);
        }
      } catch (e) {
        console.error('Failed to fetch alerts:', e);
      }
    };

    fetchAlerts();
    const interval = setInterval(fetchAlerts, 3000); // Poll every 3 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      {/* ================= PANEL ================= */}
      <motion.div
        initial={{ x: 40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        className="w-[320px] h-full bg-[#070b14]/80 border-l border-white/10 flex flex-col z-40"
      >
        {/* HEADER */}
        <div className="p-4 border-b border-white/10">
          <div className="text-[10px] text-cyan-500/60 font-black uppercase tracking-[0.4em]">
            Intelligence Stream
          </div>
          <div className="text-xl font-black font-mono tracking-widest text-white flex items-center gap-2">
            LIVE_ALERTS
            {alerts.length > 0 && (
              <span className="px-2 py-0.5 bg-red-500/20 text-red-400 text-xs rounded">
                {alerts.length}
              </span>
            )}
          </div>
        </div>

        {/* ALERT LIST */}
        <div className="flex-1 overflow-y-auto">
          {alerts.length === 0 ? (
            <div className="p-6 text-center">
              <div className="text-white/20 text-sm font-mono">No active alerts</div>
              <div className="text-white/10 text-xs mt-1">
                Alerts trigger when congestion exceeds 40%
              </div>
            </div>
          ) : (
            alerts.map((alert) => {
              const config = severityConfig[alert.severity] || severityConfig.medium;
              const Icon = config.icon;
              const location = DRONE_REGION_MAP[alert.source] || alert.source;

              // Extract drone number from source (e.g., "bcpdrone5" -> "DRONE 5")
              const droneMatch = alert.source.match(/bcpdrone(\d+)/i);
              const droneLabel = droneMatch ? `DRONE ${droneMatch[1]}` : alert.source.toUpperCase();

              return (
                <div
                  key={alert.id}
                  onClick={() => setSelectedAlert(alert)}
                  className={clsx(
                    'flex border-b border-white/5 hover:bg-white/[0.03] cursor-pointer transition-colors',
                    config.bgColor
                  )}
                >
                  {/* side accent */}
                  <div className={clsx('w-1', config.color.replace('text-', 'bg-'))} />

                  <div className="p-3 flex gap-3 flex-1 font-mono">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-black uppercase text-white">
                          {location}
                        </span>
                        <span className="text-[9px] font-bold text-cyan-500/70 bg-cyan-500/10 px-1.5 py-0.5">
                          {droneLabel}
                        </span>
                      </div>
                      <div className={clsx('text-[9px] uppercase font-bold mt-1', config.color)}>
                        {config.label} • {alert.congestion}% CONGESTION
                      </div>
                      <div className="text-[9px] text-white/40 mt-1 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {alert.time_str} ({formatTimeAgo(alert.timestamp)})
                      </div>
                    </div>

                    <Icon className={clsx('w-5 h-5 mt-1', config.color)} />
                  </div>
                </div>
              );
            })
          )}
        </div>
      </motion.div>

      {/* ================= ALERT DETAIL MODAL ================= */}
      <AnimatePresence>
        {selectedAlert && (
          <motion.div
            className="fixed inset-0 bg-black/90 z-[100] flex items-center justify-center p-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedAlert(null)}
          >
            <motion.div
              onClick={(e) => e.stopPropagation()}
              className="bg-[#0a1120] max-w-4xl w-full border border-white/10 relative overflow-hidden"
              initial={{ scale: 0.96 }}
              animate={{ scale: 1 }}
            >
              {/* Close button */}
              <button
                onClick={() => setSelectedAlert(null)}
                className="absolute top-4 right-4 z-10 border border-white/10 p-2 bg-black/50 hover:bg-white/10 transition-colors"
              >
                <X className="w-4 h-4 text-white/70" />
              </button>

              {/* Header */}
              <div className="p-4 border-b border-white/10 bg-black/30">
                <div className="flex items-center gap-3">
                  {(() => {
                    const config = severityConfig[selectedAlert.severity] || severityConfig.medium;
                    const Icon = config.icon;
                    const droneMatch = selectedAlert.source.match(/bcpdrone(\d+)/i);
                    const droneLabel = droneMatch ? `DRONE ${droneMatch[1]}` : selectedAlert.source.toUpperCase();
                    return (
                      <>
                        <Icon className={clsx('w-6 h-6', config.color)} />
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-lg font-black text-white uppercase tracking-wide">
                              {DRONE_REGION_MAP[selectedAlert.source] || selectedAlert.source}
                            </span>
                            <span className="text-xs font-bold text-cyan-400 bg-cyan-500/20 px-2 py-0.5">
                              {droneLabel}
                            </span>
                          </div>
                          <div className={clsx('text-sm font-bold', config.color)}>
                            {config.label} ALERT • {selectedAlert.congestion}% Congestion
                          </div>
                        </div>
                      </>
                    );
                  })()}
                </div>
              </div>

              {/* Content */}
              <div className="flex flex-col md:flex-row">
                {/* Screenshot */}
                <div className="flex-1 relative bg-black">
                  <img
                    src={`${API_BASE_URL}/alerts/${selectedAlert.id}/screenshot`}
                    alt="Alert Screenshot"
                    className="w-full h-auto max-h-[50vh] object-contain"
                    onError={(e) => {
                      e.target.src = '/placeholder-alert.jpg';
                      e.target.onerror = null;
                    }}
                  />
                  <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/70 text-xs text-white/60 font-mono">
                    {selectedAlert.time_str}
                  </div>
                </div>

                {/* Metrics Panel */}
                <div className="w-full md:w-64 p-4 bg-black/20 border-t md:border-t-0 md:border-l border-white/10">
                  <div className="text-xs text-cyan-500/60 font-black uppercase tracking-widest mb-4">
                    Metrics Snapshot
                  </div>

                  <div className="space-y-3">
                    {/* Congestion */}
                    <div className="flex items-center gap-3">
                      <Gauge className="w-4 h-4 text-red-400" />
                      <div className="flex-1">
                        <div className="text-[10px] text-white/40 uppercase">Congestion</div>
                        <div className="text-lg font-black text-red-400">
                          {selectedAlert.metrics?.congestion_index || selectedAlert.congestion}%
                        </div>
                      </div>
                    </div>

                    {/* Density */}
                    <div className="flex items-center gap-3">
                      <Activity className="w-4 h-4 text-orange-400" />
                      <div className="flex-1">
                        <div className="text-[10px] text-white/40 uppercase">Traffic Density</div>
                        <div className="text-lg font-black text-orange-400">
                          {selectedAlert.metrics?.traffic_density || 0}%
                        </div>
                      </div>
                    </div>

                    {/* Mobility */}
                    <div className="flex items-center gap-3">
                      <Activity className="w-4 h-4 text-cyan-400" />
                      <div className="flex-1">
                        <div className="text-[10px] text-white/40 uppercase">Mobility Index</div>
                        <div className="text-lg font-black text-cyan-400">
                          {selectedAlert.metrics?.mobility_index || 0}
                        </div>
                      </div>
                    </div>

                    {/* Detection Count */}
                    <div className="flex items-center gap-3">
                      <Users className="w-4 h-4 text-white/60" />
                      <div className="flex-1">
                        <div className="text-[10px] text-white/40 uppercase">Vehicles Detected</div>
                        <div className="text-lg font-black text-white">
                          {selectedAlert.metrics?.detection_count || 0}
                        </div>
                      </div>
                    </div>

                    {/* Speed Distribution */}
                    <div className="mt-4 pt-3 border-t border-white/10">
                      <div className="text-[10px] text-white/40 uppercase mb-2">Speed Distribution</div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-red-400">Stalled</span>
                          <span className="text-white/70">{selectedAlert.metrics?.stalled_pct || 0}%</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-orange-400">Slow</span>
                          <span className="text-white/70">{selectedAlert.metrics?.slow_pct || 0}%</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-yellow-400">Medium</span>
                          <span className="text-white/70">{selectedAlert.metrics?.medium_pct || 0}%</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-emerald-400">Fast</span>
                          <span className="text-white/70">{selectedAlert.metrics?.fast_pct || 0}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default RightPanel;
