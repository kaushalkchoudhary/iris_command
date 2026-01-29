import React, { useState, useEffect } from 'react';
import Card from '../UI/Card';
import { AlertTriangle, AlertCircle, Info, XCircle, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

const ALERT_DATA = {
    traffic: [
        { id: 1, severity: 'critical', location: 'MG Road Junction', reason: 'Major accident - 3 vehicles involved', time: '2m ago', icon: XCircle, screenshot: 'https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=800&h=600&fit=crop' },
        { id: 2, severity: 'high', location: 'Outer Ring Road', reason: 'Road construction blocking 2 lanes', time: '5m ago', icon: AlertTriangle, screenshot: 'https://images.unsplash.com/photo-1581092918056-0c4c3acd3789?w=800&h=600&fit=crop' },
        { id: 3, severity: 'medium', location: 'Whitefield Main', reason: 'Heavy rainfall causing slow movement', time: '8m ago', icon: AlertCircle, screenshot: 'https://images.unsplash.com/photo-1527786356703-4b100091cd2c?w=800&h=600&fit=crop' },
        { id: 4, severity: 'high', location: 'Silk Board Signal', reason: 'Traffic light malfunction', time: '12m ago', icon: AlertTriangle, screenshot: 'https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=800&h=600&fit=crop' },
        { id: 5, severity: 'low', location: 'Marathahalli Bridge', reason: 'Minor vehicle breakdown cleared', time: '15m ago', icon: Info, screenshot: 'https://images.unsplash.com/photo-1581092918056-0c4c3acd3789?w=800&h=600&fit=crop' },
    ],
    crowd: [
        { id: 101, severity: 'critical', location: 'Main Terminal', reason: 'Density exceeded safety threshold (5.2 p/m²)', time: '1m ago', icon: XCircle, screenshot: 'https://images.unsplash.com/photo-1517457373958-b7bdd4587205?w=800&h=600&fit=crop' },
        { id: 102, severity: 'high', location: 'Exit Gate B', reason: 'Bottleneck detected - flow restricted', time: '4m ago', icon: AlertTriangle, screenshot: 'https://images.unsplash.com/photo-1519750783826-e2420f4d687f?w=800&h=600&fit=crop' },
        { id: 103, severity: 'medium', location: 'Food Court', reason: 'Rapid density accumulation detected', time: '7m ago', icon: AlertCircle, screenshot: 'https://images.unsplash.com/photo-1556740738-b6a63e27c4df?w=800&h=600&fit=crop' },
        { id: 104, severity: 'critical', location: 'Plaza West', reason: 'Protest gathering - Unauthorized assembly', time: '12m ago', icon: XCircle, screenshot: 'https://images.unsplash.com/photo-1540910419892-f0c7620baea7?w=800&h=600&fit=crop' },
    ],
    safety: [
        { id: 201, severity: 'critical', location: 'Platform 04', reason: 'Unattended object for >15 minutes', time: '2m ago', icon: XCircle, screenshot: 'https://images.unsplash.com/photo-1596701062351-8c2c14d1fcd1?w=800&h=600&fit=crop' },
        { id: 202, severity: 'high', location: 'Loading Bay', reason: 'Person detected in restricted machinery zone', time: '6m ago', icon: AlertTriangle, screenshot: 'https://images.unsplash.com/photo-1504917595217-d4dc5ebe6122?w=800&h=600&fit=crop' },
        { id: 203, severity: 'critical', location: 'North Wing', reason: 'Smoke signature detected - Thermal Alert', time: '10m ago', icon: XCircle, screenshot: 'https://images.unsplash.com/photo-1534951009808-dfd1f57ba4b8?w=800&h=600&fit=crop' },
        { id: 204, severity: 'medium', location: 'Corridor 2', reason: 'Erratic behavior pattern identified', time: '15m ago', icon: Info, screenshot: 'https://images.unsplash.com/photo-1520038410233-7141f77e47a0?w=800&h=600&fit=crop' },
    ],
    perimeter: [
        { id: 301, severity: 'critical', location: 'South Fence Line', reason: 'Fence breach detected - Optic Fiber Alert', time: '1m ago', icon: XCircle, screenshot: 'https://images.unsplash.com/photo-1508191203025-a745d61416fb?w=800&h=600&fit=crop' },
        { id: 302, severity: 'high', location: 'restricted_zone_02', reason: 'Unauthorized vehicle movement', time: '4m ago', icon: AlertTriangle, screenshot: 'https://images.unsplash.com/photo-1566230985223-28952dd8629b?w=800&h=600&fit=crop' },
        { id: 303, severity: 'medium', location: 'Airspace 01', reason: 'Drone activity detected in no-fly zone', time: '9m ago', icon: AlertCircle, screenshot: 'https://images.unsplash.com/photo-1506947411487-a56738267384?w=800&h=600&fit=crop' },
        { id: 304, severity: 'high', location: 'Main Bunker', reason: 'Biometric access failure - Forced entry', time: '13m ago', icon: AlertTriangle, screenshot: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800&h=600&fit=crop' },
    ]
};

const RightPanel = ({ useCase = 'traffic' }) => {
    const [alerts, setAlerts] = useState(ALERT_DATA[useCase] || ALERT_DATA.traffic);
    const [selectedAlert, setSelectedAlert] = useState(null);

    useEffect(() => {
        setAlerts(ALERT_DATA[useCase] || ALERT_DATA.traffic);
    }, [useCase]);

    useEffect(() => {
        const interval = setInterval(() => {
            setAlerts(prev => {
                const updated = [...prev];
                const randomIndex = Math.floor(Math.random() * updated.length);
                updated[randomIndex] = {
                    ...updated[randomIndex],
                    time: `${Math.floor(Math.random() * 30)}m ago`
                };
                return updated;
            });
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    const getSeverityColor = (severity) => {
        switch (severity) {
            case 'critical': return 'text-red-500';
            case 'high': return 'text-orange-500';
            case 'medium': return 'text-yellow-500';
            default: return 'text-cyan-400';
        }
    };

    const getSeverityStyle = (severity) => {
        switch (severity) {
            case 'critical': return 'border-red-500/50 text-red-400 bg-red-500/10';
            case 'high': return 'border-orange-500/50 text-orange-400 bg-orange-500/10';
            case 'medium': return 'border-yellow-500/50 text-yellow-400 bg-yellow-500/10';
            default: return 'border-cyan-500/50 text-cyan-400 bg-cyan-500/10';
        }
    };

    return (
        <>
            <motion.div
                initial={{ x: 50, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ duration: 1, ease: "easeOut" }}
                className="w-[320px] h-full bg-[#070b14]/80 backdrop-blur-xl border-l border-white/5 pointer-events-auto overflow-hidden flex flex-col z-40"
            >
                {/* Header: Tactical Alert Stream */}
                <div className="p-4 border-b border-white/10 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/5 blur-3xl rounded-full -mr-16 -mt-16"></div>
                    <div className="flex flex-col gap-1 relative z-10">
                        <div className="flex items-center justify-between">
                            <span className="text-[10px] text-cyan-500/60 font-black uppercase tracking-[0.4em]">Intelligence Stream</span>
                            <div className="flex gap-1">
                                <div className="w-1 h-1 bg-cyan-500 rounded-full animate-ping"></div>
                                <div className="w-1 h-1 bg-cyan-500 rounded-full"></div>
                            </div>
                        </div>
                        <h1 className="text-2xl font-black text-white tracking-widest font-mono">LIVE_ALERTS</h1>
                    </div>
                </div>

                {/* Technical Alerts Scroll Area */}
                <div className="flex-1 overflow-y-auto no-scrollbar py-2">
                    {alerts.map((alert, index) => {
                        const Icon = alert.icon;
                        const isCritical = alert.severity === 'critical';

                        return (
                            <motion.div
                                key={alert.id}
                                initial={{ x: 20, opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                transition={{ delay: index * 0.05 }}
                                onClick={() => setSelectedAlert(alert)}
                                className={clsx(
                                    "group relative flex border-b border-white/5 cursor-pointer hover:bg-white/[0.03] transition-all duration-300",
                                    isCritical && "bg-red-500/5"
                                )}
                            >
                                {/* Side Accent Line */}
                                <div className={clsx(
                                    "w-1 transition-all duration-300",
                                    isCritical ? "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]" : "bg-cyan-500/20 group-hover:bg-cyan-500"
                                )}></div>

                                {/* Content */}
                                <div className="flex-1 p-3 flex gap-3">
                                    {/* Thumbnail container */}
                                    <div className="w-16 h-16 shrink-0 border border-white/10 relative overflow-hidden rounded-sm bg-black/40">
                                        <img
                                            src={alert.screenshot}
                                            className="w-full h-full object-cover grayscale opacity-60 group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-500"
                                            alt="Target"
                                        />
                                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent"></div>
                                        <div className="absolute top-0.5 right-0.5"><Icon className={clsx("w-2.5 h-2.5", getSeverityColor(alert.severity))} /></div>
                                    </div>

                                    {/* Text Info */}
                                    <div className="flex-1 min-w-0 font-mono flex flex-col justify-between py-0.5">
                                        <div>
                                            <div className="flex items-center justify-between mb-0.5">
                                                <span className="text-[9px] text-cyan-500/40 font-bold uppercase tracking-tighter">LVL_{index.toString().padStart(2, '0')}</span>
                                                <span className="text-[9px] text-white/30">{alert.time}</span>
                                            </div>
                                            <h3 className="text-[11px] text-white/90 font-black uppercase truncate leading-none">
                                                {alert.location}
                                            </h3>
                                        </div>

                                        <div className="flex items-center justify-between mt-1 pt-1 border-t border-white/[0.03]">
                                            <span className={clsx("text-[9px] font-bold uppercase tracking-wider", getSeverityColor(alert.severity))}>
                                                {alert.severity}
                                            </span>
                                            <div className="flex items-center gap-1">
                                                <div className="w-6 h-0.5 bg-white/5 rounded-full overflow-hidden">
                                                    <motion.div
                                                        animate={{ x: [-24, 24] }}
                                                        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                                                        className="w-6 h-full bg-cyan-500/40"
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>

                {/* Footer Branding */}
                <div className="p-4 border-t border-white/5 bg-black/60 flex items-center justify-between">
                    <div className="flex flex-col">
                        <span className="text-[8px] text-white/20 font-black uppercase tracking-[0.3em]">Surveillance Core</span>
                        <span className="text-[9px] text-cyan-500/40 font-mono font-bold">MODE_VER_4.0.2</span>
                    </div>
                    <div className="flex gap-1 h-3 items-end">
                        {[...Array(5)].map((_, i) => (
                            <motion.div
                                key={i}
                                animate={{ height: [4, 12, 4] }}
                                transition={{ duration: 1, repeat: Infinity, delay: i * 0.1 }}
                                className="w-1 bg-cyan-500/20"
                            />
                        ))}
                    </div>
                </div>
            </motion.div>

            {/* Image Popup Modal */}
            <AnimatePresence>
                {selectedAlert && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setSelectedAlert(null)}
                        className="fixed inset-0 bg-black/95 backdrop-blur-md z-[100] flex items-center justify-center p-8 pointer-events-auto"
                    >
                        <motion.div
                            initial={{ scale: 0.95, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.95, opacity: 0 }}
                            onClick={(e) => e.stopPropagation()}
                            className="relative max-w-5xl w-full bg-[#0a1120] border border-white/10 overflow-hidden shadow-2xl"
                        >
                            {/* Close Button */}
                            <button
                                onClick={() => setSelectedAlert(null)}
                                className="absolute top-6 right-6 z-20 group"
                            >
                                <div className="absolute inset-0 bg-red-500 blur-md opacity-0 group-hover:opacity-40 transition-opacity"></div>
                                <div className="relative p-2 bg-black/50 border border-white/10 group-hover:border-red-500/50 transition-colors">
                                    <X className="w-5 h-5 text-white/50 group-hover:text-red-400" />
                                </div>
                            </button>

                            {/* Main Display */}
                            <div className="flex flex-col lg:flex-row h-[70vh]">
                                {/* Capture View */}
                                <div className="flex-1 bg-black relative overflow-hidden group/img">
                                    <img
                                        src={selectedAlert.screenshot}
                                        alt={selectedAlert.location}
                                        className="w-full h-full object-cover opacity-80"
                                    />
                                    <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent"></div>

                                    {/* HUD Overlays on Image */}
                                    <div className="absolute top-6 left-6 flex flex-col gap-2">
                                        <div className="px-3 py-1 bg-black/60 border-l-2 border-cyan-500 text-[10px] text-cyan-400 font-black uppercase tracking-widest backdrop-blur-md">
                                            LIVE_CAPTURE_STREAM
                                        </div>
                                        <div className="px-3 py-1 bg-black/60 border-l-2 border-white/30 text-[10px] text-white/60 font-black uppercase tracking-widest backdrop-blur-md">
                                            TIMESTAMP: 2026.05.10_14:42:01
                                        </div>
                                    </div>

                                    {/* Scanning Bar Animation */}
                                    <motion.div
                                        animate={{ top: ['0%', '100%'] }}
                                        transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                                        className="absolute left-0 right-0 h-0.5 bg-cyan-500/30 shadow-[0_0_15px_rgba(6,182,212,0.5)] z-10"
                                    />
                                </div>

                                {/* Modal Technical Info Sidebar */}
                                <div className="w-full lg:w-96 bg-[#070b14] border-l border-white/10 p-8 flex flex-col font-mono">
                                    <div className="mb-8">
                                        <div className="text-[10px] text-cyan-500/50 font-black uppercase tracking-[0.4em] mb-2">Primary Incident</div>
                                        <h2 className="text-3xl font-black text-white uppercase tracking-tighter leading-none mb-1">
                                            {selectedAlert.location}
                                        </h2>
                                        <div className={`inline-block px-3 py-1 text-[10px] font-black border uppercase mt-4 ${getSeverityStyle(selectedAlert.severity)}`}>
                                            Status: {selectedAlert.severity}
                                        </div>
                                    </div>

                                    <div className="flex-1 space-y-6">
                                        <div>
                                            <div className="text-[10px] text-white/30 font-black uppercase tracking-widest mb-2 border-b border-white/5 pb-1">Context Analysis</div>
                                            <p className="text-sm text-white/70 leading-relaxed italic">
                                                "{selectedAlert.reason}"
                                            </p>
                                        </div>

                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <div className="text-[10px] text-white/30 font-black uppercase tracking-widest mb-1">Alert Age</div>
                                                <div className="text-xs text-cyan-400 font-bold">{selectedAlert.time}</div>
                                            </div>
                                            <div>
                                                <div className="text-[10px] text-white/30 font-black uppercase tracking-widest mb-1">Confidence</div>
                                                <div className="text-xs text-emerald-400 font-bold">98.4%</div>
                                            </div>
                                        </div>

                                        <div>
                                            <div className="text-[10px] text-white/30 font-black uppercase tracking-widest mb-3 border-b border-white/5 pb-1">Threat Vector</div>
                                            <div className="flex gap-2">
                                                {[...Array(4)].map((_, i) => (
                                                    <div key={i} className={`h-8 flex-1 bg-white/5 border border-white/10 relative overflow-hidden`}>
                                                        <motion.div
                                                            initial={{ height: 0 }}
                                                            animate={{ height: `${Math.random() * 80 + 20}%` }}
                                                            className={`absolute bottom-0 left-0 right-0 ${i === 0 ? 'bg-red-500/40' : 'bg-cyan-500/20'}`}
                                                        />
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    <button
                                        onClick={() => setSelectedAlert(null)}
                                        className="mt-8 w-full py-4 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-cyan-500/50 text-white font-black uppercase tracking-[0.3em] text-xs transition-all duration-300"
                                    >
                                        Close_Terminal
                                    </button>
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
