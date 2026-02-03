import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
    Activity,
    Car,
    GitBranch,
    Crosshair,
    ChevronRight,
    Users,
    ShieldCheck,
    Cpu
} from 'lucide-react';

const USE_CASES = [
    {
        id: 'congestion',
        title: 'Congestion Analytics',
        description: 'Real-time heatmap visualization with speed distribution and congestion index monitoring.',
        icon: Activity,
        color: 'text-cyan-400',
        borderColor: 'border-cyan-400/30',
        bgGlow: 'bg-cyan-400/10',
        stats: '12 Active Nodes',
        status: 'Operational'
    },
    {
        id: 'vehicle',
        title: 'Vehicle Analytics',
        description: 'Per-class vehicle counting and classification — cars, trucks, buses, vans, motorcycles, bicycles.',
        icon: Car,
        color: 'text-emerald-400',
        borderColor: 'border-emerald-400/30',
        bgGlow: 'bg-emerald-400/10',
        stats: '12 Active Nodes',
        status: 'Operational'
    },
    {
        id: 'flow',
        title: 'Traffic Flow',
        description: 'Movement trail visualization with slow/fast speed labels and mobility index tracking.',
        icon: GitBranch,
        color: 'text-purple-400',
        borderColor: 'border-purple-400/30',
        bgGlow: 'bg-purple-400/10',
        stats: '12 Active Nodes',
        status: 'Operational'
    },
    {
        id: 'forensics',
        title: 'IRIS Forensics',
        description: 'Prompt-based intelligent detection — ask anything about the scene in natural language.',
        icon: Crosshair,
        color: 'text-amber-400',
        borderColor: 'border-amber-400/30',
        bgGlow: 'bg-amber-400/10',
        stats: 'Neural Engine',
        status: 'Operational'
    },
    {
        id: 'crowd',
        title: 'Crowd Analytics',
        description: 'Density mapping, bottleneck identification, and real-time crowd flow monitoring.',
        icon: Users,
        color: 'text-teal-400',
        borderColor: 'border-teal-400/30',
        bgGlow: 'bg-teal-400/10',
        stats: '08 Active Nodes',
        status: 'Idle'
    },
    {
        id: 'safety',
        title: 'Public Safety',
        description: 'Anomaly detection, unattended object tracking, and automated threat alerting.',
        icon: ShieldCheck,
        color: 'text-emerald-400',
        borderColor: 'border-emerald-400/30',
        bgGlow: 'bg-emerald-400/10',
        stats: '23 Active Nodes',
        status: 'Operational'
    }
];

const WelcomeScreen = () => {
    const navigate = useNavigate();
    return (
        <div className="fixed inset-0 bg-[#050a14] z-[100] flex flex-col overflow-y-auto">
            {/* Background Effects */}
            <div className="fixed inset-0 z-0 opacity-20 pointer-events-none"
                style={{
                    backgroundImage: 'linear-gradient(rgba(0, 255, 255, 0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 255, 255, 0.2) 1px, transparent 1px)',
                    backgroundSize: '30px 30px'
                }}>
            </div>
            <div className="fixed inset-0 z-0 bg-[radial-gradient(circle_at_center,transparent_0%,#000_100%)] opacity-80 pointer-events-none" />

            {/* Top Bar — sticky */}
            <header className="sticky top-0 z-20 w-full h-12 border-b border-white/5 bg-black/40 backdrop-blur-md flex items-center justify-between px-8 shrink-0">
                <div className="flex items-center gap-6" />
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                        <span className="text-[10px] text-emerald-500 font-bold tracking-widest uppercase">Nodes: Operational</span>
                    </div>
                </div>
            </header>

            {/* Content — no scroll, fit viewport */}
            <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-8 py-4 max-w-7xl mx-auto w-full">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-row items-center gap-4 md:gap-6 mb-6"
                >
                    {/* Logo on the left */}
                    <div className="relative w-20 h-20 md:w-36 md:h-36 flex-shrink-0 self-center">
                        <img
                            src="https://upload.wikimedia.org/wikipedia/en/thumb/f/fc/Bangalore_City_Police_Logo.png/250px-Bangalore_City_Police_Logo.png"
                            alt="IRIS Logo"
                            className="w-full h-full object-contain filter drop-shadow-[0_0_30px_rgba(6,182,212,0.3)] animate-pulse"
                        />
                        <div className="absolute inset-0 bg-cyan-500/10 blur-3xl -z-10 rounded-full animate-pulse" />
                    </div>

                    <div className="text-left">
                        <h2 className="text-2xl md:text-5xl font-black text-white mb-1 tracking-tighter uppercase leading-[0.85]">
                            IRIS <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">COMMAND</span>
                        </h2>
                        <p className="text-cyan-400 font-mono tracking-[0.1em] uppercase text-[10px] md:text-base font-bold mb-1 md:mb-2">
                            Integrated Realtime Intelligence System
                        </p>
                        <div className="w-16 md:w-24 h-0.5 bg-cyan-500 mb-2 md:mb-3 opacity-40" />
                        <p className="text-white/40 max-w-2xl font-medium text-xs md:text-sm leading-relaxed hidden md:block">
                            Deploy high-altitude vision nodes and specialized AI modules<br/>to begin real-time tactical oversight.
                        </p>
                    </div>
                </motion.div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full">
                    {USE_CASES.map((useCase, idx) => (
                        <motion.button
                            key={useCase.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            onClick={() => navigate(`/${useCase.id}`)}
                            className={`group relative flex flex-col text-left p-4 bg-white/5 border ${useCase.borderColor} hover:border-white/30 transition-all duration-500 overflow-hidden text-white`}
                        >
                            {/* Card Hover Glow */}
                            <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 ${useCase.bgGlow}`} />

                            {/* Card Decoration */}
                            <div className="absolute top-0 right-0 w-12 h-12 overflow-hidden pointer-events-none">
                                <div className={`absolute top-0 right-0 w-[140%] h-[2px] ${useCase.color.replace('text-', 'bg-')} transform rotate-45 translate-x-1/2 -translate-y-1/2 opacity-20 group-hover:opacity-100 transition-opacity`} />
                            </div>

                            <div className="relative z-10">
                                <div className={`w-9 h-9 rounded-lg bg-black/40 border border-white/10 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform duration-500`}>
                                    <useCase.icon className={`w-4 h-4 ${useCase.color}`} />
                                </div>

                                <h3 className="text-sm font-black tracking-wider mb-1 group-hover:text-cyan-400 transition-colors uppercase">{useCase.title}</h3>
                                <p className="text-xs text-white/50 leading-relaxed mb-3">
                                    {useCase.description}
                                </p>

                                <div className="pt-2 border-t border-white/5 flex items-center justify-between">
                                    <div className="flex flex-col">
                                        <span className="text-[9px] text-white/30 uppercase font-black tracking-widest">{useCase.stats}</span>
                                        <span className={`text-[9px] font-bold uppercase ${useCase.status === 'Operational' ? 'text-emerald-500' : useCase.status === 'Idle' ? 'text-cyan-500/50' : 'text-white/20'}`}>{useCase.status}</span>
                                    </div>
                                    <ChevronRight className="w-4 h-4 text-white/20 group-hover:text-white group-hover:translate-x-1 transition-all" />
                                </div>
                            </div>

                            {/* Corner Accents */}
                            <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-white/20" />
                            <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-white/20" />
                        </motion.button>
                    ))}
                </div>

            </main>

            {/* Footer Status — sticky bottom */}
            <footer className="sticky bottom-0 z-20 w-full h-12 border-t border-white/5 bg-black/40 backdrop-blur-md flex items-center justify-between px-8 text-[10px] font-mono text-white/20 shrink-0">
                <div className="flex gap-8">
                    <div className="flex gap-2"><span className="text-white/40">GEO_LOC:</span> BANGALORE_HUB</div>
                    <div className="flex gap-2"><span className="text-white/40">LATENCY:</span> 12.4ms</div>
                    <div className="flex gap-2"><span className="text-white/40">PROC_LOAD:</span> [|||.......] 22%</div>
                </div>
                <div>VERSION 4.0.2 // STABLE_DIST</div>
            </footer>
        </div>
    );
};

export default WelcomeScreen;
