import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
    Car,
    Users,
    ShieldAlert,
    Fence,
    ChevronRight,
    Search,
    Power,
    Settings,
    Cpu
} from 'lucide-react';

const USE_CASES = [
    {
        id: 'traffic',
        title: 'Aerial Traffic Intelligence',
        description: 'UAV-deployed vehicle tracking, flow analysis, and congestion forecasting via real-time drone vision.',
        icon: Car,
        color: 'text-cyan-400',
        borderColor: 'border-cyan-400/30',
        bgGlow: 'bg-cyan-400/10',
        stats: '14 Active Nodes',
        status: 'Operational'
    },
    {
        id: 'crowd',
        title: 'Crowd Analytics',
        description: 'Density mapping, bottleneck identification, and movement pattern recognition for public spaces.',
        icon: Users,
        color: 'text-purple-400',
        borderColor: 'border-purple-400/30',
        bgGlow: 'bg-purple-400/10',
        stats: '08 Active Nodes',
        status: 'Idle'
    },
    {
        id: 'safety',
        title: 'Public Safety',
        description: 'Anomaly detection, unattended object tracking, and automated emergency response triggers.',
        icon: ShieldAlert,
        color: 'text-emerald-400',
        borderColor: 'border-emerald-400/30',
        bgGlow: 'bg-emerald-400/10',
        stats: '22 Active Nodes',
        status: 'Operational'
    },
    {
        id: 'perimeter',
        title: 'Perimeter Security',
        description: 'Multi-spectral intrusion detection and automated target acquisition for high-security zones.',
        icon: Fence,
        color: 'text-rose-400',
        borderColor: 'border-rose-400/30',
        bgGlow: 'bg-rose-400/10',
        stats: '05 Active Nodes',
        status: 'Maintenance'
    }
];

const WelcomeScreen = () => {
    const navigate = useNavigate();
    return (
        <div className="fixed inset-0 bg-[#050a14] z-[100] flex flex-col overflow-hidden">
            {/* Background Effects */}
            <div className="absolute inset-0 z-0 opacity-20 pointer-events-none"
                style={{
                    backgroundImage: 'linear-gradient(rgba(0, 255, 255, 0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 255, 255, 0.2) 1px, transparent 1px)',
                    backgroundSize: '30px 30px'
                }}>
            </div>
            <div className="absolute inset-0 z-0 bg-[radial-gradient(circle_at_center,transparent_0%,#000_100%)] opacity-80" />

            {/* Top Bar */}
            <header className="relative z-10 w-full h-20 border-b border-white/5 bg-black/40 backdrop-blur-md flex items-center justify-between px-8">
                <div className="flex items-center gap-6" />
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                        <span className="text-[10px] text-emerald-500 font-bold tracking-widest uppercase">Nodes: Operational</span>
                    </div>
                    <div className="h-6 w-[1px] bg-white/10" />
                    <button className="p-2 text-white/40 hover:text-white transition-colors">
                        <Settings className="w-4 h-4" />
                    </button>
                    <button className="p-2 text-white/40 hover:text-rose-500 transition-colors">
                        <Power className="w-4 h-4" />
                    </button>
                </div>
            </header>

            {/* Main Content */}
            <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-8 max-w-7xl mx-auto w-full">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col md:flex-row items-center gap-12 mb-20"
                >
                    {/* Logo on the left */}
                    <div className="relative w-48 h-48 flex-shrink-0">
                        <img
                            src="https://upload.wikimedia.org/wikipedia/en/thumb/f/fc/Bangalore_City_Police_Logo.png/250px-Bangalore_City_Police_Logo.png"
                            alt="IRIS Logo"
                            className="w-full h-full object-contain filter drop-shadow-[0_0_30px_rgba(6,182,212,0.3)] animate-pulse"
                        />
                        <div className="absolute inset-0 bg-cyan-500/10 blur-3xl -z-10 rounded-full animate-pulse" />
                    </div>

                    <div className="text-center md:text-left">
                        <h2 className="text-8xl font-black text-white mb-3 tracking-tighter uppercase leading-[0.8]">
                            IRIS <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">COMMAND</span>
                        </h2>
                        <p className="text-cyan-400 font-mono tracking-[0.1em] uppercase text-2xl font-bold mb-6">
                            Integrated Realtime Intelligence System
                        </p>
                        <div className="w-32 h-1 bg-cyan-500 mb-8 opacity-40 mx-auto md:mx-0" />
                        <p className="text-white/40 max-w-2xl font-medium text-lg leading-relaxed">
                            Deploy high-altitude vision nodes and specialized AI modules to begin real-time tactical oversight.
                        </p>
                    </div>
                </motion.div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full">
                    {USE_CASES.map((useCase, idx) => (
                        <motion.button
                            key={useCase.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            onClick={() => navigate(`/${useCase.id}`)}
                            className={`group relative flex flex-col text-left p-6 bg-white/5 border ${useCase.borderColor} hover:border-white/30 transition-all duration-500 overflow-hidden text-white`}
                        >
                            {/* Card Hover Glow */}
                            <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 ${useCase.bgGlow}`} />

                            {/* Card Decoration */}
                            <div className="absolute top-0 right-0 w-12 h-12 overflow-hidden pointer-events-none">
                                <div className={`absolute top-0 right-0 w-[140%] h-[2px] ${useCase.color.replace('text-', 'bg-')} transform rotate-45 translate-x-1/2 -translate-y-1/2 opacity-20 group-hover:opacity-100 transition-opacity`} />
                            </div>

                            <div className="relative z-10">
                                <div className={`w-12 h-12 rounded-lg bg-black/40 border border-white/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500`}>
                                    <useCase.icon className={`w-6 h-6 ${useCase.color}`} />
                                </div>

                                <h3 className="text-xl font-black tracking-wider mb-2 group-hover:text-cyan-400 transition-colors uppercase">{useCase.title}</h3>
                                <p className="text-sm text-white/50 leading-relaxed mb-6 h-12 overflow-hidden line-clamp-2">
                                    {useCase.description}
                                </p>

                                <div className="pt-4 border-t border-white/5 flex items-center justify-between">
                                    <div className="flex flex-col">
                                        <span className="text-[10px] text-white/30 uppercase font-black tracking-widest">{useCase.stats}</span>
                                        <span className={`text-[10px] font-bold uppercase ${useCase.status === 'Operational' ? 'text-emerald-500' : 'text-white/20'}`}>{useCase.status}</span>
                                    </div>
                                    <ChevronRight className="w-5 h-5 text-white/20 group-hover:text-white group-hover:translate-x-1 transition-all" />
                                </div>
                            </div>

                            {/* Corner Accents */}
                            <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-white/20" />
                            <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-white/20" />
                        </motion.button>
                    ))}
                </div>

                {/* Search / Global Tools */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.8 }}
                    className="mt-16 flex items-center gap-4 px-6 py-3 bg-white/5 border border-white/10 rounded-full cursor-pointer hover:bg-white/10 transition-all"
                >
                    <Search className="w-4 h-4 text-white/40" />
                    <span className="text-sm text-white/40 font-bold uppercase tracking-widest">Discover additional surveillance modules...</span>
                </motion.div>
            </main>

            {/* Footer Status */}
            <footer className="relative z-10 w-full h-12 border-t border-white/5 bg-black/20 flex items-center justify-between px-8 text-[10px] font-mono text-white/20">
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
