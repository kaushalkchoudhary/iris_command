import React from 'react';
import { Shield, Clock, Cloud, Signal } from 'lucide-react';

const Header = ({ useCase, onReset }) => {
    const useCaseLabels = {
        traffic: 'Traffic Intelligence',
        crowd: 'Crowd Analytics',
        safety: 'Public Safety',
        perimeter: 'Perimeter Security'
    };

    return (
        <div className="absolute top-0 w-full z-50 p-6 flex justify-between items-start pointer-events-none select-none">
            {/* Brand */}
            <div className="flex items-center gap-4 pointer-events-auto">
                <button
                    onClick={onReset}
                    className="group flex items-center gap-4 text-left hover:opacity-80 transition-opacity"
                >
                    <Shield className="w-10 h-10 text-cyan-400" strokeWidth={1.5} />
                    <div>
                        <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-[0.2em] uppercase">
                            IRIS COMMAND
                        </h1>
                        <div className="flex items-center gap-2 mt-1">
                            <span className="px-2 py-0.5 bg-cyan-500 text-black text-[10px] font-black uppercase tracking-widest rounded leading-none">
                                {useCaseLabels[useCase] || 'MISSION_ACTIVE'}
                            </span>
                            <span className="text-[10px] text-cyan-300/50 uppercase tracking-[0.2em] ml-1">
                                [ Click to Switch Mission ]
                            </span>
                        </div>
                    </div>
                </button>
            </div>

            {/* Decorative / Stats */}
            <div className="flex items-center gap-8 pointer-events-auto text-cyan-400 font-mono text-sm">
                <div className="flex flex-col items-end">
                    <div className="flex items-center gap-2 text-white">
                        <Clock className="w-4 h-4 text-cyan-500" />
                        <span>15:20:59</span>
                    </div>
                    <span className="text-[10px] text-cyan-500/60 uppercase">System Time</span>
                </div>
                <div className="h-8 w-[1px] bg-cyan-500/20"></div>
                <div className="flex flex-col items-end">
                    <div className="flex items-center gap-2 text-white">
                        <Cloud className="w-4 h-4 text-cyan-500" />
                        <span>32°C</span>
                    </div>
                    <span className="text-[10px] text-cyan-500/60 uppercase">Temperature</span>
                </div>
            </div>
        </div>
    );
};

export default Header;
