import React from 'react';
import clsx from 'clsx';

const Card = ({ children, className, title }) => {
    return (
        <div className={clsx(
            "border border-cyan-500/20 p-5 rounded-sm relative overflow-hidden",
            className
        )}>
            {/* Corner Accents */}
            <div className="absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2 border-cyan-400"></div>
            <div className="absolute top-0 right-0 w-2 h-2 border-t-2 border-r-2 border-cyan-400"></div>
            <div className="absolute bottom-0 left-0 w-2 h-2 border-b-2 border-l-2 border-cyan-400"></div>
            <div className="absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2 border-cyan-400"></div>

            {title && (
                <div className="mb-4 pb-2 border-b border-cyan-500/20 flex items-center justify-between">
                    <h3 className="text-cyan-400 font-bold uppercase tracking-wider text-sm flex items-center gap-2">
                        <span className="w-1 h-4 bg-cyan-400 block shrink-0"></span>
                        {title}
                    </h3>
                </div>
            )}
            {children}
        </div>
    );
};

export default Card;
