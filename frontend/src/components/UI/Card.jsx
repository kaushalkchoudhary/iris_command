import React from 'react';
import clsx from 'clsx';

const Card = ({ children, className, title }) => {
  return (
    <div
      className={clsx(
        `
        relative overflow-hidden
        border border-emerald-500/15
        bg-black/20
        px-4 py-3
        font-mono
        `,
        className
      )}
    >
      {/* Corner Brackets — minimal, HUD-style */}
      <span className="absolute top-0 left-0 w-2 h-2 border-t border-l border-emerald-400/50" />
      <span className="absolute top-0 right-0 w-2 h-2 border-t border-r border-emerald-400/50" />
      <span className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-emerald-400/50" />
      <span className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-emerald-400/50" />

      {/* Title */}
      {title && (
        <div className="mb-2 pb-1.5 border-b border-emerald-500/10 flex items-center gap-2">
          {/* Signal bar */}
          <span className="w-[1.5px] h-3 bg-emerald-400/70 shrink-0" />

          <h3 className="text-[10px] text-emerald-400 font-bold uppercase tracking-[0.3em] leading-none">
            {title}
          </h3>
        </div>
      )}

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};

export default Card;
