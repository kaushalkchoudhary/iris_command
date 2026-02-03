import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Shield, Lock, User, ChevronRight } from 'lucide-react';

const API_BASE_URL = '/api';

const Login = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        localStorage.setItem('iris_username', data.username || username);
        onLogin();
      } else {
        setError(data.error || 'Invalid credentials');
      }
    } catch {
      setError('Connection error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-[#050a14] z-[100] flex flex-col overflow-hidden font-mono">
      {/* Background Effects - matching WelcomeScreen */}
      <div
        className="absolute inset-0 z-0 opacity-20 pointer-events-none"
        style={{
          backgroundImage:
            'linear-gradient(rgba(0, 255, 255, 0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 255, 255, 0.2) 1px, transparent 1px)',
          backgroundSize: '30px 30px',
        }}
      />
      <div className="absolute inset-0 z-0 bg-[radial-gradient(circle_at_center,transparent_0%,#000_100%)] opacity-80" />

      {/* Top Bar */}
      <header className="relative z-10 w-full h-20 border-b border-white/5 bg-black/40 backdrop-blur-md flex items-center justify-between px-8">
        <div className="flex items-center gap-4">
          <Shield className="w-8 h-8 text-cyan-400" />
          <span className="text-xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-[0.2em] uppercase">
            IRIS COMMAND
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
          <span className="text-[10px] text-emerald-500 font-bold tracking-widest uppercase">
            System: Online
          </span>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex items-center justify-center p-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-md"
        >
          {/* Login Card */}
          <div className="group relative bg-white/5 border border-cyan-400/30 hover:border-cyan-400/50 transition-all duration-500 overflow-hidden">
            {/* Card Hover Glow */}
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 bg-cyan-400/5" />

            {/* Card Decoration - top right corner slash */}
            <div className="absolute top-0 right-0 w-12 h-12 overflow-hidden pointer-events-none">
              <div className="absolute top-0 right-0 w-[140%] h-[2px] bg-cyan-400 transform rotate-45 translate-x-1/2 -translate-y-1/2 opacity-20 group-hover:opacity-100 transition-opacity" />
            </div>

            {/* Corner Accents */}
            <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-white/20" />
            <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-white/20" />
            <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-white/20" />
            <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-white/20" />

            {/* Content */}
            <div className="relative z-10 p-8">
              {/* Header */}
              <div className="text-center mb-8">
                <div className="w-16 h-16 rounded-lg bg-black/40 border border-white/10 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-500">
                  <Lock className="w-8 h-8 text-cyan-400" />
                </div>
                <h2 className="text-2xl font-black text-white tracking-wider uppercase mb-2">
                  Secure Access
                </h2>
                <p className="text-white/40 text-sm uppercase tracking-widest">
                  Authentication Required
                </p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label className="block text-[10px] font-bold text-cyan-500/60 uppercase tracking-widest mb-2 flex items-center gap-2">
                    <User className="w-3 h-3" />
                    Identifier
                  </label>
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="w-full bg-black/50 border border-white/10 hover:border-cyan-500/30 focus:border-cyan-500/50 px-4 py-3 text-white focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition-all font-mono tracking-wider"
                    placeholder="ENTER ID"
                  />
                </div>

                <div>
                  <label className="block text-[10px] font-bold text-cyan-500/60 uppercase tracking-widest mb-2 flex items-center gap-2">
                    <Lock className="w-3 h-3" />
                    Passkey
                  </label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full bg-black/50 border border-white/10 hover:border-cyan-500/30 focus:border-cyan-500/50 px-4 py-3 text-white focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition-all font-mono tracking-wider"
                    placeholder="••••••••"
                  />
                </div>

                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-red-400 text-[10px] font-bold text-center bg-red-500/10 py-2 border border-red-500/20 uppercase tracking-widest"
                  >
                    {error}
                  </motion.div>
                )}

                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full bg-cyan-600 hover:bg-cyan-500 text-black font-black py-3 uppercase tracking-widest transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 group/btn"
                >
                  {isLoading ? (
                    <span>Authenticating...</span>
                  ) : (
                    <>
                      <span>Initialize Session</span>
                      <ChevronRight className="w-4 h-4 group-hover/btn:translate-x-1 transition-transform" />
                    </>
                  )}
                </button>
              </form>

              {/* Footer */}
              <div className="mt-8 pt-6 border-t border-white/5 flex items-center justify-between">
                <span className="text-[10px] text-white/20 uppercase tracking-widest">
                  Restricted Access
                </span>
                <span className="text-[10px] text-white/20 uppercase tracking-widest">
                  V.4.0.2
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      </main>

      {/* Footer Status */}
      <footer className="relative z-10 w-full h-12 border-t border-white/5 bg-black/20 flex items-center justify-between px-8 text-[10px] font-mono text-white/20">
        <div className="flex gap-8">
          <div className="flex gap-2">
            <span className="text-white/40">GEO_LOC:</span> BANGALORE_HUB
          </div>
          <div className="flex gap-2">
            <span className="text-white/40">STATUS:</span> AWAITING_AUTH
          </div>
        </div>
        <div>IRIS COMMAND // SECURE_PORTAL</div>
      </footer>
    </div>
  );
};

export default Login;
