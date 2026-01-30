import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Shield } from 'lucide-react';

const API_BASE_URL = import.meta.env.DEV
  ? '/api'
  : `http://${window.location.hostname}:9010`;

const Login = ({ onLogin }) => {
  const [username, setUsername] = useState('admin');
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
        onLogin();
      } else {
        setError(data.error || 'Login failed');
      }
    } catch {
      setError('Connection error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-[#050a14] relative overflow-hidden font-mono">
      {/* Background Grid */}
      <div
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{
          backgroundImage:
            'linear-gradient(rgba(0,255,255,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,255,0.2) 1px, transparent 1px)',
          backgroundSize: '30px 30px',
        }}
      />

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md relative z-10 px-4"
      >
        <div className="relative overflow-hidden p-8 bg-black/40 backdrop-blur-xl">
          
          {/* IRIS legacy frame */}
          <div className="pointer-events-none absolute inset-0 z-0">
            {/* thin faded border */}
            <div
              className="absolute inset-0"
              style={{
                border: '1px solid rgba(34,211,238,0.35)',
                maskImage: `
                  linear-gradient(to right,
                    transparent 0%,
                    black 12%,
                    black 88%,
                    transparent 100%),
                  linear-gradient(to bottom,
                    transparent 0%,
                    black 12%,
                    black 88%,
                    transparent 100%)
                `,
                WebkitMaskComposite: 'source-in',
                maskComposite: 'intersect',
              }}
            />
            
            {/* soft inner glow */}
            <div
              className="absolute inset-0"
              style={{
                boxShadow: `
                  inset 0 0 20px rgba(34,211,238,0.08),
                  inset 0 0 40px rgba(34,211,238,0.04)
                `,
              }}
            />

            {/* Corner brackets */}
            {/* Top Left */}
            <div className="absolute top-0 left-0 w-10 h-10 border-t border-l border-cyan-400/50" />
            <div className="absolute top-1 left-1 w-6 h-6 border-t border-l border-gray-400/40" />

            {/* Top Right */}
            <div className="absolute top-0 right-0 w-10 h-10 border-t border-r border-cyan-400/50" />
            <div className="absolute top-1 right-1 w-6 h-6 border-t border-r border-gray-400/40" />

            {/* Bottom Left */}
            <div className="absolute bottom-0 left-0 w-10 h-10 border-b border-l border-cyan-400/50" />
            <div className="absolute bottom-1 left-1 w-6 h-6 border-b border-l border-gray-400/40" />

            {/* Bottom Right */}
            <div className="absolute bottom-0 right-0 w-10 h-10 border-b border-r border-cyan-400/50" />
            <div className="absolute bottom-1 right-1 w-6 h-6 border-b border-r border-gray-400/40" />
          </div>

          {/* Content */}
          <div className="relative z-10">
            {/* Header */}
            <div className="text-center mb-8">
              <Shield className="w-12 h-12 text-cyan-400 mx-auto mb-4" />
              <h1 className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-[0.2em] mb-2">
                IRIS COMMAND
              </h1>
              <p className="text-cyan-500/70 text-sm uppercase tracking-[0.2em] font-bold">
                Secure Access
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-xs font-bold text-cyan-500/60 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <span className="w-1 h-4 bg-cyan-400/80 block shrink-0" />
                  Identifier
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full bg-black/50 border border-cyan-500/20 px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all"
                  placeholder="ENTER ID"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-cyan-500/60 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <span className="w-1 h-4 bg-cyan-400/80 block shrink-0" />
                  Passkey
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-black/50 border border-cyan-500/20 px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all"
                  placeholder="••••••••"
                />
              </div>

              {error && (
                <div className="text-red-400 text-xs font-bold text-center bg-red-500/10 py-2 border border-red-500/20">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-cyan-600 hover:bg-cyan-500 text-black font-black py-3 uppercase tracking-wider transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Authenticating…' : 'Initialize Session'}
              </button>
            </form>

            {/* Footer */}
            <div className="mt-8 pt-6 border-t border-cyan-500/20 text-center">
              <p className="text-[10px] text-cyan-500/40">
                SYSTEM V.4.0.2 // RESTRICTED ACCESS
              </p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Login;