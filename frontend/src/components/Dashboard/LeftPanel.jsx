import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = import.meta.env.DEV ? '/api' : `http://${window.location.hostname}:9010`;

const LeftPanel = ({ onUploadSuccess }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [message, setMessage] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate extension
    const validExtensions = ['video/mp4', 'video/x-matroska', 'video/webm']; // rough check
    const filename = file.name.toLowerCase();
    if (!filename.endsWith('.mp4') && !filename.endsWith('.mkv')) {
      setMessage({ type: 'error', text: 'Only MP4 or MKV files allowed' });
      return;
    }

    await uploadFile(file);
  };

  const uploadFile = async (file) => {
    setIsUploading(true);
    setUploadProgress(0);
    setMessage(null);

    try {
      // Use raw body upload with filename in query param
      const url = `${API_BASE_URL}/upload?filename=${encodeURIComponent(file.name)}`;

      const xhr = new XMLHttpRequest();
      xhr.open('POST', url, true);
      xhr.setRequestHeader('Content-Type', 'application/octet-stream');

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const percentComplete = (e.loaded / e.total) * 100;
          setUploadProgress(percentComplete);
        }
      };

      xhr.onload = function () {
        if (xhr.status === 200) {
          setMessage({ type: 'success', text: 'Upload successful' });
          setUploadProgress(100);
          if (onUploadSuccess) onUploadSuccess();
          setTimeout(() => setMessage(null), 3000);
        } else {
          setMessage({ type: 'error', text: `Upload failed: ${xhr.responseText}` });
        }
        setIsUploading(false);
      };

      xhr.onerror = function () {
        setMessage({ type: 'error', text: 'Connection error during upload' });
        setIsUploading(false);
      };

      xhr.send(file);

    } catch (e) {
      console.error(e);
      setMessage({ type: 'error', text: 'Upload error' });
      setIsUploading(false);
    }
  };

  return (
    <>
      <div
        className={`fixed left-0 top-1/2 -translate-y-1/2 z-50 transition-all duration-300 ${isOpen ? 'translate-x-0' : '-translate-x-[calc(100%-20px)]'}`}
      >
        <div className="flex items-center">
          {/* Panel Content */}
          <div className="w-80 bg-black/80 backdrop-blur-xl border-r border-y border-emerald-500/30 rounded-r-xl p-6 shadow-2xl relative">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-emerald-400 font-black tracking-wider text-sm flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                UPLOAD FEED
              </h2>
              <button onClick={() => setIsOpen(false)} className="text-white/50 hover:text-white">✕</button>
            </div>

            <div className="space-y-4">
              <div className="border-2 border-dashed border-white/20 rounded-lg p-8 text-center hover:border-emerald-500/50 transition-colors group relative cursor-pointer">
                <input
                  type="file"
                  accept=".mp4,.mkv"
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  disabled={isUploading}
                />
                <div className="text-4xl mb-2 group-hover:scale-110 transition-transform">📤</div>
                <p className="text-xs text-emerald-500 font-bold uppercase tracking-wider mb-1">Upload Video</p>
                <p className="text-[10px] text-gray-500">MP4 / MKV Format Only</p>
              </div>

              <AnimatePresence>
                {isUploading && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-2 overflow-hidden"
                  >
                    <div className="flex justify-between text-[10px] uppercase font-bold text-gray-400">
                      <span>Uploading...</span>
                      <span>{Math.round(uploadProgress)}%</span>
                    </div>
                    <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </motion.div>
                )}

                {message && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    className={`text-[10px] font-bold p-2 rounded border ${message.type === 'error'
                        ? 'bg-red-500/10 border-red-500/20 text-red-400'
                        : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                      }`}
                  >
                    {message.text}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Toggle Tab */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="bg-emerald-600 text-black font-black text-[10px] uppercase tracking-widest py-8 px-1 rounded-r-md hover:bg-emerald-500 transition-colors shadow-[0_0_15px_rgba(16,185,129,0.5)] [writing-mode:vertical-rl] flex items-center gap-2"
          >
            {isOpen ? 'CLOSE' : 'UPLOAD'}
          </button>
        </div>
      </div>
    </>
  );
};

export default LeftPanel;
