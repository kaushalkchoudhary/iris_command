import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = import.meta.env.DEV ? '/api' : `http://${window.location.hostname}:9010`;

const LeftPanel = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [message, setMessage] = useState(null);

  // Job tracking
  const [jobs, setJobs] = useState([]);
  const pollRef = useRef(null);
  const activeCount = jobs.filter(j => j.status === 'processing').length;

  // Start/stop polling based on whether there are active jobs
  useEffect(() => {
    if (activeCount > 0 && !pollRef.current) {
      pollRef.current = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/jobs`);
          if (res.ok) {
            const data = await res.json();
            if (data.jobs) {
              setJobs(prev => {
                const jobMap = {};
                data.jobs.forEach(j => { jobMap[j.id] = j; });
                return prev.map(j => jobMap[j.id] || j);
              });
            }
          }
        } catch (e) {
          console.error('Failed to poll jobs:', e);
        }
      }, 2000);
    } else if (activeCount === 0 && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [activeCount]);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

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
      const url = `${API_BASE_URL}/upload?filename=${encodeURIComponent(file.name)}`;

      const xhr = new XMLHttpRequest();
      xhr.open('POST', url, true);
      xhr.setRequestHeader('Content-Type', 'application/octet-stream');

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          setUploadProgress((e.loaded / e.total) * 100);
        }
      };

      xhr.onload = function () {
        if (xhr.status === 200) {
          setUploadProgress(100);
          try {
            const result = JSON.parse(xhr.responseText);
            if (result.job_id) {
              setJobs(prev => [...prev, {
                id: result.job_id,
                filename: result.filename || file.name,
                status: 'processing',
                output_file: null,
                error: null,
              }]);
              setMessage({ type: 'success', text: 'Processing started' });
            }
          } catch {
            setMessage({ type: 'success', text: 'Upload complete' });
          }
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

  const handleDownload = (outputFile) => {
    const link = document.createElement('a');
    link.href = `${API_BASE_URL}/download?file=${encodeURIComponent(outputFile)}`;
    link.download = outputFile;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <>
      <div
        className={`fixed left-0 top-1/2 -translate-y-1/2 z-50 transition-all duration-300 ${isOpen ? 'translate-x-0' : '-translate-x-[calc(100%-20px)]'}`}
      >
        <div className="flex items-center">
          {/* Panel Content */}
          <div className="w-80 bg-black/80 backdrop-blur-xl border-r border-y border-emerald-500/30 rounded-r-xl p-6 shadow-2xl relative max-h-[80vh] overflow-y-auto">
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

              {/* Job list */}
              {jobs.length > 0 && (
                <div className="space-y-2 pt-2 border-t border-white/10">
                  <div className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Processing Jobs</div>
                  {jobs.map(job => (
                    <div key={job.id} className="bg-white/5 border border-white/10 rounded-lg p-3 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-white/80 font-mono truncate max-w-[180px]" title={job.filename}>
                          {job.filename}
                        </span>
                        {job.status === 'processing' && (
                          <span className="text-[10px] text-yellow-400 font-bold uppercase flex items-center gap-1">
                            <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <circle cx="12" cy="12" r="10" strokeDasharray="31.4 31.4" strokeDashoffset="10" />
                            </svg>
                            PROCESSING
                          </span>
                        )}
                        {job.status === 'done' && (
                          <span className="text-[10px] text-emerald-400 font-bold uppercase">DONE</span>
                        )}
                        {job.status === 'error' && (
                          <span className="text-[10px] text-red-400 font-bold uppercase">ERROR</span>
                        )}
                      </div>

                      {job.status === 'processing' && (
                        <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                          <div className="h-full bg-yellow-500/60 rounded-full animate-pulse" style={{ width: '100%' }} />
                        </div>
                      )}

                      {job.status === 'done' && job.output_file && (
                        <button
                          onClick={() => handleDownload(job.output_file)}
                          className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-emerald-500/20 border border-emerald-500/40 rounded-md text-emerald-400 text-xs font-bold uppercase tracking-wider hover:bg-emerald-500/30 transition-all"
                        >
                          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="7 10 12 15 17 10" />
                            <line x1="12" y1="15" x2="12" y2="3" />
                          </svg>
                          Download
                        </button>
                      )}

                      {job.status === 'error' && job.error && (
                        <div className="text-[10px] text-red-400/70">{job.error}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
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
