import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { API_BASE_URL } from './config'

const installFrontendLogger = () => {
  const sendLog = (level, args, context = {}) => {
    try {
      const message = args.map((a) => {
        if (a instanceof Error) return a.message;
        if (typeof a === 'string') return a;
        try { return JSON.stringify(a); } catch { return String(a); }
      }).join(' ');

      fetch(`${API_BASE_URL}/logs/frontend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          level,
          message,
          context,
          ts: new Date().toISOString(),
        }),
        keepalive: true,
      }).catch(() => {});
    } catch (e) {}
  };

  console.log = (...args) => sendLog('info', args);
  console.info = (...args) => sendLog('info', args);
  console.warn = (...args) => sendLog('warn', args);
  console.error = (...args) => sendLog('error', args);

  window.addEventListener('error', (event) => {
    sendLog('error', [event.message], {
      source: event.filename,
      line: event.lineno,
      column: event.colno,
    });
  });

  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason instanceof Error ? event.reason.message : String(event.reason);
    sendLog('error', ['UnhandledRejection', reason]);
  });
};

installFrontendLogger();

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
