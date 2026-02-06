import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { API_BASE_URL } from './config'

const TAB_ID_KEY = 'iris_tab_id';
const AUTH_TOKEN_KEY = 'iris_auth_token';
const AUTH_FLAG_KEY = 'iris_authenticated';

const getOrCreateTabId = () => {
  let tabId = sessionStorage.getItem(TAB_ID_KEY);
  if (!tabId) {
    tabId = (crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`);
    sessionStorage.setItem(TAB_ID_KEY, tabId);
  }
  return tabId;
};

const installAuthFetchInterceptor = () => {
  const rawFetch = window.fetch.bind(window);

  window.fetch = async (input, init = {}) => {
    const reqUrl = typeof input === 'string' ? input : (input?.url || '');
    const isApiRequest = reqUrl.includes('/api/');
    const isLogin = reqUrl.endsWith('/api/login');
    const isHealth = reqUrl.endsWith('/api/health');

    const headers = new Headers(init.headers || {});
    if (isApiRequest) {
      headers.set('X-Client-Tab', getOrCreateTabId());
    }

    const token = sessionStorage.getItem(AUTH_TOKEN_KEY) || localStorage.getItem(AUTH_TOKEN_KEY);
    if (isApiRequest && token && !isLogin) {
      headers.set('Authorization', `Bearer ${token}`);
    }

    const response = await rawFetch(input, { ...init, headers });

    if (isApiRequest && response.status === 401 && !isLogin && !isHealth) {
      sessionStorage.removeItem(AUTH_TOKEN_KEY);
      sessionStorage.removeItem(AUTH_FLAG_KEY);
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_FLAG_KEY);
      localStorage.removeItem('iris_username');
      if (window.location.pathname !== '/') {
        window.location.href = '/';
      }
    }

    return response;
  };
};

installAuthFetchInterceptor();

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
