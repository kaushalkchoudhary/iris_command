const LOCAL_MODE = import.meta.env.VITE_LOCAL === '1';

const API_BASE_URL = LOCAL_MODE
  ? `http://${window.location.hostname}:9010/api`
  : 'https://iriscmdapi.stagingbot.xyz/api';

const WEBRTC_BASE = LOCAL_MODE
  ? `http://${window.location.hostname}:8889`
  : 'https://mediamtx1.stagingbot.xyz';

const HLS_BASE = LOCAL_MODE
  ? `http://${window.location.hostname}:8888`
  : 'https://hls.stagingbot.xyz';

export { API_BASE_URL, WEBRTC_BASE, HLS_BASE };

