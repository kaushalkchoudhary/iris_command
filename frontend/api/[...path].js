export default async function handler(req, res) {
  const path = req.url.replace(/^\/api/, '');
  const backendUrl = `${process.env.BACKEND_URL || 'http://139.84.145.157:9010'}/api${path}`;

  try {
    const headers = { ...req.headers };
    delete headers.host;
    delete headers['content-length'];

    const fetchOptions = {
      method: req.method,
      headers: { 'Content-Type': 'application/json' },
    };

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      fetchOptions.body = JSON.stringify(req.body);
    }

    const response = await fetch(backendUrl, fetchOptions);
    const data = await response.text();

    res.status(response.status);
    response.headers.forEach((value, key) => {
      if (key !== 'transfer-encoding' && key !== 'content-encoding') {
        res.setHeader(key, value);
      }
    });
    res.send(data);
  } catch (e) {
    res.status(502).json({ error: 'Backend unreachable', detail: e.message });
  }
}
