// Thin client for the Flask backend.
// In dev, Vite proxies /api and /predict to http://127.0.0.1:5000 (vite.config.js).
// In production the same Flask process serves this bundle, so relative URLs work.

const BASE = import.meta.env.VITE_API_BASE ?? '';

async function toJson(response) {
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`Backend returned a non-JSON response (HTTP ${response.status}).`);
  }
  if (!response.ok || data.success === false) {
    throw new Error(data.error || `Request failed (HTTP ${response.status}).`);
  }
  return data;
}

export function getHealth() {
  return fetch(`${BASE}/api/health`).then(toJson);
}

export function getVariants() {
  return fetch(`${BASE}/api/variants`).then(toJson);
}

export function getEvaluation() {
  return fetch(`${BASE}/api/evaluation`).then(toJson);
}

export function postPredict(file, { variant, blend, computeIdentity }) {
  const body = new FormData();
  body.append('image', file);
  body.append('variant', variant);
  body.append('blend', blend);
  body.append('compute_identity', computeIdentity ? '1' : '0');
  return fetch(`${BASE}/api/predict`, { method: 'POST', body }).then(toJson);
}

export function postLandmarks(file) {
  const body = new FormData();
  body.append('image', file);
  return fetch(`${BASE}/api/landmarks`, { method: 'POST', body }).then(toJson);
}
