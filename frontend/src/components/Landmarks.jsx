import { useState } from 'react';
import Panel from './Panel.jsx';

const LEGEND = [
  { label: 'Eyes', color: '#dc3545', keys: ['left_eye', 'right_eye'] },
  { label: 'Eyebrows', color: '#ff9500', keys: ['left_eyebrow', 'right_eyebrow'] },
  { label: 'Nose', color: '#19a05a', keys: ['nose'] },
  { label: 'Lips', color: '#2563eb', keys: ['mouth'] },
];

export default function Landmarks({ file, result, busy, error, onDetect }) {
  const [zoomed, setZoomed] = useState(false);
  const counts = result?.region_counts ?? {};

  return (
    <section id="landmarks" className="mt-5 pt-4 fade-section">
      <h2 className="fw-bold">
        <i className="fas fa-map-pin text-primary me-2" />
        Facial Landmarks
      </h2>

      <div className="row g-4 mt-3">
        <div className="col-md-6">
          <div className="card shadow-sm border-0 rounded-4 p-3 text-center h-100">
            <Panel title="Original Image" src={result?.images?.original} caption="256 × 256" zoomed={zoomed} />
          </div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm border-0 rounded-4 p-3 text-center h-100">
            <Panel
              title="Landmarks Detected"
              src={result?.images?.landmarks}
              caption={
                result
                  ? result.face_detected
                    ? `${result.landmark_count} MediaPipe Face Mesh points`
                    : 'no face detected'
                  : 'upload an image to detect'
              }
              zoomed={zoomed}
            />

            <div className="d-flex justify-content-center gap-3 mt-2 flex-wrap small">
              {LEGEND.map((item) => {
                const total = item.keys.reduce((sum, key) => sum + (counts[key] ?? 0), 0);
                return (
                  <span key={item.label}>
                    <span style={{ color: item.color }}>●</span> {item.label}
                    {total ? ` (${total})` : ''}
                  </span>
                );
              })}
            </div>

            <div className="d-flex justify-content-center gap-2 mt-3 flex-wrap">
              <button
                type="button"
                className="btn btn-outline-secondary btn-sm rounded-pill"
                onClick={() => setZoomed((v) => !v)}
                disabled={!result}
              >
                <i className={`fas ${zoomed ? 'fa-search-minus' : 'fa-search-plus'} me-1`} />
                {zoomed ? 'Reset zoom' : 'Zoom'}
              </button>
              <button
                type="button"
                className="btn btn-outline-primary btn-sm rounded-pill"
                onClick={onDetect}
                disabled={!file || busy}
              >
                {busy ? (
                  <>
                    <i className="fas fa-spinner fa-spin me-1" />
                    Detecting…
                  </>
                ) : (
                  <>
                    <i className="fas fa-crosshairs me-1" />
                    Detect landmarks
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {error ? (
        <div className="alert alert-danger py-2 px-3 mt-3 mb-0">
          <i className="fas fa-exclamation-triangle me-1" />
          {error}
        </div>
      ) : null}

      {result && !result.face_detected ? (
        <div className="alert alert-warning py-2 px-3 mt-3 mb-0 small">
          <i className="fas fa-triangle-exclamation me-1" />
          MediaPipe could not find a face in this image. Semantic masks need a detected face; irregular-blob variants
          still work with a fallback position.
        </div>
      ) : null}
    </section>
  );
}
