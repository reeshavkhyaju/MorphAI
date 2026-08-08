import { useRef, useState } from 'react';
import Panel from './Panel.jsx';

export default function Predict({
  file,
  previewUrl,
  onFile,
  onReset,
  onPredict,
  busy,
  prediction,
  error,
  options,
  setOptions,
  variantList,
  blendList,
  identityAvailable,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (fileList) => {
    if (fileList && fileList.length) onFile(fileList[0]);
  };

  const semantic = variantList.filter((v) => v.id.startsWith('semantic_'));
  const irregular = variantList.filter((v) => !v.id.startsWith('semantic_'));

  let badge = { text: 'Waiting for input', className: 'bg-secondary' };
  if (busy) badge = { text: 'Processing…', className: 'bg-info' };
  else if (error) badge = { text: 'Error', className: 'bg-danger' };
  else if (prediction) badge = { text: 'Prediction Complete', className: 'bg-success' };
  else if (file) badge = { text: 'Image loaded', className: 'bg-warning text-dark' };

  const metrics = prediction?.metrics;
  const identityPct =
    metrics && metrics.identity !== null && metrics.identity !== undefined
      ? (metrics.identity * 100).toFixed(2)
      : '--';

  return (
    <section id="predict" className="mt-5 pt-4 fade-section">
      <h2 className="fw-bold">
        <i className="fas fa-microchip text-primary me-2" />
        Prediction
      </h2>

      <div className="row g-4 mt-3">
        {/* ---------- Upload + controls ---------- */}
        <div className="col-lg-6">
          <div className="card shadow-sm border-0 rounded-4 p-4 h-100">
            <h5>
              <i className="fas fa-cloud-upload-alt me-2 text-primary" />
              Upload Image
            </h5>

            <div
              className={`drop-zone p-4 text-center border rounded-4 bg-light${dragging ? ' dragging' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setDragging(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                handleFiles(e.dataTransfer.files);
              }}
            >
              <i className="fas fa-image fa-3x text-secondary opacity-50" />
              <p className="mt-2 mb-1">
                Drag &amp; drop or <strong>click</strong> to browse
              </p>
              <p className="text-muted small">JPG, PNG, JPEG · resized to 256×256</p>
              <input
                ref={inputRef}
                type="file"
                className="d-none"
                accept="image/*"
                onChange={(e) => handleFiles(e.target.files)}
              />
              <button
                type="button"
                className="btn btn-outline-primary btn-sm rounded-pill"
                onClick={(e) => {
                  e.stopPropagation();
                  inputRef.current?.click();
                }}
              >
                <i className="fas fa-upload me-1" />
                Choose file
              </button>
            </div>

            {previewUrl ? (
              <div className="mt-3 text-center">
                <img
                  src={previewUrl}
                  alt="preview"
                  className="img-fluid rounded-3 shadow-sm"
                  style={{ maxHeight: 200 }}
                />
                <div className="small text-muted mt-1 text-truncate">{file?.name}</div>
              </div>
            ) : null}

            {/* Controls */}
            <div className="row g-3 mt-1">
              <div className="col-sm-6">
                <label className="form-label small fw-semibold" htmlFor="variant-select">
                  Mask variant
                </label>
                <select
                  id="variant-select"
                  className="form-select form-select-sm rounded-3"
                  value={options.variant}
                  onChange={(e) => setOptions({ ...options, variant: e.target.value })}
                >
                  <optgroup label="Semantic region">
                    {semantic.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.label}
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="Irregular blob">
                    {irregular.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.label}
                      </option>
                    ))}
                  </optgroup>
                </select>
              </div>

              <div className="col-sm-6">
                <label className="form-label small fw-semibold" htmlFor="blend-select">
                  Blending
                </label>
                <select
                  id="blend-select"
                  className="form-select form-select-sm rounded-3"
                  value={options.blend}
                  onChange={(e) => setOptions({ ...options, blend: e.target.value })}
                >
                  {blendList.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="col-12">
                <div className="form-check">
                  <input
                    className="form-check-input"
                    type="checkbox"
                    id="arcface-check"
                    checked={options.computeIdentity}
                    disabled={!identityAvailable}
                    onChange={(e) => setOptions({ ...options, computeIdentity: e.target.checked })}
                  />
                  <label className="form-check-label small" htmlFor="arcface-check">
                    Calculate ArcFace identity score
                    {identityAvailable ? '' : ' (unavailable on this machine)'}
                  </label>
                </div>
              </div>
            </div>

            <div className="d-flex gap-2 mt-3">
              <button
                type="button"
                className="btn btn-primary rounded-pill px-4"
                onClick={onPredict}
                disabled={busy || !file}
              >
                {busy ? (
                  <>
                    <i className="fas fa-spinner fa-spin me-1" />
                    Processing…
                  </>
                ) : (
                  <>
                    <i className="fas fa-wand-magic-sparkles me-1" />
                    Predict
                  </>
                )}
              </button>
              <button type="button" className="btn btn-outline-secondary rounded-pill px-4" onClick={onReset}>
                <i className="fas fa-undo me-1" />
                Reset
              </button>
            </div>

            <div className="mt-3">
              {busy ? (
                <div className="d-flex align-items-center gap-2">
                  <div className="spinner-border spinner-border-sm text-primary" role="status" />
                  <span className="small">Running LGNet inference on the backend…</span>
                </div>
              ) : null}
              {!busy && error ? (
                <div className="alert alert-danger py-2 px-3 mb-0">
                  <i className="fas fa-exclamation-triangle me-1" />
                  {error}
                </div>
              ) : null}
              {!busy && !error && prediction ? (
                <div className="alert alert-success py-2 px-3 mb-0 rounded-4 small">
                  <i className="fas fa-check-circle me-1" />
                  Done in {prediction.processing_time}s · SSIM {metrics.ssim} · PSNR {metrics.psnr} dB · mask{' '}
                  {metrics.mask_coverage}%
                </div>
              ) : null}
            </div>
          </div>
        </div>

        {/* ---------- Result summary ---------- */}
        <div className="col-lg-6">
          <div className="card shadow-sm border-0 rounded-4 p-4 h-100">
            <h5>
              <i className="fas fa-chart-simple me-2 text-primary" />
              Prediction Result
            </h5>

            <div className="row align-items-center text-center g-2">
              <div className="col-5">
                <Panel title="Original" src={prediction?.images?.original ?? previewUrl} />
              </div>
              <div className="col-2">
                <i className="fas fa-arrow-right fa-2x text-secondary" />
              </div>
              <div className="col-5">
                <Panel title="Generated" src={prediction?.images?.reconstructed} />
              </div>
            </div>

            <div className="mt-3 text-center">
              <span className={`badge ${badge.className}`}>{badge.text}</span>
            </div>

            <div className="row text-center mt-3 g-2 small">
              <div className="col-6">
                <div className="fw-semibold">Identity (ArcFace)</div>
                <div className="fs-5">{identityPct}%</div>
              </div>
              <div className="col-6">
                <div className="fw-semibold">Time</div>
                <div className="fs-5">{prediction ? prediction.processing_time : '--'}s</div>
              </div>
              <div className="col-6">
                <div className="fw-semibold">SSIM</div>
                <div className="fs-6">{metrics ? metrics.ssim : '--'}</div>
              </div>
              <div className="col-6">
                <div className="fw-semibold">PSNR</div>
                <div className="fs-6">{metrics ? `${metrics.psnr} dB` : '--'}</div>
              </div>
              <div className="col-6">
                <div className="fw-semibold">PSNR (masked area)</div>
                <div className="fs-6">{metrics?.psnr_masked ? `${metrics.psnr_masked} dB` : '--'}</div>
              </div>
              <div className="col-6">
                <div className="fw-semibold">Mask coverage</div>
                <div className="fs-6">{metrics ? `${metrics.mask_coverage}%` : '--'}</div>
              </div>
            </div>

            {prediction ? (
              <div className="text-center mt-3 small text-muted">
                {prediction.variant_label} · {prediction.blend} blend ·{' '}
                {prediction.face_detected
                  ? `${prediction.landmark_count} landmarks detected`
                  : 'no face detected (fallback mask)'}{' '}
                · {prediction.device}
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {/* ---------- Pipeline panels ---------- */}
      <h5 className="mt-4 mb-3">
        <i className="fas fa-layer-group text-primary me-2" />
        Inference Pipeline
      </h5>
      <div className="card shadow-sm border-0 rounded-4 p-4">
        <div className="row g-4">
          <div className="col-md-4">
            <Panel title="1. Original Input" src={prediction?.images?.original} caption="256 × 256" />
          </div>
          <div className="col-md-4">
            <Panel title="2. Masked Input" src={prediction?.images?.masked} caption="5-channel tensor input" />
          </div>
          <div className="col-md-4">
            <Panel
              title="3. Reconstructed Output"
              src={prediction?.images?.reconstructed}
              caption={prediction ? `${prediction.blend} blend` : 'blended result'}
            />
          </div>
        </div>
        <div className="row g-4 mt-1">
          <div className="col-md-4">
            <Panel title="Raw Generator Output" src={prediction?.images?.generated} caption="before blending" />
          </div>
          <div className="col-md-4">
            <Panel title="Landmark Heatmap" src={prediction?.images?.heatmap} caption="conditioning channel" />
          </div>
          <div className="col-md-4">
            <Panel title="Binary Mask" src={prediction?.images?.mask} caption="255 = inpainted region" />
          </div>
        </div>
      </div>
    </section>
  );
}
