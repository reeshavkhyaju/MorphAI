import { useState } from 'react';

export default function Compare({ prediction, notify }) {
  const [split, setSplit] = useState(50);
  const [sliderMode, setSliderMode] = useState(true);

  const original = prediction?.images?.original;
  const predicted = prediction?.images?.reconstructed;
  const ready = Boolean(original && predicted);

  const download = () => {
    if (!predicted) return;
    const link = document.createElement('a');
    link.href = predicted;
    link.download = `morphai_${prediction.variant}_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    notify('Reconstruction downloaded.', 'success');
  };

  return (
    <section id="compare" className="mt-5 pt-4 fade-section">
      <h2 className="fw-bold">
        <i className="fas fa-arrows-left-right text-primary me-2" />
        Image Comparison
      </h2>

      <div className="card shadow-sm border-0 rounded-4 p-4 mt-3">
        {!ready ? (
          <p className="text-muted text-center mb-0 py-4">
            <i className="fas fa-circle-info me-1" />
            Run a prediction to compare the original and reconstructed faces.
          </p>
        ) : sliderMode ? (
          <>
            <div className="compare-viewport" style={{ '--split': `${split}%` }}>
              <img src={original} alt="Original" />
              <img className="compare-top" src={predicted} alt="Predicted" />
              <div className="compare-handle" style={{ left: `${split}%` }} />
              <span className="compare-tag" style={{ left: '0.5rem' }}>
                Original
              </span>
              <span className="compare-tag" style={{ right: '0.5rem' }}>
                Predicted
              </span>
              <input
                className="compare-range"
                type="range"
                min="0"
                max="100"
                value={split}
                onChange={(e) => setSplit(Number(e.target.value))}
                aria-label="Comparison split"
              />
            </div>
            <p className="text-center text-muted small mt-2 mb-0">Drag across the image to wipe between the two.</p>
          </>
        ) : (
          <div className="row align-items-center">
            <div className="col-md-5 text-center">
              <img src={original} className="img-fluid rounded-3 shadow" alt="Original" />
              <div className="small text-muted mt-1">Original</div>
            </div>
            <div className="col-md-2 text-center">
              <i className="fas fa-arrow-right fa-2x text-secondary" />
            </div>
            <div className="col-md-5 text-center">
              <img src={predicted} className="img-fluid rounded-3 shadow" alt="Predicted" />
              <div className="small text-muted mt-1">Predicted</div>
            </div>
          </div>
        )}

        <div className="d-flex justify-content-center gap-3 mt-3">
          <button
            type="button"
            className="btn btn-outline-primary btn-sm rounded-pill"
            onClick={() => setSliderMode((v) => !v)}
            disabled={!ready}
          >
            <i className="fas fa-sliders-h me-1" />
            {sliderMode ? 'Side by side' : 'Slider'}
          </button>
          <button
            type="button"
            className="btn btn-outline-primary btn-sm rounded-pill"
            onClick={download}
            disabled={!ready}
          >
            <i className="fas fa-download me-1" />
            Download
          </button>
        </div>
      </div>
    </section>
  );
}
