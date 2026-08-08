// Renders the recorded dataset-level evaluation
// (MorphAI/samples/final_eval_results.json) served by GET /api/evaluation.

const METRIC_CONFIG = {
  psnr_full: {
    label: 'PSNR',
    hint: 'full image',
    unit: ' dB',
    better: 'higher',
    color: 'primary',
    bar: (v) => Math.min(v / 45, 1) * 100,
    digits: 2,
  },
  ssim_full: {
    label: 'SSIM',
    hint: 'full image',
    unit: '',
    better: 'higher',
    color: 'success',
    bar: (v) => v * 100,
    digits: 4,
  },
  lpips: {
    label: 'LPIPS',
    hint: 'perceptual distance',
    unit: '',
    better: 'lower',
    color: 'accent',
    bar: (v) => (1 - Math.min(v, 1)) * 100,
    digits: 4,
  },
  identity_sim: {
    label: 'ArcFace ID',
    hint: 'identity similarity',
    unit: '',
    better: 'higher',
    color: 'primary',
    bar: (v) => v * 100,
    digits: 4,
  },
  psnr_masked: {
    label: 'PSNR',
    hint: 'masked region',
    unit: ' dB',
    better: 'higher',
    color: 'success',
    bar: (v) => Math.min(v / 45, 1) * 100,
    digits: 2,
  },
  mae_masked: {
    label: 'MAE',
    hint: 'masked region',
    unit: '',
    better: 'lower',
    color: 'accent',
    bar: (v) => (1 - Math.min(v, 1)) * 100,
    digits: 4,
  },
  mae_full: {
    label: 'MAE',
    hint: 'full image',
    unit: '',
    better: 'lower',
    color: 'primary',
    bar: (v) => (1 - Math.min(v, 1)) * 100,
    digits: 4,
  },
};

const ORDER = ['psnr_full', 'ssim_full', 'lpips', 'identity_sim', 'psnr_masked', 'mae_masked', 'mae_full'];

export default function Evaluation({ evaluation, error }) {
  const metrics = evaluation?.metrics ?? {};
  const keys = [
    ...ORDER.filter((k) => k in metrics),
    ...Object.keys(metrics).filter((k) => !ORDER.includes(k)),
  ];

  return (
    <section id="evaluate" className="mt-5 pt-4 fade-section">
      <h2 className="fw-bold">
        <i className="fas fa-chart-line text-primary me-2" />
        Evaluation
      </h2>

      {error ? (
        <div className="alert alert-danger py-2 px-3 mt-3">
          <i className="fas fa-exclamation-triangle me-1" />
          {error}
        </div>
      ) : null}

      {!error && !evaluation ? <p className="text-muted mt-3">Loading recorded metrics…</p> : null}

      {evaluation && !evaluation.available ? (
        <div className="alert alert-warning py-2 px-3 mt-3">
          <i className="fas fa-circle-info me-1" />
          No evaluation file found at <code>{evaluation.source}</code>.
        </div>
      ) : null}

      {evaluation?.available ? (
        <>
          <p className="text-muted mt-2 mb-0 small">
            Recorded on {evaluation.n_samples?.toLocaleString() ?? '—'} held-out samples · source{' '}
            <code>{evaluation.source}</code>
          </p>

          <div className="row g-4 mt-1">
            {keys.map((key) => {
              const cfg = METRIC_CONFIG[key] ?? {
                label: key,
                hint: '',
                unit: '',
                better: 'higher',
                color: 'primary',
                bar: (v) => Math.min(Math.abs(v) * 100, 100),
                digits: 4,
              };
              const value = metrics[key];
              const width = Math.max(0, Math.min(100, cfg.bar(value)));

              return (
                <div className="col-md-4 col-sm-6" key={key}>
                  <div className="card shadow-sm border-0 rounded-4 p-3 text-center h-100">
                    <h5 className="mb-0">{cfg.label}</h5>
                    <div className="small text-muted mb-2">
                      {cfg.hint}
                      {cfg.hint ? ' · ' : ''}
                      {cfg.better === 'higher' ? '↑ better' : '↓ better'}
                    </div>
                    <div className={`display-6 fw-bold text-${cfg.color}`}>
                      {Number(value).toFixed(cfg.digits)}
                      <span className="fs-6">{cfg.unit}</span>
                    </div>
                    <div className="progress mt-2">
                      <div className={`progress-bar bg-${cfg.color}`} style={{ width: `${width}%` }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <p className="text-muted small mt-3 mb-0">
            <i className="fas fa-circle-info me-1" />
            FID is a distribution-level score and was not part of this evaluation run, so it is not reported here.
          </p>
        </>
      ) : null}
    </section>
  );
}
