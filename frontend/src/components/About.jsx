export default function About({ health }) {
  const checkpoint = health?.checkpoint_name ?? 'morphai_epoch_05.pt';
  const device = health?.device ? health.device.toUpperCase() : '—';

  return (
    <section id="about" className="mt-5 pt-4 fade-section">
      <h2 className="fw-bold">
        <i className="fas fa-info-circle text-primary me-2" />
        About Project
      </h2>
      <div className="row g-4 mt-3">
        <div className="col-md-6">
          <div className="card shadow-sm border-0 rounded-4 p-4 h-100">
            <h5>Overview</h5>
            <p>
              MorphAI leverages Generative Deep Learning to predict facial appearance changes from a single image.
              Focused on eyes, nose and lips, it uses a custom LGNet architecture and MediaPipe Face Mesh for precise
              landmark detection.
            </p>
            <ul className="list-unstyled mt-3 mb-0">
              <li>
                <i className="fas fa-check-circle text-accent me-2" />
                Objective: Generate high-fidelity facial predictions
              </li>
              <li>
                <i className="fas fa-check-circle text-accent me-2" />
                Dataset: CelebA (200k+ images)
              </li>
              <li>
                <i className="fas fa-check-circle text-accent me-2" />
                Framework: PyTorch
              </li>
            </ul>
          </div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm border-0 rounded-4 p-4 h-100">
            <h5>Tech Stack</h5>
            <div className="d-flex flex-wrap gap-2 mt-2">
              <span className="badge bg-light text-dark p-2">
                <i className="fab fa-python me-1" />
                Python
              </span>
              <span className="badge bg-light text-dark p-2">
                <i className="fas fa-fire me-1" />
                PyTorch
              </span>
              <span className="badge bg-light text-dark p-2">
                <i className="fas fa-crop-alt me-1" />
                MediaPipe
              </span>
              <span className="badge bg-light text-dark p-2">
                <i className="fas fa-code me-1" />
                LGNet
              </span>
              <span className="badge bg-light text-dark p-2">
                <i className="fas fa-flask me-1" />
                Flask
              </span>
              <span className="badge bg-light text-dark p-2">
                <i className="fab fa-react me-1" />
                React
              </span>
            </div>
            <hr />
            <div>
              <span className="fw-semibold">Evaluation:</span> PSNR · SSIM · LPIPS · ArcFace identity
            </div>
            <div>
              <span className="fw-semibold">Landmark detection:</span> 468 points (MediaPipe Face Mesh)
            </div>
            <div>
              <span className="fw-semibold">Checkpoint:</span> {checkpoint}
            </div>
            <div>
              <span className="fw-semibold">Inference device:</span> {device}
              {health?.torch_version ? ` · torch ${health.torch_version}` : ''}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
