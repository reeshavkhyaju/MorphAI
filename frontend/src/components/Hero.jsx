export default function Hero() {
  return (
    <section id="home" className="row align-items-center g-4 fade-section">
      <div className="col-lg-6">
        <span className="badge bg-primary-soft text-primary mb-2 px-3 py-2 rounded-pill">
          <i className="fas fa-wand-magic-sparkles me-1" /> Generative AI
        </span>
        <h1 className="display-4 fw-bold">MorphAI</h1>
        <p className="fs-5 text-secondary">AI-Powered Facial Appearance Prediction System</p>
        <p className="lead">
          Upload a facial image and generate AI-based facial appearance predictions focusing on eyes, nose and lips
          using Generative Deep Learning.
        </p>
        <a href="#predict" className="btn btn-primary btn-lg rounded-pill px-5 mt-2 shadow-lg">
          <i className="fas fa-rocket me-2" />
          Get Started
        </a>
      </div>
      <div className="col-lg-6 text-center">
        <div className="hero-illustration p-4">
          <i className="fas fa-user-astronaut" style={{ fontSize: '8rem', color: '#2563EB', opacity: 0.9 }} />
          {/* text-dark is required: Bootstrap's .badge forces white text, which
              is invisible against bg-white. */}
          <div className="floating-badge badge bg-white text-dark shadow-sm p-3 rounded-4">
            <i className="fas fa-face-smile text-primary me-2" /> AI · Face Mesh
          </div>
        </div>
      </div>
    </section>
  );
}
