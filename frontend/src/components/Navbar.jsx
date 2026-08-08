import { useState } from 'react';

export default function Navbar({ sections, active, dark, onToggleDark, health, healthError }) {
  const [open, setOpen] = useState(false);

  let dotClass = 'pending';
  let statusText = 'Connecting…';
  if (healthError) {
    dotClass = 'offline';
    statusText = 'Backend offline';
  } else if (health) {
    dotClass = health.model_loaded ? 'online' : 'offline';
    statusText = health.model_loaded ? `Model ready · ${health.device.toUpperCase()}` : 'Model not loaded';
  }

  return (
    <nav className="navbar navbar-expand-lg sticky-top glass-nav">
      <div className="container">
        <a className="navbar-brand fw-bold" href="#home">
          <i className="fas fa-brain me-2 text-primary" />
          MorphAI
        </a>

        <button
          className="navbar-toggler border-0"
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon" />
        </button>

        <div className={`collapse navbar-collapse${open ? ' show' : ''}`}>
          <ul className="navbar-nav ms-auto align-items-lg-center">
            {sections.map((section) => (
              <li className="nav-item" key={section.id}>
                <a
                  className={`nav-link${active === section.id ? ' active' : ''}`}
                  href={`#${section.id}`}
                  onClick={() => setOpen(false)}
                >
                  {section.label}
                </a>
              </li>
            ))}
            <li className="nav-item ms-lg-3 d-flex align-items-center">
              <span className="small text-secondary text-nowrap" title={healthError || statusText}>
                <span className={`status-dot ${dotClass}`} />
                {statusText}
              </span>
            </li>
            <li className="nav-item ms-lg-2 mt-2 mt-lg-0">
              <button
                type="button"
                className="btn btn-outline-primary btn-sm rounded-pill px-3"
                onClick={onToggleDark}
                aria-label="Toggle dark mode"
              >
                <i className={dark ? 'fas fa-sun' : 'fas fa-moon'} />
              </button>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
