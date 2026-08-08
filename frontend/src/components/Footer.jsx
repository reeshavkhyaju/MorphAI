export default function Footer({ health }) {
  return (
    <footer className="mt-5 pt-4 border-top text-center text-muted small">
      <p className="mb-1">
        <i className="fas fa-copyright me-1" />
        2026 MorphAI · Computer Engineering Project
      </p>
      <p className="mb-3">
        React frontend · Flask backend · LGNet generator
        {health?.checkpoint_name ? ` · ${health.checkpoint_name}` : ''}
      </p>
    </footer>
  );
}
