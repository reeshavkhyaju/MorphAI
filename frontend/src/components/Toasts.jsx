const COLORS = {
  success: '#10B981',
  error: '#EF4444',
  info: '#2563EB',
  warning: '#F59E0B',
};

const ICONS = {
  success: 'fa-check-circle',
  error: 'fa-triangle-exclamation',
  info: 'fa-info-circle',
  warning: 'fa-circle-exclamation',
};

export default function Toasts({ toasts, onDismiss }) {
  return (
    <div className="toast-stack">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className="toast show align-items-center text-white border-0 rounded-4 shadow-lg"
          style={{ background: COLORS[toast.type] ?? COLORS.info }}
          role="alert"
        >
          <div className="d-flex">
            <div className="toast-body">
              <i className={`fas ${ICONS[toast.type] ?? ICONS.info} me-2`} />
              {toast.message}
            </div>
            <button
              type="button"
              className="btn-close btn-close-white me-2 m-auto"
              aria-label="Close"
              onClick={() => onDismiss(toast.id)}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
