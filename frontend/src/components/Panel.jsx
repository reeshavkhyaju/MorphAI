export default function Panel({ title, src, caption, zoomed = false, onClick }) {
  return (
    <div className="text-center">
      <div
        className={`panel-frame${zoomed ? ' zoomed' : ''}`}
        onClick={onClick}
        style={onClick ? { cursor: 'zoom-in' } : undefined}
      >
        {src ? (
          <img src={src} alt={title} />
        ) : (
          <span className="placeholder-text">
            <i className="fas fa-image fa-2x d-block mb-2 opacity-50" />
            {title}
          </span>
        )}
      </div>
      <div className="small fw-semibold mt-2">{title}</div>
      {caption ? <div className="small text-muted">{caption}</div> : null}
    </div>
  );
}
