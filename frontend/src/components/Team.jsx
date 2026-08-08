const MEMBERS = [
  { name: 'Aditya S.', role: 'Project Lead', email: 'aditya@college.edu', icon: 'fa-user-circle' },
  { name: 'Priya M.', role: 'AI Engineer', email: 'priya@college.edu', icon: 'fa-user-circle' },
  { name: 'Rahul K.', role: 'Frontend', email: 'rahul@college.edu', icon: 'fa-user-circle' },
  { name: 'Dr. Sharma', role: 'Guide', email: 'sharma@college.edu', icon: 'fa-chalkboard-user' },
];

export default function Team() {
  return (
    <section id="team" className="mt-5 pt-4 fade-section">
      <h2 className="fw-bold">
        <i className="fas fa-users text-primary me-2" />
        Team
      </h2>
      <div className="row g-4 mt-3">
        {MEMBERS.map((member) => (
          <div className="col-md-3 col-sm-6" key={member.name}>
            <div className="card shadow-sm border-0 rounded-4 p-3 text-center h-100">
              <i className={`fas ${member.icon} fa-4x text-secondary`} />
              <h6 className="mt-2 mb-0">{member.name}</h6>
              <p className="text-muted small mb-2">{member.role}</p>
              <span className="badge bg-light text-dark text-truncate">{member.email}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
