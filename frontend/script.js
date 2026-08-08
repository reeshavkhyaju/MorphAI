// ==================================================
// MORPHAI - Frontend Interactive Scripts
// ==================================================

document.addEventListener('DOMContentLoaded', function() {

  // ---- DARK MODE TOGGLE ----
  const darkToggle = document.getElementById('darkModeToggle');
  darkToggle.addEventListener('click', function() {
    document.body.classList.toggle('dark-mode');
    const icon = this.querySelector('i');
    if (document.body.classList.contains('dark-mode')) {
      icon.className = 'fas fa-sun';
    } else {
      icon.className = 'fas fa-moon';
    }
  });

  // ---- IMAGE UPLOAD (simulated) ----
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const uploadBtn = document.getElementById('uploadBtn');
  const previewContainer = document.getElementById('previewContainer');
  const imagePreview = document.getElementById('imagePreview');
  const predictBtn = document.getElementById('predictBtn');
  const resetBtn = document.getElementById('resetBtn');
  const origResult = document.getElementById('origResult');
  const genResult = document.getElementById('genResult');
  const statusBadge = document.querySelector('#predictionStatus .badge');
  const confidenceSpan = document.getElementById('confidenceVal');
  const timeSpan = document.getElementById('timeVal');
  const progressStatus = document.getElementById('progressStatus');

  let uploadedFile = null;

  // Click to upload
  uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener('click', () => fileInput.click());

  // Drag & drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#2563EB';
    dropZone.style.background = 'rgba(37,99,235,0.05)';
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = '#cbd5e1';
    dropZone.style.background = '';
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#cbd5e1';
    dropZone.style.background = '';
    if (e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', function() {
    if (this.files.length) {
      handleFile(this.files[0]);
    }
  });

  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      showNotification('Please upload an image file.', 'error');
      return;
    }
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = function(e) {
      imagePreview.src = e.target.result;
      previewContainer.classList.remove('d-none');
      // Also set as original in result
      origResult.src = e.target.result;
      statusBadge.textContent = 'Image loaded';
      statusBadge.className = 'badge bg-warning text-dark';
      confidenceSpan.textContent = '--';
      timeSpan.textContent = '--';
      // Reset generated image
      genResult.src = 'https://placehold.co/150x150/eee/bbb?text=Generated';
      showNotification('Image uploaded successfully!', 'success');
    };
    reader.readAsDataURL(file);
  }

  // ---- PREDICT (simulated) ----
  predictBtn.addEventListener('click', function() {
    if (!uploadedFile) {
      showNotification('Please upload an image first.', 'error');
      return;
    }

    // Disable button & show loading
    this.disabled = true;
    this.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Processing...';
    statusBadge.textContent = 'Processing...';
    statusBadge.className = 'badge bg-info';
    progressStatus.innerHTML = `
      <div class="d-flex align-items-center gap-2">
        <div class="spinner-border spinner-border-sm text-primary" role="status"></div>
        <span>Running LGNet prediction...</span>
      </div>
    `;

    // Simulate AI processing (3-5 sec)
    const startTime = performance.now();
    setTimeout(() => {
      const endTime = ((performance.now() - startTime) / 1000).toFixed(1);
      
      // Dummy generated image (using a placeholder with different text)
      genResult.src = 'https://placehold.co/150x150/2563EB/fff?text=Predicted';
      
      // Update status
      statusBadge.textContent = 'Prediction Complete';
      statusBadge.className = 'badge bg-success';
      confidenceSpan.textContent = (85 + Math.random() * 14).toFixed(1);
      timeSpan.textContent = endTime;
      
      progressStatus.innerHTML = `
        <div class="alert alert-success py-2 px-3 mb-0 rounded-pill">
          <i class="fas fa-check-circle me-1"></i> Prediction finished in ${endTime}s
        </div>
      `;

      // Re-enable button
      this.disabled = false;
      this.innerHTML = '<i class="fas fa-wand-magic-sparkles me-1"></i>Predict';
      
      showNotification('Prediction completed!', 'success');
    }, 2500 + Math.random() * 2500);
  });

  // ---- RESET ----
  resetBtn.addEventListener('click', function() {
    uploadedFile = null;
    fileInput.value = '';
    previewContainer.classList.add('d-none');
    imagePreview.src = '#';
    origResult.src = 'https://placehold.co/150x150/eee/bbb?text=Original';
    genResult.src = 'https://placehold.co/150x150/eee/bbb?text=Generated';
    statusBadge.textContent = 'Waiting for input';
    statusBadge.className = 'badge bg-secondary';
    confidenceSpan.textContent = '--';
    timeSpan.textContent = '--';
    progressStatus.innerHTML = '';
    predictBtn.disabled = false;
    predictBtn.innerHTML = '<i class="fas fa-wand-magic-sparkles me-1"></i>Predict';
    showNotification('Reset successful', 'info');
  });

  // ---- NOTIFICATION SYSTEM ----
  function showNotification(message, type = 'info') {
    const colors = {
      success: '#10B981',
      error: '#EF4444',
      info: '#2563EB',
      warning: '#F59E0B'
    };
    const icon = {
      success: 'fa-check-circle',
      error: 'fa-exclamation-circle',
      info: 'fa-info-circle',
      warning: 'fa-triangle-exclamation'
    };
    const toast = document.createElement('div');
    toast.className = 'position-fixed bottom-0 end-0 p-3';
    toast.style.zIndex = '9999';
    toast.innerHTML = `
      <div class="toast show align-items-center text-white border-0 rounded-4 shadow-lg" 
           style="background: ${colors[type] || '#2563EB'};" role="alert">
        <div class="d-flex">
          <div class="toast-body">
            <i class="fas ${icon[type] || 'fa-info-circle'} me-2"></i> ${message}
          </div>
          <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
      </div>
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.remove();
    }, 4000);
  }

  // ---- TOOLTIPS (Bootstrap 5) ----
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.map(function (el) {
    return new bootstrap.Tooltip(el);
  });

  // ---- BACK TO TOP (via footer link) ----
  // (optional: we add a small floating button via JS)
  const backBtn = document.createElement('button');
  backBtn.className = 'btn btn-primary rounded-circle position-fixed bottom-5 end-5 shadow-lg';
  backBtn.style.bottom = '2rem';
  backBtn.style.right = '2rem';
  backBtn.style.width = '48px';
  backBtn.style.height = '48px';
  backBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
  backBtn.style.display = 'none';
  backBtn.style.zIndex = '999';
  backBtn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
  document.body.appendChild(backBtn);

  window.addEventListener('scroll', function() {
    if (window.scrollY > 400) {
      backBtn.style.display = 'block';
    } else {
      backBtn.style.display = 'none';
    }
  });

  // ---- RIPPLE EFFECT ON BUTTONS ----
  document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('click', function(e) {
      const rect = this.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const ripple = document.createElement('span');
      ripple.className = 'ripple';
      ripple.style.left = x + 'px';
      ripple.style.top = y + 'px';
      ripple.style.position = 'absolute';
      ripple.style.width = '20px';
      ripple.style.height = '20px';
      ripple.style.borderRadius = '50%';
      ripple.style.background = 'rgba(255,255,255,0.4)';
      ripple.style.transform = 'scale(0)';
      ripple.style.animation = 'rippleAnim 0.6s linear';
      ripple.style.pointerEvents = 'none';
      this.style.position = 'relative';
      this.style.overflow = 'hidden';
      this.appendChild(ripple);
      setTimeout(() => ripple.remove(), 700);
    });
  });

  // ---- SLIDER COMPARE (dummy) ----
  const sliderBtn = document.querySelector('#compare .btn-outline-primary');
  if (sliderBtn) {
    sliderBtn.addEventListener('click', function() {
      showNotification('Slider comparison activated (dummy)', 'info');
    });
  }

  // ---- DOWNLOAD BUTTON (dummy) ----
  const downloadBtn = document.querySelector('#compare .btn-outline-primary:last-child');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', function() {
      showNotification('Downloading comparison image... (dummy)', 'success');
    });
  }

  // ---- ZOOM (dummy) ----
  const zoomBtn = document.querySelector('#landmarks .btn-outline-secondary');
  if (zoomBtn) {
    zoomBtn.addEventListener('click', function() {
      showNotification('Zoom feature activated (dummy)', 'info');
    });
  }

  console.log('MorphAI frontend ready ✅');
});