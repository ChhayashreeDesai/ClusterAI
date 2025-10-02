// ClusterAI JavaScript Functions

// Global variables
let uploadedFile = null;
let analysisInProgress = false;

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize file upload functionality
    initializeFileUpload();
    
    // Initialize animations
    initializeAnimations();
    
    // Initialize form validations
    initializeFormValidation();
    
    // Initialize tooltips and popovers
    initializeTooltips();
}

// File Upload Functionality
function initializeFileUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    
    if (!uploadArea || !fileInput) return;
    
    // Drag and drop handlers
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File input change handler
    fileInput.addEventListener('change', handleFileSelect);
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file
    const validation = validateFile(file);
    if (!validation.valid) {
        showAlert(validation.message, 'danger');
        return;
    }
    
    uploadedFile = file;
    displayFileInfo(file);
    enableAnalysisButton();
    
    // Preview file content if CSV
    if (file.name.endsWith('.csv')) {
        previewCSVFile(file);
    }
}

function validateFile(file) {
    // Check file type
    const allowedTypes = ['.csv', '.xlsx', '.xls'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
        return {
            valid: false,
            message: 'Please upload a CSV or Excel file (.csv, .xlsx, .xls)'
        };
    }
    
    // Check file size (16MB limit)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        return {
            valid: false,
            message: 'File size must be less than 16MB'
        };
    }
    
    return { valid: true };
}

function displayFileInfo(file) {
    const fileInfoDiv = document.getElementById('file-info');
    if (!fileInfoDiv) return;
    
    const fileSize = formatFileSize(file.size);
    const fileType = file.name.split('.').pop().toUpperCase();
    
    fileInfoDiv.innerHTML = `
        <div class="file-info">
            <div class="row align-items-center">
                <div class="col-auto">
                    <i class="fas fa-file-${getFileIcon(file.name)} file-icon"></i>
                </div>
                <div class="col">
                    <h6 class="mb-1">${file.name}</h6>
                    <small class="text-muted">${fileSize} • ${fileType}</small>
                </div>
                <div class="col-auto">
                    <button class="btn btn-sm btn-outline-danger" onclick="removeFile()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        </div>
    `;
    
    fileInfoDiv.style.display = 'block';
}

function getFileIcon(filename) {
    const extension = filename.split('.').pop().toLowerCase();
    switch (extension) {
        case 'csv': return 'csv';
        case 'xlsx':
        case 'xls': return 'excel';
        default: return 'alt';
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function removeFile() {
    uploadedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('file-info').style.display = 'none';
    document.getElementById('sample-data').style.display = 'none';
    disableAnalysisButton();
}

function previewCSVFile(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const csv = e.target.result;
        const lines = csv.split('\n').slice(0, 6); // First 5 rows + header
        const sampleDataDiv = document.getElementById('sample-data');
        
        if (sampleDataDiv && lines.length > 1) {
            const preview = lines.map(line => {
                const cells = line.split(',').slice(0, 5); // First 5 columns
                return cells.join(' | ');
            }).join('\n');
            
            sampleDataDiv.innerHTML = `
                <h6><i class="fas fa-eye me-2"></i>Data Preview</h6>
                <div class="sample-data">
${preview}
${lines.length > 5 ? '...' : ''}
                </div>
            `;
            sampleDataDiv.style.display = 'block';
        }
    };
    reader.readAsText(file.slice(0, 1024)); // Read first 1KB for preview
}

function enableAnalysisButton() {
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-chart-line me-2"></i>Analyze Data';
    }
}

function disableAnalysisButton() {
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Upload File First';
    }
}

// Analysis Functions
function startAnalysis() {
    if (!uploadedFile || analysisInProgress) return;
    
    analysisInProgress = true;
    showAnalysisProgress();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Analysis failed');
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            window.location.href = data.redirect_url;
        } else {
            throw new Error(data.message || 'Analysis failed');
        }
    })
    .catch(error => {
        hideAnalysisProgress();
        showAlert(error.message, 'danger');
        analysisInProgress = false;
    });
}

function showAnalysisProgress() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const progressDiv = document.getElementById('analysis-progress');
    
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="loading"></span> Analyzing...';
    }
    
    if (progressDiv) {
        progressDiv.style.display = 'block';
        animateProgress();
    }
}

function hideAnalysisProgress() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const progressDiv = document.getElementById('analysis-progress');
    
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-chart-line me-2"></i>Analyze Data';
    }
    
    if (progressDiv) {
        progressDiv.style.display = 'none';
    }
}

function animateProgress() {
    const progressBar = document.querySelector('#analysis-progress .progress-bar');
    if (!progressBar) return;
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress >= 90) {
            progress = 90;
            clearInterval(interval);
        }
        progressBar.style.width = progress + '%';
    }, 500);
}

// Animation Functions
function initializeAnimations() {
    // Animate statistics on scroll
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateStatNumbers(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe stat elements
    document.querySelectorAll('.stat-number').forEach(stat => {
        observer.observe(stat);
    });
    
    // Animate cards on scroll
    document.querySelectorAll('.feature-card').forEach(card => {
        observer.observe(card);
    });
}

function animateStatNumbers(element) {
    const finalNumber = parseInt(element.dataset.count || element.textContent);
    const duration = 2000;
    const startTime = Date.now();
    
    function updateNumber() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const currentNumber = Math.floor(finalNumber * easeOut);
        
        element.textContent = currentNumber.toLocaleString();
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        } else {
            element.textContent = finalNumber.toLocaleString();
        }
    }
    
    updateNumber();
}

// Form Validation
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

// Tooltip Initialization
function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// Utility Functions
function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.getElementById('alert-container') || createAlertContainer();
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    alertContainer.appendChild(alertDiv);
    
    // Auto-dismiss after duration
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, duration);
}

function createAlertContainer() {
    const container = document.createElement('div');
    container.id = 'alert-container';
    container.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
    document.body.appendChild(container);
    return container;
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showAlert('Copied to clipboard!', 'success', 2000);
    }).catch(() => {
        showAlert('Failed to copy to clipboard', 'danger', 2000);
    });
}

function downloadData(data, filename, type = 'application/json') {
    const blob = new Blob([data], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Export Functions for Results Page
function exportToCSV() {
    // This would be implemented with actual data from the analysis
    showAlert('CSV export functionality will be implemented with actual analysis data', 'info');
}

function exportToExcel() {
    showAlert('Excel export functionality will be implemented with actual analysis data', 'info');
}

function exportToPDF() {
    showAlert('PDF export functionality will be implemented with actual analysis data', 'info');
}

// Demo Functions
function runDemo() {
    showAlert('Loading demo dataset...', 'info');
    
    fetch('/demo')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            window.location.href = data.redirect_url;
        } else {
            throw new Error(data.message || 'Demo failed to load');
        }
    })
    .catch(error => {
        showAlert(error.message, 'danger');
    });
}

// Global error handler
window.addEventListener('error', function(e) {
    console.error('JavaScript Error:', e.error);
    showAlert('An unexpected error occurred. Please refresh the page and try again.', 'danger');
});

// Global event listeners
document.addEventListener('click', function(e) {
    // Handle demo button clicks
    if (e.target.matches('.demo-btn, .demo-btn *')) {
        e.preventDefault();
        runDemo();
    }
    
    // Handle analysis button clicks
    if (e.target.matches('#analyze-btn')) {
        e.preventDefault();
        startAnalysis();
    }
});

// Smooth scrolling for anchor links
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="#"]')) {
        e.preventDefault();
        const target = document.querySelector(e.target.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + U for upload
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.click();
        }
    }
    
    // Escape to close modals/alerts
    if (e.key === 'Escape') {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(alert => alert.remove());
    }
});