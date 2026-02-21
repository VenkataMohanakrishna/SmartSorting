document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadText = document.querySelector('.upload-text');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultCard = document.getElementById('result-card');
    const loader = document.getElementById('loader');
    
    // Result elements
    const resultImg = document.getElementById('result-img');
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');

    let selectedFile = null;

    // Drag & Drop Handlers
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a valid image file (JPG, PNG, WEBP).');
            return;
        }

        selectedFile = file;
        uploadText.innerHTML = `<strong>${file.name}</strong> selected.`;
        analyzeBtn.disabled = false;
        
        // Preview image in the upload area background (optional enhancement)
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadArea.style.backgroundImage = `url(${e.target.result})`;
            uploadArea.style.backgroundSize = 'cover';
            uploadArea.style.backgroundPosition = 'center';
            uploadArea.querySelector('i').style.opacity = '0';
            uploadArea.querySelector('p').style.background = 'rgba(255,255,255,0.8)';
            uploadArea.querySelector('p').style.padding = '5px 10px';
            uploadArea.querySelector('p').style.borderRadius = '5px';
            uploadArea.style.border = 'none';
        }
        reader.readAsDataURL(file);
    }

    analyzeBtn.addEventListener('click', () => {
        if (!selectedFile) return;

        // Show Loader, Hide details
        loader.style.display = 'block';
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        resultCard.style.display = 'none';

        const formData = new FormData();
        formData.append('file', selectedFile);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loader.style.display = 'none';
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Image';

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Populate Results
            resultCard.style.display = 'block';
            resultImg.src = data.image_url;
            predictionLabel.textContent = data.label;
            
            // Set styles based on class
            if (data.label.toLowerCase().includes('rotten')) {
                predictionLabel.className = 'prediction-label rotten';
                confidenceBar.style.background = 'var(--secondary)';
            } else {
                predictionLabel.className = 'prediction-label healthy';
                confidenceBar.style.background = 'var(--primary)';
            }

            // Animate confidence bar
            setTimeout(() => {
                confidenceBar.style.width = data.confidence;
            }, 100);
            
            confidenceText.textContent = `Confidence: ${data.confidence}`;
            
            // Scroll to results
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        })
        .catch(err => {
            console.error(err);
            loader.style.display = 'none';
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Image';
            alert('An error occurred during analysis.');
        });
    });
});
