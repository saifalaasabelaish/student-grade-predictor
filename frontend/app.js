/**
 * Frontend JavaScript for Student Grade Predictor
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const btnText = document.getElementById('btnText');
const loadingIcon = document.getElementById('loadingIcon');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const predictionResults = document.getElementById('predictionResults');
const errorMessage = document.getElementById('errorMessage');

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    // Check API health on page load
    checkAPIHealth();
    
    // Add form submit event listener
    predictionForm.addEventListener('submit', handleFormSubmit);
    
    // Add input change listeners for real-time validation
    addInputValidation();
});

/**
 * Check if the API is healthy and model is loaded
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (!data.model_loaded) {
            showError('Model not loaded. Please ensure the model is trained and saved.');
        }
    } catch (error) {
        console.warn('Could not check API health:', error);
        showError('Could not connect to the prediction service. Please ensure the API server is running.');
    }
}

/**
 * Handle form submission
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    
    // Get form data
    const formData = new FormData(predictionForm);
    const studentData = {
        gender: formData.get('gender'),
        transportation: formData.get('transportation'),
        accommodation: formData.get('accommodation'),
        mid_exam: formData.get('mid_exam'),
        taking_notes: formData.get('taking_notes')
    };
    
    // Validate form data
    if (!validateFormData(studentData)) {
        showError('Please fill in all fields.');
        return;
    }
    
    // Show loading state
    setLoadingState(true);
    hideResults();
    
    try {
        // Make prediction request
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(studentData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const prediction = await response.json();
        showResults(prediction, studentData);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(`Prediction failed: ${error.message}`);
    } finally {
        setLoadingState(false);
    }
}

/**
 * Validate form data
 */
function validateFormData(data) {
    return Object.values(data).every(value => value && value.trim() !== '');
}

/**
 * Set loading state for the submit button
 */
function setLoadingState(isLoading) {
    if (isLoading) {
        predictBtn.disabled = true;
        btnText.textContent = 'Predicting...';
        loadingIcon.classList.remove('hidden');
    } else {
        predictBtn.disabled = false;
        btnText.textContent = 'Predict Grade';
        loadingIcon.classList.add('hidden');
    }
}

/**
 * Show prediction results
 */
function showResults(prediction, inputData) {
    hideError();
    
    const grade = prediction.predicted_grade;
    const confidence = (prediction.confidence * 100).toFixed(1);
    const probabilities = prediction.probabilities;
    const modelInfo = prediction.model_info;
    
    // Create results HTML
    const resultsHTML = `
        <div class="grid md:grid-cols-2 gap-6">
            <!-- Main Prediction -->
            <div class="text-center">
                <div class="grade-${grade} text-white rounded-xl p-6 mb-4">
                    <h3 class="text-3xl font-bold mb-2">Predicted Grade</h3>
                    <div class="text-6xl font-bold mb-2">${grade}</div>
                    <p class="text-lg">Confidence: ${confidence}%</p>
                </div>
                <div class="text-sm text-gray-600">
                    <p><strong>Model:</strong> ${modelInfo.version || 'Unknown'}</p>
                    <p><strong>Accuracy:</strong> ${modelInfo.test_accuracy ? (modelInfo.test_accuracy * 100).toFixed(1) + '%' : 'Unknown'}</p>
                </div>
            </div>
            
            <!-- Probability Distribution -->
            <div>
                <h4 class="text-lg font-semibold mb-4">Grade Probabilities</h4>
                <div class="space-y-3">
                    ${Object.entries(probabilities)
                        .sort(([,a], [,b]) => b - a)
                        .map(([gradeLabel, prob]) => {
                            const percentage = (prob * 100).toFixed(1);
                            const isTop = gradeLabel === grade;
                            return `
                                <div class="flex items-center">
                                    <span class="w-8 text-center font-semibold ${isTop ? 'text-blue-600' : ''}">${gradeLabel}</span>
                                    <div class="flex-1 mx-3">
                                        <div class="bg-gray-200 rounded-full h-3">
                                            <div class="h-3 rounded-full transition-all duration-500 ${
                                                isTop ? 'bg-blue-500' : 'bg-gray-400'
                                            }" style="width: ${percentage}%"></div>
                                        </div>
                                    </div>
                                    <span class="text-sm text-gray-600 w-12 text-right">${percentage}%</span>
                                </div>
                            `;
                        }).join('')}
                </div>
            </div>
        </div>
        
        <!-- Input Summary -->
        <div class="mt-6 pt-6 border-t border-gray-200">
            <h4 class="text-lg font-semibold mb-3">Input Summary</h4>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <div><strong>Gender:</strong> ${inputData.gender}</div>
                <div><strong>Transportation:</strong> ${inputData.transportation}</div>
                <div><strong>Accommodation:</strong> ${inputData.accommodation}</div>
                <div><strong>Mid-term Prep:</strong> ${inputData.mid_exam}</div>
                <div><strong>Taking Notes:</strong> ${inputData.taking_notes}</div>
            </div>
        </div>
        
        <!-- Actions -->
        <div class="mt-6 text-center">
            <button onclick="resetForm()" class="bg-gray-500 hover:bg-gray-600 text-white px-6 py-2 rounded-lg mr-4 transition duration-200">
                <i class="fas fa-redo mr-2"></i>New Prediction
            </button>
            <button onclick="downloadResults()" class="bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded-lg transition duration-200">
                <i class="fas fa-download mr-2"></i>Download Results
            </button>
        </div>
    `;
    
    predictionResults.innerHTML = resultsHTML;
    resultsSection.classList.remove('hidden');
    resultsSection.classList.add('show');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Show error message
 */
function showError(message) {
    hideResults();
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
    errorSection.classList.add('show');
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Hide error message
 */
function hideError() {
    errorSection.classList.add('hidden');
    errorSection.classList.remove('show');
}

/**
 * Hide results
 */
function hideResults() {
    resultsSection.classList.add('hidden');
    resultsSection.classList.remove('show');
}

/**
 * Reset the form
 */
function resetForm() {
    predictionForm.reset();
    hideResults();
    hideError();
    
    // Scroll back to form
    predictionForm.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Download results as JSON
 */
function downloadResults() {
    const results = {
        timestamp: new Date().toISOString(),
        prediction: predictionResults.textContent,
        // Add more structured data here if needed
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `grade-prediction-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Add input validation and styling
 */
function addInputValidation() {
    const inputs = predictionForm.querySelectorAll('select');
    
    inputs.forEach(input => {
        input.addEventListener('change', function() {
            if (this.value) {
                this.classList.remove('border-red-300');
                this.classList.add('border-green-300');
            } else {
                this.classList.remove('border-green-300');
                this.classList.add('border-red-300');
            }
        });
    });
}

/**
 * Handle keyboard shortcuts
 */
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to submit form
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        if (!predictBtn.disabled) {
            predictionForm.dispatchEvent(new Event('submit'));
        }
    }
    
    // Escape to reset form
    if (event.key === 'Escape') {
        resetForm();
    }
});

/**
 * Add some interactive features
 */
function addInteractiveFeatures() {
    // Add hover effects to form fields
    const formFields = document.querySelectorAll('select');
    formFields.forEach(field => {
        field.addEventListener('focus', function() {
            this.parentElement.classList.add('transform', 'scale-105');
        });
        
        field.addEventListener('blur', function() {
            this.parentElement.classList.remove('transform', 'scale-105');
        });
    });
}

// Initialize interactive features
document.addEventListener('DOMContentLoaded', addInteractiveFeatures);