# Student Grade Predictor

A complete machine learning application that predicts student grades using a Naive Bayes classifier. The project includes data preprocessing, model training, evaluation, and a full-stack web application with FastAPI backend and modern HTML/CSS/JavaScript frontend.

## 🏗️ Project Structure

```
student_grade_predictor/
├── data/                          # Data loading utilities
│   ├── __init__.py
│   └── loader.py                  # CSV loading and feature extraction
├── models/                        # Machine learning models
│   ├── __init__.py
│   ├── naive_bayes.py            # Naive Bayes classifier implementation
│   └── serialization.py         # Model saving/loading utilities
├── evaluation/                    # Model evaluation and plotting
│   ├── __init__.py
│   ├── metrics.py                # Accuracy calculation
│   └── analysis.py               # Performance comparison plotting
├── api/                          # FastAPI backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   └── schemas.py                # Pydantic data models
├── frontend/                     # Web frontend
│   ├── index.html                # Main HTML page
│   └── app.js                    # JavaScript functionality
├── saved_models/                 # Trained models storage
│   └── best_model_improved.pkl  # Best performing model
├── preprocessed_data.csv         # Integer-encoded preprocessed dataset
├── main.py                       # Original training pipeline
├── requirements.txt              # Python dependencies
└── README.md                   
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/saifalaasabelaish/student-grade-predictor
cd student_grade_predictor

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing (Already Done)

The data has been preprocessed using the included Jupyter notebook (`preprocessing.ipynb`):

**Original Data Format:**
- Raw categorical strings (e.g., "female", "Bus", "Dorms")

**Preprocessed Data Format (`preprocessed_data.csv`):**
- All categorical variables encoded as integers
- Clean, stratified data ready for machine learning
- 100 samples with 7 columns (including STUDENT ID and GRADE)

**Encoding Mappings:**
- **Gender**: 0=female, 1=male
- **Transportation**: 0=Bus, 1=Car  
- **Accommodation**: 0=Dorms, 1=With Family
- **MidExam Preparation**: 0=Closest day to exam, 1=Regularly during semester
- **Taking Notes**: 0=Always, 1=Sometimes
- **Grade**: 0=A, 1=B, 2=C, 3=D

### 3. Train Models

```bash
# Run the training pipeline 

python main.py
```

### 4. Start the API Server

```bash
# Navigate to API directory
cd api

# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

### 5. Open the Frontend

```bash
# Option 1: Open directly in browser
start frontend/index.html

# Option 2: Serve with Python HTTP server
cd frontend
python -m http.server 8080
# Then open http://localhost:8080
```

## 📊 Features

### Data Preprocessing
- **Integer Encoding**: All categorical variables converted to integers for efficient processing
- **Stratified Sampling**: Ensures balanced class distribution in train/validation/test splits
- **Data Quality**: Cleaned data with removed NaN values and consistent formatting
- **Feature Analysis**: Comprehensive analysis of feature-grade relationships

### Machine Learning Pipeline
- **Naive Bayes Implementation**: Built from scratch with Laplace smoothing
- **Model Variants**: Tests 4 different smoothing parameters (k=0,1,2,3)
- **Stratified Validation**: 80% train, 10% holdout, 10% test with balanced splits
- **Performance Tracking**: Training vs validation accuracy to detect overfitting
- **Automatic Selection**: Chooses best performing model based on holdout accuracy

### API Features
- **Intelligent Conversion**: Converts string inputs to integer encoding automatically
- **Grade Mapping**: Converts integer predictions back to letter grades (A,B,C,D)
- **Robust Error Handling**: Handles NaN values, edge cases, and validation errors
- **Batch Predictions**: Support for multiple student predictions
- **Model Metadata**: Returns model version, k-value, and test accuracy

### Frontend Features
- **User-Friendly Interface**: Clean forms with categorical dropdowns
- **Real-Time Validation**: Instant feedback on form completion
- **Visual Results**: Grade prediction with confidence scores and probability distribution
- **Interactive Charts**: Visual representation of prediction probabilities
- **Export Functionality**: Download predictions as JSON

## 🔧 API Endpoints

### Health Check
```
GET /health
```
Returns API status and model loading status.

### Single Prediction
```
POST /predict
Content-Type: application/json

{
  "gender": "female",
  "transportation": "Bus",
  "accommodation": "Dorms",
  "mid_exam": "Regularly During The Semester",
  "taking_notes": "Always"
}
```

**Response:**
```json
{
  "predicted_grade": "A",
  "confidence": 0.85,
  "probabilities": {
    "A": 0.85,
    "B": 0.10,
    "C": 0.03,
    "D": 0.02
  },
  "model_info": {
    "version": "v1",
    "k_value": 0,
    "test_accuracy": 0.5
  }
}
```

### Batch Prediction
```
POST /predict/batch
Content-Type: application/json

[
  {
    "gender": "female",
    "transportation": "Bus",
    "accommodation": "Dorms",
    "mid_exam": "Regularly During The Semester",
    "taking_notes": "Always"
  }
]
```

## 📈 Model Performance & Data Insights

### Key Findings from Data Analysis

**Most Predictive Feature - Midterm Preparation:**
- Students who prepare regularly: 92% get A's or B's
- Students who cram (closest day): 73% get C's or D's
- **This is the strongest predictor of academic success!**

**Second Most Predictive - Note Taking:**
- Students who always take notes: Balanced grade distribution
- Students who sometimes take notes: 55% get D's

**Class Distribution:**
- Grade A: 18% (least common)
- Grade B: 25% 
- Grade C: 19%
- Grade D: 38% (most common)

### Model Performance
With the improved pipeline using stratified sampling:
- **Training Accuracy**: ~56%
- **Holdout Accuracy**: ~30%
- **Test Accuracy**: ~50%
- **Best Model**: Usually k=0 or k=1 (minimal smoothing works best)

*Note: Performance is limited by small dataset size (100 samples). Larger datasets typically achieve 80-90%+ accuracy.*

## 🧪 Testing the Application

### 1. Test the Training Pipeline
```bash
# Test improved pipeline (recommended)
python main_improved.py
```

Expected output:
- Comprehensive data analysis with feature relationships
- Training progress for each k value (v1, v2, v3, v4)
- Overfitting detection (training vs holdout accuracy)
- Performance comparison plot
- Best model selection and saving

### 2. Test the API
```bash
# Start the API server
cd api
uvicorn main:app --reload

# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "transportation": "Bus",
    "accommodation": "Dorms",
    "mid_exam": "Regularly During The Semester",
    "taking_notes": "Always"
  }'
```

### 3. Test the Frontend
1. Open `frontend/index.html` in a web browser
2. Fill out the student information form
3. Click "Predict Grade"
4. Verify prediction results with confidence scores

## 📁 Data Format

### Preprocessed Data (`preprocessed_data.csv`)
The model works with integer-encoded categorical data:

| Column | Encoded Values | Original Meaning |
|--------|---------------|------------------|
| Gender | 0, 1 | 0=female, 1=male |
| Transportation | 0, 1 | 0=Bus, 1=Car |
| Accommodation | 0, 1 | 0=Dorms, 1=With Family |
| Preparation to midterm | 0, 1 | 0=Closest day, 1=Regularly |
| Taking notes in classes | 0, 1 | 0=Always, 1=Sometimes |
| GRADE | 0, 1, 2, 3 | 0=A, 1=B, 2=C, 3=D |

### API Input Format (User-Friendly Strings)
```json
{
  "gender": "male|female",
  "transportation": "Bus|Car",
  "accommodation": "Dorms|With Familly",
  "mid_exam": "Regularly During The Semester|Closest Day To The Exam",
  "taking_notes": "Always|Sometimes"
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   ```
   Solution: Run `python main_improved.py` to train and save the improved model
   ```

2. **Prediction validation errors**
   ```
   Solution: Ensure the improved model is trained and API is restarted
   ```

3. **Low accuracy results**
   ```
   Expected: Small dataset (100 samples) limits performance
   Solution: Collect more data for better accuracy
   ```

4. **Import errors**
   ```
   Solution: Install dependencies: pip install -r requirements.txt
   ```

5. **Integer vs string errors**
   ```
   Solution: Use main_improved.py which handles integer encoding properly
   ```

## 🔄 Development Workflow

### Data Preprocessing
1. Use `preprocessing.ipynb` for data exploration and encoding
2. Save cleaned data as `preprocessed_data.csv`
3. Update encoding mappings in API if data format changes

### Model Training
1. Run `python main_improved.py` for best results
2. Check data analysis output for feature insights
3. Monitor overfitting in training vs holdout accuracy

### API Development
1. Test endpoints with `http://localhost:8000/docs`
2. Verify string→integer→string conversion pipeline
3. Check logs for prediction debugging

## 📊 Performance Analysis

### Why Accuracy is Lower Than Expected
- **Small Dataset**: Only 100 samples (need 500+ for reliable ML)
- **Class Imbalance**: 38% D grades vs 18% A grades
- **Limited Features**: Only 5 basic categorical features
- **High Variance**: Small validation sets (10 samples) create unreliable estimates

### Model Insights
- **Study Habits Matter Most**: Preparation timing is the strongest predictor
- **Note-Taking Impact**: Consistent note-taking improves grades significantly
- **Feature Engineering Opportunity**: Combine related features for better prediction

## 🚀 Deployment

### Local Development
```bash
# Start API server
cd api && uvicorn main:app --reload

# Serve frontend
cd frontend && python -m http.server 8080
```

### Production Considerations
1. **Larger Dataset**: Collect more samples for better accuracy
2. **Feature Engineering**: Add numerical features (GPA, test scores, attendance)
3. **Model Validation**: Use cross-validation with larger datasets
4. **Monitoring**: Track prediction accuracy over time

## 📝 Assignment Requirements Met

✅ **Naive Bayes from Scratch**: Complete implementation with Laplace smoothing  
✅ **Multiple K Values**: Tests k=0,1,2,3 with performance comparison  
✅ **Data Splitting**: 80% train, 10% holdout, 10% test  
✅ **Model Selection**: Automatic selection based on holdout performance  
✅ **Performance Plot**: X-axis=version, Y-axis=accuracy percentage  
✅ **Model Persistence**: Best model saved and loadable  
✅ **Full-Stack Application**: FastAPI + Frontend integration  
✅ **Modular Code**: Clean separation of data, models, evaluation, API

## 🎓 Educational Value

This project demonstrates:
- **Practical Machine Learning**: Real implementation challenges and solutions
- **Data Preprocessing**: Importance of proper categorical encoding
- **Model Evaluation**: Overfitting detection and validation strategies  
- **Feature Analysis**: Understanding what drives academic performance
- **Full-Stack Development**: Complete ML application deployment

The insights about study habits and academic performance are valuable beyond the technical implementation!
