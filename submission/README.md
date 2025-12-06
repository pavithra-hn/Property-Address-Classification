# Property Address Classification

## Project Overview
This project focuses on classifying property addresses into 5 distinct categories using Machine Learning. The goal is to accurately categorize unstructured address text into:
- `flat`
- `houseorplot`
- `landparcel`
- `commercial unit`
- `others`

## Approach
1. **Data Preprocessing**: 
   - Text cleaning (lowercase, removal of special characters).
   - TF-IDF Vectorization (`max_features=5000`) to convert text to numerical features.
2. **Model**: 
   - **Logistic Regression** was chosen as the baseline model due to its efficiency with high-dimensional sparse data and interpretability.
   - Configured with `class_weight='balanced'` to handle class imbalances.

## Performance
The model achieves a **Validation Accuracy of ~87%**.

**Classification Report:**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 0.93 | 0.93 | 0.93 |
| flat | 0.96 | 0.89 | 0.92 |
| houseorplot | 0.90 | 0.84 | 0.87 |
| landparcel | 0.72 | 0.85 | 0.78 |
| others | 0.68 | 0.83 | 0.75 |

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- seaborn
- jupyter

## Project Structure
- `submission/Property_Classification.ipynb`: Main Jupyter Notebook with the end-to-end solution.
- `submission/requirements.txt`: Python dependencies.
- `task_dataset - training_dataset.csv`: Training data.
- `task_dataset - validation_dataset.csv`: Validation data.
- `best_model/`: Directory containing saved model artifacts.

## Usage
1. Install dependencies:
   ```bash
   pip install -r submission/requirements.txt
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook submission/Property_Classification.ipynb
   ```
3. Run the cells to train the model or use the saved model for inference.
