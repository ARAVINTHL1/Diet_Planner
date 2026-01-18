# Machine Learning for Diet Plan

This folder contains scripts for training a machine learning model to predict healthy food options and nutritional quality.

## Datasets
1. [World Food Facts](https://www.kaggle.com/datasets/openfoodfacts/world-food-facts)
2. [Nutrition Dataset for Healthy Food Prediction](https://www.kaggle.com/datasets/yashkaggle27/nutrition-dataset-for-healthy-food-prediction)

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the datasets from Kaggle and place them in `ml/data/`.
   - `world-food-facts` -> `ml/data/en.openfoodfacts.org.products.csv`
   - `nutrition-dataset` -> `ml/data/nutrition.csv` (or similar)

## Training
Run `python train.py` to process the data and train the model. The trained model will be saved as `food_model.joblib`.
