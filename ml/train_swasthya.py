import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

def load_sample_data():
    """
    Load the sample nutrition dataset
    """
    path = 'data/sample_nutrition.csv'
    
    if os.path.exists(path):
        print(f"Loading sample nutrition data from {path}...")
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} food items")
        return df
    else:
        print(f"Error: {path} not found.")
        return None

def preprocess_data(df):
    """
    Prepare data for training
    """
    print("\n=== Data Preprocessing ===")
    
    # Features for prediction
    feature_cols = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'iron', 'vitamin_c']
    target_col = 'healthy_label'
    
    # Check if all columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return None, None, None
    
    # Separate features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Encode target labels (Healthy=2, Moderate=1, Unhealthy=0)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target distribution:")
    for label, count in zip(*np.unique(y, return_counts=True)):
        print(f"  {label}: {count}")
    
    return X, y_encoded, label_encoder

def train_model(X, y):
    """
    Train Random Forest Classifier
    """
    print("\n=== Model Training ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== Model Performance ===")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Unhealthy', 'Moderate', 'Healthy']))
    
    # Feature importance
    print("\n=== Feature Importance ===")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))
    
    return model

def save_model(model, label_encoder):
    """
    Save trained model and encoder
    """
    print("\n=== Saving Model ===")
    joblib.dump(model, 'food_health_model.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')
    print("✓ Model saved to 'food_health_model.joblib'")
    print("✓ Label encoder saved to 'label_encoder.joblib'")

def test_prediction(model, label_encoder):
    """
    Test the model with sample predictions
    """
    print("\n=== Sample Predictions ===")
    
    # Test samples: [calories, protein, carbs, fat, fiber, iron, vitamin_c]
    test_samples = [
        ([370, 7.9, 77.2, 2.9, 3.5, 1.47, 0], "Brown Rice"),
        ([165, 31, 0, 3.6, 0, 0.9, 0], "Chicken Breast"),
        ([266, 11, 33, 10, 2.5, 1.5, 2], "Pizza"),
        ([23, 2.9, 3.6, 0.4, 2.2, 2.71, 28.1], "Spinach"),
        ([900, 0, 0, 100, 0, 0, 0], "Ghee")
    ]
    
    for features, food_name in test_samples:
        prediction = model.predict([features])[0]
        label = label_encoder.inverse_transform([prediction])[0]
        probabilities = model.predict_proba([features])[0]
        
        print(f"\n{food_name}:")
        print(f"  Prediction: {label}")
        print(f"  Confidence: Healthy={probabilities[2]:.1%}, Moderate={probabilities[1]:.1%}, Unhealthy={probabilities[0]:.1%}")

if __name__ == "__main__":
    print("=" * 60)
    print("SWASTHYA AI - FOOD HEALTH PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Create data dir if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Load data
    df = load_sample_data()
    
    if df is not None:
        # Preprocess
        X, y, label_encoder = preprocess_data(df)
        
        if X is not None:
            # Train
            model = train_model(X, y)
            
            # Save
            save_model(model, label_encoder)
            
            # Test
            test_prediction(model, label_encoder)
            
            print("\n" + "=" * 60)
            print("✓ TRAINING COMPLETE!")
            print("=" * 60)
        else:
            print("Error: Data preprocessing failed.")
    else:
        print("Error: No data loaded.")
