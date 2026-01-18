import joblib
import numpy as np
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model = joblib.load(os.path.join(script_dir, 'food_health_model.joblib'))
label_encoder = joblib.load(os.path.join(script_dir, 'label_encoder.joblib'))


def predict_food_health(calories, protein, carbs, fat, fiber, iron=0, vitamin_c=0):
    """
    Predict if a food is Healthy, Moderate, or Unhealthy
    
    Args:
        calories: Calories per 100g
        protein: Protein in grams per 100g
        carbs: Carbohydrates in grams per 100g
        fat: Fat in grams per 100g
        fiber: Fiber in grams per 100g
        iron: Iron in mg per 100g (optional)
        vitamin_c: Vitamin C in mg per 100g (optional)
    
    Returns:
        dict with prediction and confidence scores
    """
    # Prepare features
    features = np.array([[calories, protein, carbs, fat, fiber, iron, vitamin_c]])
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get label
    label = label_encoder.inverse_transform([prediction])[0]
    
    # Create result
    result = {
        'prediction': label,
        'confidence': {
            'Healthy': float(probabilities[2]),
            'Moderate': float(probabilities[1]),
            'Unhealthy': float(probabilities[0])
        },
        'score': float(probabilities[2]) * 100  # Health score out of 100
    }
    
    return result

def get_health_recommendation(prediction_result):
    """
    Get health recommendation based on prediction
    """
    label = prediction_result['prediction']
    score = prediction_result['score']
    
    if label == 'Healthy':
        return f"✓ Great choice! This food is healthy with a score of {score:.0f}/100. Include it regularly in your diet."
    elif label == 'Moderate':
        return f"⚠ Moderate choice with a score of {score:.0f}/100. Consume in moderation as part of a balanced diet."
    else:
        return f"✗ Not recommended. Health score: {score:.0f}/100. Consider healthier alternatives."

# Example usage
if __name__ == "__main__":
    print("=== Swasthya AI - Food Health Predictor ===\n")
    
    # Test with some foods
    test_foods = [
        {"name": "Brown Rice", "calories": 370, "protein": 7.9, "carbs": 77.2, "fat": 2.9, "fiber": 3.5},
        {"name": "Pizza", "calories": 266, "protein": 11, "carbs": 33, "fat": 10, "fiber": 2.5},
        {"name": "Spinach", "calories": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4, "fiber": 2.2, "iron": 2.71, "vitamin_c": 28.1},
    ]
    
    for food in test_foods:
        name = food.pop('name')
        result = predict_food_health(**food)
        recommendation = get_health_recommendation(result)
        
        print(f"{name}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Health Score: {result['score']:.0f}/100")
        print(f"  {recommendation}\n")
