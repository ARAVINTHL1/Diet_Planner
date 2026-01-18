from flask import Flask, request, jsonify
from flask_cors import CORS














































For issues, check [Render Docs](https://render.com/docs)## Support- 750 hours/month free compute- Services spin down after inactivity (cold start ~30 seconds)## Free Tier Limitations- `DEBUG`: Set to `False` for production- `PORT`: Auto-set by Render## Environment Variables (Optional)Use the included `render.yaml` file for automated deployment of both services.## Alternative: One-Click Deploy with render.yamlAfter backend is deployed, update your frontend code to use the Render API URL instead of `localhost:5000`.### 3. Update Frontend API URL5. Click **Create Static Site**   - Destination: `/index.html`   - Source: `/*`4. Add **Rewrite Rule**:   - **Plan**: Free   - **Publish Directory**: `dist`   - **Build Command**: `echo "Using pre-built dist"`   - **Name**: `swasthya-ai-frontend`3. Configure:2. Connect the same GitHub repository1. In Render Dashboard, click **New +** → **Static Site**### 2. Deploy Frontend7. Copy your API URL (e.g., `https://swasthya-ai-api.onrender.com`)6. Wait for deployment (5-10 minutes)5. Click **Create Web Service**   - **Plan**: Free   - **Start Command**: `cd ml && gunicorn --bind 0.0.0.0:$PORT api_server:app`   - **Build Command**: `pip install -r ml/requirements_api.txt`   - **Runtime**: Python 3   - **Name**: `swasthya-ai-api`4. Configure:3. Connect your GitHub repository: `ARAVINTHL1/Diet_Planner`2. Click **New +** → **Web Service**1. Go to [Render Dashboard](https://dashboard.render.com/)### 1. Deploy API Backend## Quick Deploy Guideimport joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load ML model
script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, 'food_health_model.joblib'))
label_encoder = joblib.load(os.path.join(script_dir, 'label_encoder.joblib'))

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Swasthya AI ML API is running'
    })

@app.route('/api/predict', methods=['POST'])
def predict_food():
    """
    Predict food healthiness
    
    Expected JSON body:
    {
        "calories": 370,
        "protein": 7.9,
        "carbs": 77.2,
        "fat": 2.9,
        "fiber": 3.5,
        "iron": 1.47,      // optional
        "vitamin_c": 0     // optional
    }
    """
    try:
        data = request.get_json()
        
        # Extract features
        calories = float(data.get('calories', 0))
        protein = float(data.get('protein', 0))
        carbs = float(data.get('carbs', 0))
        fat = float(data.get('fat', 0))
        fiber = float(data.get('fiber', 0))
        iron = float(data.get('iron', 0))
        vitamin_c = float(data.get('vitamin_c', 0))
        
        # Prepare features array
        features = np.array([[calories, protein, carbs, fat, fiber, iron, vitamin_c]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get label
        label = label_encoder.inverse_transform([prediction])[0]
        
        # Calculate health score
        health_score = float(probabilities[2]) * 100  # Healthy probability
        
        # Generate recommendation
        if label == 'Healthy':
            recommendation = f"✓ Great choice! This food is healthy with a score of {health_score:.0f}/100. Include it regularly in your diet."
            emoji = "🥗"
        elif label == 'Moderate':
            recommendation = f"⚠ Moderate choice with a score of {health_score:.0f}/100. Consume in moderation as part of a balanced diet."
            emoji = "🍽️"
        else:
            recommendation = f"✗ Not recommended. Health score: {health_score:.0f}/100. Consider healthier alternatives."
            emoji = "🚫"
        
        # Return result
        return jsonify({
            'success': True,
            'prediction': label,
            'health_score': round(health_score, 1),
            'confidence': {
                'healthy': round(float(probabilities[2]) * 100, 1),
                'moderate': round(float(probabilities[1]) * 100, 1),
                'unhealthy': round(float(probabilities[0]) * 100, 1)
            },
            'recommendation': recommendation,
            'emoji': emoji
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/analyze-meal', methods=['POST'])
def analyze_meal():
    """
    Analyze a complete meal with multiple food items
    
    Expected JSON body:
    {
        "foods": [
            {"name": "Rice", "calories": 370, "protein": 7.9, ...},
            {"name": "Chicken", "calories": 165, "protein": 31, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        foods = data.get('foods', [])
        
        results = []
        total_score = 0
        
        for food in foods:
            name = food.get('name', 'Unknown')
            
            # Extract nutritional values
            calories = float(food.get('calories', 0))
            protein = float(food.get('protein', 0))
            carbs = float(food.get('carbs', 0))
            fat = float(food.get('fat', 0))
            fiber = float(food.get('fiber', 0))
            iron = float(food.get('iron', 0))
            vitamin_c = float(food.get('vitamin_c', 0))
            
            # Predict
            features = np.array([[calories, protein, carbs, fat, fiber, iron, vitamin_c]])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            label = label_encoder.inverse_transform([prediction])[0]
            health_score = float(probabilities[2]) * 100
            
            results.append({
                'name': name,
                'prediction': label,
                'health_score': round(health_score, 1)
            })
            
            total_score += health_score
        
        # Calculate meal score
        meal_score = total_score / len(foods) if foods else 0
        
        if meal_score >= 70:
            meal_rating = "Excellent"
            meal_emoji = "🌟"
        elif meal_score >= 50:
            meal_rating = "Good"
            meal_emoji = "👍"
        else:
            meal_rating = "Needs Improvement"
            meal_emoji = "⚠️"
        
        return jsonify({
            'success': True,
            'meal_score': round(meal_score, 1),
            'meal_rating': meal_rating,
            'meal_emoji': meal_emoji,
            'foods': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print("🚀 Swasthya AI ML API Server")
    print("=" * 60)
    print(f"Server running on port: {port}")
    print("Endpoints:")
    print("  GET  /api/health        - Health check")
    print("  POST /api/predict       - Predict single food")
    print("  POST /api/analyze-meal  - Analyze complete meal")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'False') == 'True')
