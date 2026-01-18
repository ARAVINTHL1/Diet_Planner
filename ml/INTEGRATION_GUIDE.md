# How to Connect ML Model to Your Swasthya AI Website

## 🎯 Quick Start

### Step 1: Install Flask Dependencies
```bash
cd ml
pip install -r requirements_api.txt
```

### Step 2: Start the ML API Server
```bash
cd ml
python api_server.py
```

The server will start on `http://localhost:5000`

### Step 3: Test the API

#### Test Health Check:
```bash
curl http://localhost:5000/api/health
```

#### Test Food Prediction:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "calories": 370,
    "protein": 7.9,
    "carbs": 77.2,
    "fat": 2.9,
    "fiber": 3.5
  }'
```

## 📡 API Endpoints

### 1. Health Check
```
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Swasthya AI ML API is running"
}
```

### 2. Predict Single Food
```
POST /api/predict
```

**Request Body:**
```json
{
  "calories": 370,
  "protein": 7.9,
  "carbs": 77.2,
  "fat": 2.9,
  "fiber": 3.5,
  "iron": 1.47,
  "vitamin_c": 0
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "Healthy",
  "health_score": 85.5,
  "confidence": {
    "healthy": 85.5,
    "moderate": 12.3,
    "unhealthy": 2.2
  },
  "recommendation": "✓ Great choice! This food is healthy...",
  "emoji": "🥗"
}
```

### 3. Analyze Complete Meal
```
POST /api/analyze-meal
```

**Request Body:**
```json
{
  "foods": [
    {
      "name": "Brown Rice",
      "calories": 370,
      "protein": 7.9,
      "carbs": 77.2,
      "fat": 2.9,
      "fiber": 3.5
    },
    {
      "name": "Chicken Breast",
      "calories": 165,
      "protein": 31,
      "carbs": 0,
      "fat": 3.6,
      "fiber": 0
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "meal_score": 88.5,
  "meal_rating": "Excellent",
  "meal_emoji": "🌟",
  "foods": [
    {
      "name": "Brown Rice",
      "prediction": "Healthy",
      "health_score": 85.5
    },
    {
      "name": "Chicken Breast",
      "prediction": "Healthy",
      "health_score": 91.5
    }
  ]
}
```

## 🔌 Integrate with React Frontend

### Option 1: Fetch API (Recommended)

Create a service file: `src/services/mlService.js`

```javascript
const ML_API_URL = 'http://localhost:5000/api';

export const predictFood = async (nutritionData) => {
  try {
    const response = await fetch(`${ML_API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(nutritionData),
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('ML API Error:', error);
    throw error;
  }
};

export const analyzeMeal = async (foods) => {
  try {
    const response = await fetch(`${ML_API_URL}/analyze-meal`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ foods }),
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('ML API Error:', error);
    throw error;
  }
};
```

### Option 2: Use in ChatAgent Component

Update `src/pages/ChatAgent.jsx`:

```javascript
import { predictFood } from '../services/mlService';

// Inside your chat handler
const handleFoodAnalysis = async (foodData) => {
  const result = await predictFood({
    calories: foodData.calories,
    protein: foodData.protein,
    carbs: foodData.carbs,
    fat: foodData.fat,
    fiber: foodData.fiber,
  });
  
  // Display result in chat
  addMessage({
    text: `${result.emoji} ${result.recommendation}`,
    sender: 'bot',
    healthScore: result.health_score
  });
};
```

## 🚀 Running Both Servers

You need to run TWO servers simultaneously:

### Terminal 1: React Frontend
```bash
npm run dev
# Runs on http://localhost:5173
```

### Terminal 2: ML API Backend
```bash
cd ml
python api_server.py
# Runs on http://localhost:5000
```

## 🎨 Example Integration Flow

1. **User enters food in chat**: "I ate brown rice and chicken"
2. **Frontend extracts nutrition data** (from database or user input)
3. **Frontend calls ML API**: `POST /api/analyze-meal`
4. **ML model predicts**: Returns health scores
5. **Frontend displays**: Shows recommendations in chat

## 📝 Sample Chat Integration

```javascript
// Example: Analyze user's meal
const userMeal = [
  { name: "Brown Rice", calories: 370, protein: 7.9, carbs: 77.2, fat: 2.9, fiber: 3.5 },
  { name: "Chicken", calories: 165, protein: 31, carbs: 0, fat: 3.6, fiber: 0 },
  { name: "Spinach", calories: 23, protein: 2.9, carbs: 3.6, fat: 0.4, fiber: 2.2 }
];

const result = await analyzeMeal(userMeal);

// Display in chat
console.log(`Meal Score: ${result.meal_score}/100 ${result.meal_emoji}`);
console.log(`Rating: ${result.meal_rating}`);
```

## 🔧 Troubleshooting

### CORS Issues
- Make sure Flask-CORS is installed
- API server includes `CORS(app)` in `api_server.py`

### Connection Refused
- Check if ML API server is running on port 5000
- Verify URL in frontend: `http://localhost:5000/api`

### Model Not Found
- Ensure `food_health_model.joblib` exists in `ml/` folder
- Run `python train_swasthya.py` if missing

## 🎯 Next Steps

1. ✅ ML Model trained and ready
2. ✅ API server created
3. 🔄 Install Flask: `pip install -r requirements_api.txt`
4. 🔄 Start API server: `python api_server.py`
5. 🔄 Create `mlService.js` in React
6. 🔄 Integrate with ChatAgent component
7. 🔄 Test end-to-end flow

---

**Ready to integrate!** 🚀
