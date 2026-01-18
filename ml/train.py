import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_data():
    """
    Load datasets. 
    Assumes files are in ml/data/
    """
    paths = {
        'world_facts': 'data/en.openfoodfacts.org.products.csv',
        'nutrition': 'data/nutrition_dataset.csv'
    }
    
    dfs = {}
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"Loading {name}...")
            # Using low_memory=False because OpenFoodFacts is huge and has mixed types
            dfs[name] = pd.read_csv(path, sep='\t' if 'openfoodfacts' in path else ',', low_memory=False)
        else:
            print(f"Warning: {path} not found.")
            
    return dfs

def preprocess_data(dfs):
    """
    Clean and merge data for training.
    Can handle both OpenFoodFacts and the Health Prediction dataset.
    """
    X_list = []
    y_list = []

    # 1. Process World Food Facts (if available)
    if 'world_facts' in dfs:
        print("Processing World Food Facts...")
        df1 = dfs['world_facts']
        features1 = [
            'energy_100g', 'fat_100g', 'saturated-fat_100g', 
            'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 
            'proteins_100g', 'salt_100g'
        ]
        target1 = 'nutrition_grade_fr'
        
        cols = [c for c in features1 if c in df1.columns]
        if target1 in df1.columns:
            temp_df = df1[cols + [target1]].dropna()
            grade_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
            temp_df['target'] = temp_df[target1].str.lower().map(grade_map)
            temp_df = temp_df.dropna(subset=['target'])
            
            X_list.append(temp_df[cols])
            y_list.append(temp_df['target'])

    # 2. Process Nutrition Dataset (if available)
    if 'nutrition' in dfs:
        print("Processing Nutrition Dataset...")
        df2 = dfs['nutrition']
        # Looking for standard nutrition columns (some might be capitalized)
        # Assuming columns: 'calories', 'fat', 'carbs', 'protein', 'fiber', 'label'
        # Or similar based on search results
        mapping = {
            'calories': 'energy_100g',
            'fat': 'fat_100g',
            'carbs': 'carbohydrates_100g',
            'protein': 'proteins_100g',
            'fiber': 'fiber_100g'
        }
        # Find available columns and rename to match df1
        rename_map = {}
        for col in df2.columns:
            for k, v in mapping.items():
                if k in col.lower():
                    rename_map[col] = v
        
        if rename_map:
            df2 = df2.rename(columns=rename_map)
            # Find target (usually a 'healthy' or 'label' column)
            target_cols = [c for c in df2.columns if 'label' in c.lower() or 'healthy' in c.lower() or 'class' in c.lower()]
            if target_cols:
                target2 = target_cols[0]
                cols2 = list(rename_map.values())
                temp_df = df2[cols2 + [target2]].dropna()
                # If target is binary (Healthy/Unhealthy), map to 0 and 4 (A and E) for consistency
                if temp_df[target2].dtype == 'object':
                    temp_df['target'] = temp_df[target2].apply(lambda x: 0 if 'health' in str(x).lower() else 4)
                else:
                    temp_df['target'] = temp_df[target2]
                
                X_list.append(temp_df[cols2])
                y_list.append(temp_df['target'])

    if not X_list:
        return None
    
    # Simple strategy: use the most complete dataset or concat if features match
    # For simplicity here, we'll just handle the first one that successfully processed
    # Real world: you'd want to align schemas perfectly
    X = pd.concat(X_list, ignore_index=True, sort=False).fillna(0)
    y = pd.concat(y_list, ignore_index=True)
    
    return X, y

def train_model(X, y):
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    # Create data dir if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Please place your datasets in 'ml/data/'")
    
    data_frames = load_data()
    
    if data_frames:
        processed = preprocess_data(data_frames)
        if processed:
            X, y = processed
            model = train_model(X, y)
            
            # Save the model
            joblib.dump(model, 'food_model.joblib')
            print("Model saved to 'food_model.joblib'")
        else:
            print("No data to process.")
    else:
        print("No datasets loaded. Please check the README for instructions.")
