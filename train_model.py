"""
Optional script to train and save the crop recommendation model.
Run this if you want to retrain the model with your own data.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_training_data():
    """Create synthetic training data for crop recommendation."""
    np.random.seed(42)
    n_samples = 2200
    
    data = {
        'nitrogen': np.random.randint(0, 140, n_samples),
        'phosphorus': np.random.randint(5, 145, n_samples),
        'potassium': np.random.randint(5, 205, n_samples),
        'temperature': np.random.uniform(8, 43, n_samples),
        'humidity': np.random.randint(14, 100, n_samples),
        'ph': np.random.uniform(3.5, 9.5, n_samples),
        'rainfall': np.random.uniform(20, 298, n_samples),
    }
    
    crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'coconut', 'cotton', 'coffee',
            'jute', 'sugarcane', 'potato', 'wheat', 'barley']
    
    target = np.random.choice(crops, n_samples)
    
    df = pd.DataFrame(data)
    df['crop'] = target
    
    return df

def train_model():
    """Train Extra Trees Classifier model."""
    
    print("📊 Creating training data...")
    df = create_training_data()
    
    print(f"✅ Dataset created with {len(df)} samples")
    print(f"📋 Features: {df.columns.tolist()}")
    print(f"🌾 Unique crops: {df['crop'].nunique()}")
    
    # Prepare data
    X = df.drop('crop', axis=1)
    y = df['crop']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n📚 Crop classes: {label_encoder.classes_.tolist()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    print(f"\n📈 Training set: {len(X_train)} samples")
    print(f"📉 Test set: {len(X_test)} samples")
    
    # Train model
    print("\n🤖 Training Extra Trees Classifier...")
    model = ExtraTreesClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n📊 Evaluating model...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"✅ Training accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
    print(f"✅ Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
    
    y_pred = model.predict(X_test)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=label_encoder.classes_, 
                               zero_division=0))
    
    # Feature importance
    print("\n📊 Feature Importance:")
    features = X.columns.tolist()
    importance = model.feature_importances_
    
    for feature, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {imp:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    with open('crop_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved as 'crop_model.pkl'")
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✅ Label encoder saved as 'label_encoder.pkl'")
    
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    train_model()