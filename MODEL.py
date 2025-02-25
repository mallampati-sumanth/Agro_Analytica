import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Load dataset
dataset = pd.read_csv(r'CROPDATA.csv')

# Encode target labels
le = LabelEncoder()
dataset['Crop_Type'] = le.fit_transform(dataset['Crop_Type'])

# Features and target
X = dataset.drop('Crop_Type', axis=1)
y = dataset['Crop_Type']

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Random Forest Classifier with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
rf_best = grid_search.best_estimator_

# Train the model
rf_best.fit(X_train, y_train)

# Evaluate model
y_pred = rf_best.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display crop mapping
crop_mapping = dict(zip(le.transform(le.classes_), le.classes_))
print("Crop Type Mapping:")
for number, crop in crop_mapping.items():
    print(f"{number}: {crop}")

# Save the trained model, label encoder, and scaler
joblib.dump(rf_best, 'crop_predictionmodel.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model, Label Encoder, and Scaler saved successfully.")
