# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Drop kolom yang tidak relevan
df.drop(columns=[
    'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours',
    'DailyRate', 'HourlyRate', 'MonthlyRate'
], inplace=True)

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split fitur dan label
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Simpan kolom untuk prediksi
joblib.dump(X.columns.tolist(), 'training_columns.pkl')

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')

# Evaluasi
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
