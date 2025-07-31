import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


print("Loading the feature dataset of the malware samples...")
df = pd.read_csv('/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/File Data for Training/file_features.csv')

print("Dataset loaded successfully. Here's a preview:")
print(df.head())
print("\nDataset Information:")
df.info()

print("\nPreparing data for the model...")

y = df['label']

X = df.drop(['label', 'filename'], axis=1)

print(f"Features (X) shape: {X.shape}")
print(f"Labels (y) shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39)

print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

print("\nTraining the XGBoost model...")

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5)

model.fit(X_train, y_train)

print("Model training complete.")

print("\nEvaluating the model on the test data...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"Specificity: {specificity:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malware (1)']))

model_filename = 'xgboost_malware_model.joblib'
joblib.dump(model, model_filename)
print(f"Model has been successully ran and the results have been saved  to {model_filename} in the same directory that this program was ran in")
