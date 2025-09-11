import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split

DATASET_PATH = '/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/File_Data_for_Training/file_features.csv'
MODEL_PATH = '/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/File_Data_for_Training/xgboost_malware_model.joblib'

def export_complete_shap_results():

    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATASET_PATH)

    y = df['label']
    X = df.drop(['label', 'Filename'], axis=1)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=39, stratify=y)

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap_df = pd.DataFrame(shap_values.values, columns=[f"{col}_shap" for col in X_test.columns])

    X_test_reset = X_test.reset_index(drop=True)

    combined_df = pd.concat([X_test_reset, shap_df], axis=1)

    print("Adding model predictions to the results...")
    predictions = model.predict(X_test)

    combined_df['true_label'] = y_test.reset_index(drop=True)
    combined_df['predicted_label'] = predictions

    combined_df['is_correct'] = (combined_df['true_label'] == combined_df['predicted_label'])

    output_path = '/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/Result Data/shap_complete_results.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\nAnalysis complete. Complete results exported to {output_path}")

if __name__ == "__main__":
    export_complete_shap_results()
