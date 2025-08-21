import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


DATASET_PATH = '/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/File_Data_for_Training/file_features.csv'
MODEL_PATH = 'xgboost_malware_model.joblib'

def advanced_shap_analysis():

    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found. Please run the tuning script first.")
        return

    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    y = df['label']
    X = df.drop(['label', 'filename'], axis=1)

    _, X_test, _, _ = train_test_split(X, y, test_size=0.25, random_state=39, stratify=y)
    print(f"Test data prepared with {len(X_test)} samples.")

    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    print("SHAP values calculated.")

    print("\nCalculating SHAP Interaction values ...")
    shap_interaction_values = explainer.shap_interaction_values(X_test)
    print("SHAP Interaction values calculated.")

    print("\nGenerating and saving plots...")

    print("   Saving global summary plot (summary_plot.png)...")
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/Result Data/summary_plot.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("   Saving interaction summary plot (interaction_summary_plot.png)...")
    shap.summary_plot(shap_interaction_values, X_test, show=False)
    plt.savefig('/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/Result Data/interaction_summary_plot.png', bbox_inches='tight', dpi=300)
    plt.close()

    top_feature = X_test.columns[np.abs(shap_values.values).mean(0).argmax()]
    print(f"  Saving dependence plot for top feature: '{top_feature}' (dependence_plot.png)...")

    shap.dependence_plot(
        top_feature,
        shap_values.values,
        X_test,
        show=False,
        interaction_index="auto"
    )
    plt.savefig('/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/Result Data/dependence_plot.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("\nAdvanced SHAP analysis complete. All plots have been saved.")

if __name__ == "__main__":
    advanced_shap_analysis()
