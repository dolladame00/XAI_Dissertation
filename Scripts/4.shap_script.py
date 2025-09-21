#This script loads the trained model and generates all SHAP analysis plots
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


DATASET_PATH = '/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/File_Data_for_Training/file_features.csv'
MODEL_PATH = '/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/File_Data_for_Training/xgboost_malware_model.joblib'


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
    X = df.drop(['label', 'Filename'], axis=1)

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

    print("-----Saving global summary plot (summary_plot.png)...")
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"Global Summary Plot")
    plt.savefig('/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/Result Data/summary_plot.png', bbox_inches='tight', dpi=300)
    plt.close()

    feature_importances = np.abs(shap_values.values).mean(0)
    sorted_indices = np.argsort(feature_importances)
    first_feature_name = X_test.columns[sorted_indices[-1]]
    second_feature_name = X_test.columns[sorted_indices[-2]]

    top_feature = X_test.columns[np.abs(shap_values.values).mean(0).argmax()]
    print(f"-----Saving dependence plot for top feature: '{top_feature}' (dependence_plot.png)...")

    shap.dependence_plot(
        top_feature,
        shap_values.values,
        X_test,
        show=False,
        interaction_index="auto"
    )
    plt.title(f"SHAP Dependence Plot for {first_feature_name}")
    plt.savefig('/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/Result Data/dependence_plot.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"-----Saving zoomed view of dependence plot, excluding top 22 outliers...")

    checksum_values = X_test[top_feature].sort_values(ascending=False)
    threshold = checksum_values.iloc[22]

    mask = X_test[top_feature] <= threshold

    X_test_filtered = X_test[mask]
    shap_values_filtered = shap_values[mask.values]

    shap.dependence_plot(
        top_feature,
        shap_values_filtered.values,
        X_test_filtered,
        show=False,
        interaction_index="auto"
    )
    plt.title(f"SHAP Dependence Plot for {first_feature_name} excluding the top 22 outliers ")
    plt.savefig('/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/Result Data/dependence_plot_zoomed.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"-----Saving SUPER ZOOMED view of dependence plot, excluding top 65 outliers...")

    checksum_values = X_test[top_feature].sort_values(ascending=False)
    threshold = checksum_values.iloc[65]

    mask = X_test[top_feature] <= threshold

    X_test_filtered = X_test[mask]
    shap_values_filtered = shap_values[mask.values]

    shap.dependence_plot(
        top_feature,
        shap_values_filtered.values,
        X_test_filtered,
        show=False,
        interaction_index="auto"
    )
    plt.title(f"SHAP Dependence Plot for {first_feature_name} excluding the top 65 outliers ")
    plt.savefig('/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/Result Data/dependence_plot_super_zoomed.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"-----Saving dependence plot for second top feature: '{second_feature_name}'...")
    shap.dependence_plot(
        second_feature_name,
        shap_values.values,
        X_test,
        show=False,
        interaction_index="auto"
    )
    plt.title(f"SHAP Dependence Plot for {second_feature_name}")
    plt.savefig(f'/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/Result Data/dependence_plot_{second_feature_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("\nAdvanced SHAP analysis complete. All plots have been saved.")

if __name__ == "__main__":
    advanced_shap_analysis()
