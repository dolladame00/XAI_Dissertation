Shedding Light on the Black Box: Explainable AI for Ransomware Detection
This repository contains the full source code and supplementary materials for my MSc Computer Science for Cybersecurity dissertation, completed at Oxford Brookes University.

The project focuses on addressing the "black box" problem in machine learning-based malware detection. It demonstrates that by integrating a state-of-the-art model (XGBoost) with a leading-edge eXplainable AI (XAI) framework (SHAP), it is possible to create a ransomware detector that is both highly accurate and fully transparent.

Repository Contents
-/ (Root Directory)
    -/Scripts/
        -1.unzip_7z.py (Unzips password-protected malware archives)
        -2.feature_extraction.py (Parses PE files and extracts static features into a CSV)
        -3.training_of_files.py (Trains the XGBoost classifier and saves the final model)
        -4.shap_script.py (Loads the trained model and generates all SHAP analysis plots)
        -5.convert_shap_to_csv.py (Exports a detailed CSV with test data, predictions, and SHAP values for granular analysis)
    -/Result Data/
        -shap_complete_results.csv (The detailed output file used for the case study of misclassifications)
        -summary_plot.png
        -dependence_plot.png
        -dependence_plot_zoomed.png
        -dependence_plot_super_zoomed.png	
        -dependence_plot_MajorOperatingSystemVersion.png
    -/File_Data_for_Training/
        -file_features.csv (The complete dataset of 24 extracted static features from the 1,300 PE files)
        -xgboost_malware_model.joblib (The final, trained XGBoost model)
    -README.md
    -data_manifest.csv

To run and reproduce these results, a python virtual environment has to be created first, with the commands:
“python3 -m venv venv”
“source venv/bin/activate”. 

The actual PE files used in the training and testing phases of this dissertation were not included in this repository, due to potential dangers associated with it. However, this repository has a data_manifest.csv file, which documents each PE file that was used in this experiment, including its SHA256 hash, its source, and if it is benign (0) or malicious (1). 
