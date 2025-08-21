import pefile
import os
import math
import pandas as pd


malware_parent_dir = '/home/damian/Dissertation_Work/training_samples/malicious_static'
benign_parent_dir = '/home/damian/Dissertation_Work/training_samples/benign_files/benign_windows'
output_csv_path = '/home/damian/Dissertation_Work/GitHub_Commits/XAI_Dissertation/File Data for Training/file_features.csv'


def get_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(x.to_bytes(1, 'little'))) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def extract_features(file_path):
    features = {}

    try:
        pe = pefile.PE(file_path)
    except pefile.PEFormatError:
        return None

    features['filename'] = os.path.basename(file_path)
    features['filesize'] = os.path.getsize(file_path)
    features['e_magic'] = pe.DOS_HEADER.e_magic
    features['e_lfanew'] = pe.DOS_HEADER.e_lfanew
    features['Machine'] = pe.FILE_HEADER.Machine
    features['NumberOfSections'] = pe.FILE_HEADER.NumberOfSections
    features['TimeDateStamp'] = pe.FILE_HEADER.TimeDateStamp
    features['Characteristics'] = pe.FILE_HEADER.Characteristics
    features['AddressOfEntryPoint'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    features['ImageBase'] = pe.OPTIONAL_HEADER.ImageBase
    features['SectionAlignment'] = pe.OPTIONAL_HEADER.SectionAlignment
    features['FileAlignment'] = pe.OPTIONAL_HEADER.FileAlignment
    features['MajorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
    features['MajorSubsystemVersion'] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
    features['Subsystem'] = pe.OPTIONAL_HEADER.Subsystem
    features['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
    features['SizeOfHeaders'] = pe.OPTIONAL_HEADER.SizeOfHeaders
    features['CheckSum'] = pe.OPTIONAL_HEADER.CheckSum
    features['NumberOfRvaAndSizes'] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes

    section_entropies = []
    section_sizes = []
    if hasattr(pe, 'sections'):
        for section in pe.sections:
            section_entropies.append(get_entropy(section.get_data()))
            section_sizes.append(section.SizeOfRawData)

    features['sections_mean_entropy'] = sum(section_entropies) / len(section_entropies) if section_entropies else 0
    features['sections_min_entropy'] = min(section_entropies) if section_entropies else 0
    features['sections_max_entropy'] = max(section_entropies) if section_entropies else 0
    features['sections_mean_size'] = sum(section_sizes) / len(section_sizes) if section_sizes else 0

    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        features['imports'] = len(pe.DIRECTORY_ENTRY_IMPORT)
        imported_functions = 0
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            imported_functions += len(entry.imports)
        features['imported_functions'] = imported_functions
    else:
        features['imports'] = 0
        features['imported_functions'] = 0

    return features

if __name__ == "__main__":
    all_features = []
    labels = []



    print(f"Processing directories in: {malware_parent_dir}")
    for family_folder in os.listdir(malware_parent_dir):
        malware_dir = os.path.join(malware_parent_dir, family_folder, 'unzipped')
        if not os.path.isdir(malware_dir):
            continue
        print(f" -> Processing family: {family_folder}")
        for root, _, files in os.walk(malware_dir):
            for file in files:
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                if features:
                    all_features.append(features)
                    labels.append(1) #

    print(f"\nProcessing benign directories in: {benign_parent_dir}")
    for benign_folder in os.listdir(benign_parent_dir):
        benign_dir = os.path.join(benign_parent_dir, benign_folder)
        if not os.path.isdir(benign_dir):
            continue
        print(f" -> Processing source: {benign_folder}")
        for root, _, files in os.walk(benign_dir):
            for file in files:
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                if features:
                    all_features.append(features)
                    labels.append(0)

    print(f"Successfully processed {len(all_features)} files.")

    df = pd.DataFrame(all_features)
    df['label'] = labels

    df.fillna(0, inplace=True)

    df.to_csv(output_csv_path, index=False)

    print(f"Feature dataset saved to {output_csv_path}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
