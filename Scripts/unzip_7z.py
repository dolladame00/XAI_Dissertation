import os
import subprocess

base_directory = '/home/damian/Dissertation_Work/training_samples/malicious_static'
password = 'infected'
number_of_dir_scanned='0'

def unzip_all_files():

    print(f"Starting the unzipping process in: {base_directory}\n")

    for folder_name in os.listdir(base_directory):
        source_directory = os.path.join(base_directory, folder_name)

        if not os.path.isdir(source_directory):
            continue

        print(f"--- Processing folder: {folder_name} ---")

        target_directory = os.path.join(source_directory, 'unzipped')

        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        zip_files = [f for f in os.listdir(source_directory) if f.endswith('.zip')]

        if not zip_files:
            print("No .zip files found in this folder. Skipping.\n")
            continue

        print(f"Found {len(zip_files)} zip files to process.")

        successful = 0
        failed = 0

        for zip_file in zip_files:
            zip_path = os.path.join(source_directory, zip_file)

            cmd = [
                '7z', 'x', zip_path,
                f'-o{target_directory}',
                f'-p{password}',
                '-y'
            ]

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode == 0:
                successful += 1
            else:
                print(f"  -> ERROR: Failed to unzip {zip_file}")
                print(result.stderr.decode().strip()) # .strip() cleans up extra lines
                failed += 1

        print(f"Finished. Success: {successful}, Failed: {failed}")
        print(f"Files saved in: {target_directory}\n")

if __name__ == "__main__":
    unzip_all_files()
