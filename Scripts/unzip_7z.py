import os
import subprocess

source_directory = '/home/damian/Dissertation_Work/training_samples/malicious_static/LockBit.PE'
target_directory = '/home/damian/Dissertation_Work/training_samples/malicious_static/LockBit.PE/unzipped_LockBit_PE'
password = 'infected'

def unzip_all_files():
    print("Starting the unzipping process...")

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created target directory: {target_directory}")

    zip_files = [f for f in os.listdir(source_directory) if f.endswith('.zip')]
    print(f"Found {len(zip_files)} zip files to process.")

    successful = 0
    failed = 0

    for zip_file in zip_files:
        zip_path = os.path.join(source_directory, zip_file)
        print(f"  -> Unzipping {zip_file}...")

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
            print(f"     -> ERROR: Failed to unzip {zip_file}")
            print(result.stderr.decode())
            failed += 1

    print("\n--- Unzipping Complete ---")
    print(f"Successfully unzipped: {successful}")
    print(f"Failed to unzip: {failed}")
    print(f"Files saved in: {target_directory}")

if __name__ == "__main__":
    unzip_all_files()
