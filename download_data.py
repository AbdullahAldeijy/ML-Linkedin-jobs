"""
Download script for large files not included in the repository.
Run this script after cloning to get the dataset and trained models.
"""

import os
import urllib.request

def download_from_google_drive(file_id, filename):
    """Download a file from Google Drive"""
    print(f"Downloading {filename}...")
    try:
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        urllib.request.urlretrieve(url, filename)
        print(f"✅ {filename} downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        print(f"Please download manually from: https://drive.google.com/file/d/{file_id}/view")

def main():
    print("📥 Downloading large files...")
    
    # Note: You'll need to upload these files to a cloud storage service
    # and update the URLs below
    
    # Google Drive file IDs
    files_to_download = {
        "data_jobs.csv": "1sYGST9utO45CJtUIkJEFFhVF-1DZPM4W"
    }
    
    for filename, file_id in files_to_download.items():
        if not os.path.exists(filename):
            download_from_google_drive(file_id, filename)
        else:
            print(f"✅ {filename} already exists")
    
    # Note about model files
    model_files = ["Random_Forest_level_model.pkl", "scaler_jobs_level.pkl"]
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"ℹ️  {model_file} will be generated when you run the ML pipeline")
    
    print("\n🎉 Setup complete! You can now run the ML pipeline.")

if __name__ == "__main__":
    main()