"""
Download script for large files not included in the repository.
Run this script after cloning to get the dataset and trained models.
"""

import os
import urllib.request

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ {filename} downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")

def main():
    print("📥 Downloading large files...")
    
    # Note: You'll need to upload these files to a cloud storage service
    # and update the URLs below
    
    files_to_download = {
        "data_jobs.csv": "# Upload to Google Drive/Dropbox and add URL here",
        "Random_Forest_level_model.pkl": "# Upload to Google Drive/Dropbox and add URL here",
        "scaler_jobs_level.pkl": "# Upload to Google Drive/Dropbox and add URL here"
    }
    
    for filename, url in files_to_download.items():
        if not os.path.exists(filename):
            if url.startswith("http"):
                download_file(url, filename)
            else:
                print(f"⚠️  Please upload {filename} to cloud storage and update the URL in this script")
        else:
            print(f"✅ {filename} already exists")
    
    print("\n🎉 Setup complete! You can now run the ML pipeline.")

if __name__ == "__main__":
    main()