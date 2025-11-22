import os
import requests
import sys

# URL of your model file (Direct download link)
# REPLACE THIS with your actual direct download link (Dropbox, Google Drive, etc.)
MODEL_URL = os.environ.get('https://drive.google.com/file/d/1bjb_zGpEb-8Exf-wkZncP93RObP9ie_W/view?usp=drive_link')
MODEL_PATH = 'models/best_pneumonia_model.h5'

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists at {MODEL_PATH}")
        return

    if not MODEL_URL:
        print("❌ Error: MODEL_URL environment variable is not set.")
        print("   Please set MODEL_URL in your Render dashboard.")
        sys.exit(1)

    print(f"⬇️ Downloading model from {MODEL_URL}...")
    
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"✅ Model downloaded successfully to {MODEL_PATH}")
        
    except Exception as e:
        print(f"❌ Failed to download model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
