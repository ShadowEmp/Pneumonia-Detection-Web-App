import os
import sys
import gdown

# URL of your model file (Direct download link)
# We use the environment variable 'MODEL_URL', or fall back to the hardcoded Google Drive link
DEFAULT_URL = 'https://drive.google.com/uc?id=1bjb_zGpEb-8Exf-wkZncP93RObP9ie_W'
MODEL_URL = os.environ.get('MODEL_URL', DEFAULT_URL)
MODEL_PATH = 'models/best_pneumonia_model.h5'

def download_model():
    if os.path.exists(MODEL_PATH):
        # Optional: Check file size to ensure it's not a corrupted 2KB HTML file
        if os.path.getsize(MODEL_PATH) > 1000000: # > 1MB
            print(f"✅ Model already exists at {MODEL_PATH}")
            return
        else:
            print(f"⚠️ Found corrupted/small model file. Re-downloading...")
            os.remove(MODEL_PATH)

    print(f"⬇️ Downloading model using gdown...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Download using gdown (handles Google Drive warnings automatically)
        # Extract ID from URL if possible, or use URL directly
        output = gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
        
        if output:
            print(f"✅ Model downloaded successfully to {MODEL_PATH}")
        else:
            print("❌ Download failed (no output file)")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Failed to download model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
