"""
Copy files from collab_resutl folder to appropriate project directories
"""
import shutil
import os

print("\n" + "="*70)
print("  COPYING FILES FROM collab_resutl TO PROJECT")
print("="*70 + "\n")

# Define source and destination
source_dir = "collab_resutl"
files_to_copy = [
    ("best_pneumonia_model.h5", "models/best_pneumonia_model.h5"),
    ("pneumonia_model.h5", "models/pneumonia_model.h5"),
    ("training_history.png", "results/training_history.png"),
    ("confusion_matrix.png", "results/confusion_matrix.png"),
    ("roc_curve.png", "results/roc_curve.png"),
    ("precision_recall_curve.png", "results/precision_recall_curve.png"),
    ("sample_images.png", "results/sample_images.png"),
    ("fine_tuning_history.png", "results/fine_tuning_history.png"),
    ("metrics.json", "results/metrics.json"),
]

copied = 0
for src_file, dst_file in files_to_copy:
    src_path = os.path.join(source_dir, src_file)
    
    if os.path.exists(src_path):
        # Create destination directory if needed
        dst_dir = os.path.dirname(dst_file)
        os.makedirs(dst_dir, exist_ok=True)
        
        # Copy file
        shutil.copy2(src_path, dst_file)
        file_size_mb = os.path.getsize(dst_file) / (1024 * 1024)
        print(f"✅ Copied: {src_file} → {dst_file} ({file_size_mb:.2f} MB)")
        copied += 1
    else:
        print(f"⚠️  Not found: {src_file}")

print("\n" + "="*70)
print(f"  COMPLETE - Copied {copied}/{len(files_to_copy)} files")
print("="*70 + "\n")

# Check if model is ready
if os.path.exists("models/best_pneumonia_model.h5"):
    print("🎉 Model is ready to use!")
    print("\n💡 Next steps:")
    print("   1. Start backend: python app.py")
    print("   2. Start frontend: cd frontend && npm run dev")
    print("   3. Open browser: http://localhost:3000")
else:
    print("⚠️  Model not found. Please check the files.")
