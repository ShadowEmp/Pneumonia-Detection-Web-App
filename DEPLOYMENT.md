# Deployment Guide

## 🚀 Recommended: Hugging Face Spaces (Backend)
Render's free tier (512MB RAM) is too small for our AI model. **Hugging Face Spaces** offers **16GB RAM** for free, which is perfect for this project.

### Steps to Deploy Backend:
1.  **Sign Up:** Go to [huggingface.co](https://huggingface.co) and create an account.
2.  **Create Space:**
    *   Click **New Space**.
    *   Name: `pneumonia-detection-api` (or similar).
    *   License: MIT.
    *   **SDK:** Select **Docker**.
    *   Privacy: Public.
    *   Click **Create Space**.
3.  **Upload Code:**
    *   Hugging Face gives you a Git command.
    *   In your local terminal:
        ```bash
        git remote add space https://huggingface.co/spaces/YOUR_USERNAME/pneumonia-detection-api
        git push space main --force
        ```
4.  **Set Environment Variables:**
    *   Go to your Space's **Settings** tab.
    *   Scroll to **Variables and secrets**.
    *   Add New Variable:
        *   Key: `MODEL_URL`
        *   Value: `https://drive.google.com/file/d/1bjb_zGpEb-8Exf-wkZncP93RObP9ie_W/view?usp=sharing`
5.  **Get URL:**
    *   Once built, your API URL will be: `https://your-username-pneumonia-detection-api.hf.space`
    *   **Important:** Use this URL to update your Frontend configuration.

## 2. Frontend (Vercel)
1.  **Update Configuration:**
    *   Open `frontend/vercel.json`.
    *   Replace the `destination` URL with your new **Hugging Face URL**.
    *   Commit and push to GitHub.
2.  **Deploy on Vercel:**
    *   Sign up at [vercel.com](https://vercel.com).
    *   **Add New Project** and import your GitHub repo.
    *   **Root Directory:** Select `frontend`.
    *   **Framework Preset:** Vite.
    *   **Deploy.**

## ⚠️ Important Note: Cold Starts
Hugging Face Spaces (Free Tier) will **pause after 48 hours of inactivity**.
*   **What this means:** The first request after a pause will take **1-2 minutes** to process because the server needs to wake up and re-download the AI model.
*   **Subsequent requests:** Will be instant.
*   **Recommendation:** If the app seems "stuck," just wait a minute for it to wake up.
