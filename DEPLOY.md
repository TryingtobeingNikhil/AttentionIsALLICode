# NanoGPT Deployment Guide

## Quick Deploy Options

### Option 1: Render (Free tier, no credit card)

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Ensure `input.txt` and `tiny_transformer.pth` are in your repo (add `tiny_transformer.pth` to git if needed)
6. Deploy — your API will be at `https://your-app.onrender.com`

### Option 2: Railway

1. Install CLI: `npm i -g @railway/cli`
2. Run `railway login` and `railway init`
3. Run `railway up` to deploy
4. Or connect GitHub at [railway.app](https://railway.app)

### Option 3: Docker (any cloud)

```bash
docker build -t nanogpt .
docker run -p 8000:8000 -e PORT=8000 nanogpt
```

## Important: Add model file to Git

If `tiny_transformer.pth` is not tracked:

```bash
git add tiny_transformer.pth input.txt
git commit -m "Add model weights and training data"
git push
```

## Test locally

```bash
pip install -r requirements.txt
python main.py
# Or: uvicorn main:app --reload
```

Then:
- Docs: http://localhost:8000/docs
- Generate: `curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "Hello", "max_tokens": 50}'`
