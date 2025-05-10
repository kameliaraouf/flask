# Skin Disease Classification API

## How it works
1. Upload an image.
2. The first model classifies it as "skin" or "not skin".
3. If it is skin, it is passed to a second model to diagnose the disease.

## How to run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run API
```bash
uvicorn main:app --reload
```

### Endpoint
- `POST /predict` - upload an image file.