#!/bin/bash

# Install required Python packages
pip install -r requirements.txt

# Run the FastAPI app
uvicorn main:app --host 0.0.0.0 --port 8000
