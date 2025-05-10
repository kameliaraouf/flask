
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import tensorflow as tf
import torchvision.transforms as transforms

app = FastAPI()

# Load models
skin_model = tf.keras.models.load_model("models/skin_classification_model.h5")
disease_model = torch.load("models/my_best_model.pth", map_location=torch.device('cpu'))
disease_model.eval()

# Preprocessing
def preprocess_for_skin(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0
    return img_array

def preprocess_for_disease(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)

# Replace with your actual classes
disease_classes = ["eczema", "psoriasis", "acne", "fungal infection", "vitiligo", "other"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Step 1: Skin detection
    skin_input = preprocess_for_skin(img)
    skin_pred = skin_model.predict(skin_input)[0][0]
    is_skin = skin_pred > 0.5

    if not is_skin:
        return JSONResponse(content={"is_skin": False, "diagnosis": None})

    # Step 2: Disease prediction
    disease_input = preprocess_for_disease(img)
    with torch.no_grad():
        output = disease_model(disease_input)
        pred_class = torch.argmax(output, 1).item()
        diagnosis = disease_classes[pred_class]

    return JSONResponse(content={"is_skin": True, "diagnosis": diagnosis})
