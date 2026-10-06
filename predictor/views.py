from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import os
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model

# Load the model once when the server starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'malaria_mobilenetv2_model.keras')
model = load_model(MODEL_PATH)

# Class mappings matching the generator indices
IMAGE_SIZE = 128
CLASS_LABELS = ['Parasitized', 'Uninfected']

# Create your views here.
def index(request):
    predicted_class = None
    image_url = None
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image
        uploaded_image = request.FILES['image']
        
        # Save the image to a temporary location
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        image_path = fs.path(filename)
        image_url = fs.url(filename)

        try:
            # Preprocess the image
            img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = CLASS_LABELS[np.argmax(predictions[0])]
            print(f"Predicted class index: {np.argmax(predictions[0])}")
        except Exception as e:
            print(f"Error occurred: {e}")
            # Clean up the temporary file
            os.remove(image_path)

    return render(request, 'predictor/index.html', {'predicted_class': predicted_class, 'image_url': image_url})