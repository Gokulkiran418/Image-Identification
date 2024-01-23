import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

def predictImage():
    # Load the saved model
    
    #Change file location
    model = load_model('G:\\your\\modelLocation\\model.h5')

    # Helper function to preprocess an image for prediction
    def preprocess_image(image_path):
        img = image.load_img(image_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    image_path = "G:/your/file/path/image.jpg"

    # Preprocess the user-loaded image
    input_image = preprocess_image(image_path)

    # Make a prediction using the loaded model
    predictions = model.predict(input_image)

    # Decode the prediction
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class = np.argmax(predictions)

    # Display the prediction
    print(f"\nPredicted Class: {class_names[predicted_class]}")

predictImage()