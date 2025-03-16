from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from waitress import serve
import base64

# Configura TensorFlow per non mostrare avvisi inutili
tf.get_logger().setLevel('ERROR')

# Inizializza l'app Flask
app = Flask(__name__)

# Carica il modello
model = load_model('final_model.h5')

# Funzione per preparare l'immagine
def prepare_image(image):
    if not isinstance(image, Image.Image):
        image = Image.open(BytesIO(image))

    # Converti in RGB e ridimensiona
    image = image.convert('RGB')
    img = image.resize((28, 28))
    img = img_to_array(img)

    # Scala di grigi
    img_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    img_gray = np.expand_dims(img_gray, axis=-1)

    # Inverti i colori se necessario
    if np.mean(img_gray) > 127:
        img_gray = 255 - img_gray

    # Normalizza
    img_gray = img_gray.reshape(1, 28, 28, 1).astype('float32') / 255.0

    return img_gray

# Endpoint GET per testare se il server è attivo
@app.route('/', methods=['GET'])
def index():
    return "Server Flask attivo. Usa un POST per inviare le immagini."

# Endpoint POST per ricevere le immagini e fare predizioni
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'Nessun dato immagine fornito'}), 400

    image_data = data['image']
    try:
        # Decodifica la stringa base64 in un'immagine
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        img_array = prepare_image(image)

        # Predizione
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        return jsonify({
            'prediction': int(predicted_class),
            'probabilities': prediction.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Avvia il server Flask
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server Flask in ascolto su http://localhost:{port}/")
    serve(app, host="0.0.0.0", port=port)
