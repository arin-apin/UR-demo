import os
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from inference_classification import perform_inference
import inference_streaming 

app = Flask(__name__)

# Variable global para almacenar la ruta de la imagen subida
imagen_path = ''

@app.route('/')
def root():
    return send_from_directory('static', 'index.html')

@app.route('/subir-imagen', methods=['POST'])
def subir_imagen():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se ha proporcionado ninguna imagen.'}), 400
    
    imagen = request.files['imagen']
    global imagen_path
    imagen_path = os.path.join('uploads', imagen.filename)
    imagen.save(imagen_path)
    
    return jsonify({'message': 'Imagen subida correctamente.'}), 200

@app.route('/inferencia', methods=['POST'])
def inferencia():
    global imagen_path
    
    if imagen_path == '':
        return jsonify({'error': 'No se ha subido ninguna imagen.'}), 400
    
    resultado = perform_inference(imagen_path)
    
    return jsonify({'resultado': resultado}), 200

@app.route('/inferencia-stream', methods=['POST'])
def inferencia_stream():
    
    inference_streaming.capture_frames_and_inference()
      
    resultado = inference_streaming.perform_inference_stream()
    
    return jsonify({'resultado': resultado}), 200

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run()
