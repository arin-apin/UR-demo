import speech_recognition as sr
import spacy
import os
import sys

#Introducir en terminal para evitar mensajes de error con ALSA (funciona pero no reconoce los comandos)
#python speech-recognition.py 2> /dev/null


# Redirigir la salida de errores a un archivo
error_file = open('errors.txt', 'w')
sys.stderr = error_file

# Desactivar los mensajes de error
os.environ['PYAUDIO_NO_WARN_INPUT_OVERFLOW'] = '1'


# Cargar el modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# Función para reconocer comandos de voz
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Di algo...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio, language="es-ES")
        print("Has dicho: " + text)
        process_command(text)
    except sr.UnknownValueError:
        print("No se pudo reconocer el audio")
    except sr.RequestError as e:
        print("Error al obtener los resultados del servicio de reconocimiento de voz: {0}".format(e))

# Función para procesar los comandos reconocidos
def process_command(command):

    
    if "hola" in command:
        print("¡Hola! ¿Cómo puedo ayudarte?")
    elif "activa cámara" in command:
        print("Iniciando camara...")
        
    elif "iniciar inferencia" in command:
        print("Iniciando inferencia...")
        
    else:
        print("Comando no reconocido")

# Función para saludar a una persona específica
# def greet_person(person_name):
#     print("¡Hola, " + person_name + "! ¿En qué puedo ayudarte?")

 # Función para navegar hacia una ubicación específica
# def navigate_location(location_name):
#     print("Navegando hacia " + location_name + "...")


# Ejecución principal del programa
while True:
    recognize_speech()
