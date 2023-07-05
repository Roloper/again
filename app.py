import serial
from flask import Flask
from flask import render_template
from flask import Response
import cv2
import os
from detector.entrenandoRF import train_face_recognizer
from flask import request
import config

app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
     "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
ser = serial.Serial('COM2', 9600, timeout=1)
dataPath = './Data'
imagePaths = os.listdir(dataPath)

def generate():
     while True:
          ret, frame = cap.read()
          if ret:
               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               auxFrame = gray.copy()
               faces = face_detector.detectMultiScale(gray, 1.3, 5)
               if len(faces) == 0:
                    led_state = 'X'  # No se detectó ningún rostro, apagar ambos LEDs
                    ser.write(led_state.encode())
               for (x, y, w, h) in faces:
                    rostro = auxFrame[y:y + h, x:x + w]
                    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                    result = face_recognizer.predict(rostro)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
               if imagePaths and isinstance(result[0], int) and 0 <= result[0] < len(imagePaths):
                    cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1,
                                cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
               else:
                    cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

               (flag, encodedImage) = cv2.imencode(".jpg", frame)
               if not flag:
                    continue
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')

def register(personName):
     dataPath = './Data'
     personPath = dataPath + '/' + personName
     if not os.path.exists(personPath):
          print('Carpeta creada:', personPath)
          os.makedirs(personPath)

     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

     faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
     count = 0
     while True:
          ret, frame = cap.read()
          if ret:
               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               auxFrame = frame.copy()
               faces = face_detector.detectMultiScale(gray, 1.3, 5)
               for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    rostro = auxFrame[y:y + h, x:x + w]
                    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
                    count = count + 1
               (flag, encodedImage) = cv2.imencode(".jpg", frame)
               if not flag:
                    continue
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')
               k = cv2.waitKey(1)
               if k == 27 or count >= 300:
                    break
     train_face_recognizer()
     render_template("index.html")

@app.route("/")
def index():
     return render_template("index.html")


@app.route("/registrar")
def registrar():
     return render_template("registrar.html")

@app.route("/registro", methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        nombre = request.form.get('nombre')
        # Hacer algo con el nombre recibido, como guardarlo en la base de datos o procesarlo
        return Response(register(nombre),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    else:
        return render_template("registro.html")

@app.route("/video_reg")
def video_reg():
     return Response(register(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed")
def video_feed():
     return Response(generate(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
     app.run()
cap.release()