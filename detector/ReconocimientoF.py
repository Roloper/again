import cv2
import os
import serial
import time

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
     "haarcascade_frontalface_default.xml")

def face_recognition_with_led():
    ser = serial.Serial('COM2', 9600, timeout=1)
    time.sleep(2)

    dataPath = './detector/Data'
    imagePaths = os.listdir(dataPath)
    print('imagePaths =', imagePaths)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modeloLBPHFace.xml')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    led_state = 'X'  # Estado inicial de los leds (ningún rostro detectado)

    while True:
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
                if result[1] < 60:
                    cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1,
                                cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    led_state = 'R'  # Encender LED azul
                else:
                    cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    led_state = 'N'  # Encender LED rojo

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
    ser.close()
