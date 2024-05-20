import cv2
import mediapipe as mp

# Загрузка обученной модели
model = cv2.face.LBPHFaceRecognizer_create()
model.read('face_recognizer.yml')

# Загрузка классификатора лиц
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Инициализация mediapipe для рук
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Инициализация видео захвата
cap = cv2.VideoCapture(0)

# Ваша фамилия и имя
surname = "Gernichenko"
name = "Elena"

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Не удалось захватить изображение.")
            break

        # Преобразование изображения в серый цвет и RGB
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Обнаружение лиц
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Обнаружение рук
        hand_results = hands.process(image_rgb)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (130, 100))

            # Предсказание лица
            label, confidence = model.predict(face_resize)
            if confidence < 100:
                name_text = "It's You" if label == 0 else "Unknown"
            else:
                name_text = "Unknown"

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, name_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Проверка на наличие рук и подсчет поднятых пальцев
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    fingers_up = 0
                    landmarks = hand_landmarks.landmark
                    for i, tip in enumerate([8, 12, 16, 20]):
                        if landmarks[tip].y < landmarks[tip - 2].y:
                            fingers_up += 1

                    if fingers_up == 1:
                        cv2.putText(image, name, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif fingers_up == 2:
                        cv2.putText(image, surname, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Отображение изображения
        cv2.imshow('Webcam', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
