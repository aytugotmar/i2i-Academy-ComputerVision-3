import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Yeni Nesil MediaPipe Yapılandırması (Beyin Dosyasını Yüklüyoruz)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

# num_hands=2 yaparak iki eli birden saymasını sağlıyoruz
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
tip_ids = [8, 12, 16, 20]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Kamerayı ayna moduna çevir
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Yapay zekaya görüntüyü gönder
    detection_result = detector.detect(mp_image)
    
    total_fingers = 0

    if detection_result.hand_landmarks:
        # Ekranda birden fazla el olabilir, hepsini tek tek döngüye sokuyoruz
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            
            # Elin sağ mı sol mu olduğunu anlıyoruz
            hand_label = detection_result.handedness[idx][0].category_name
            fingers = []

            # ---- BAŞPARMAK KONTROLÜ (DÜZELTİLDİ) ----
            if hand_label == "Left": 
                # Gerçekte Sağ Elimiz: Ekranda başparmak sola bakar (X daha KÜÇÜKTÜR)
                if hand_landmarks[4].x < hand_landmarks[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else: 
                # Gerçekte Sol Elimiz: Ekranda başparmak sağa bakar (X daha BÜYÜKTÜR)
                if hand_landmarks[4].x > hand_landmarks[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # ---- DİĞER 4 PARMAK KONTROLÜ ----
            for tip in tip_ids:
                if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers += fingers.count(1)
            
            # Ekrana kırmızı eklem noktalarını çiz
            for landmark in hand_landmarks:
                 x = int(landmark.x * frame.shape[1])
                 y = int(landmark.y * frame.shape[0])
                 cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Toplam sayıyı ekrana yazdır
    cv2.putText(frame, f'Fingers: {total_fingers}', (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    cv2.imshow("i2i Academy - Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()