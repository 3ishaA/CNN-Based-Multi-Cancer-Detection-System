# CNN-Based-Multi-Cancer-Detection-System
import cv2
import numpy as np
import time
from datetime import datetime
from ai_edge_litert.interpreter import Interpreter
from picamera2 import Picamera2
from gpiozero import Button, LED, Buzzer

button = Button(17)

green_led = LED(22)
yellow_led = LED(27)
red_led = LED(23)

buzzer = Buzzer(18)

interpreter = Interpreter(model_path="multi_cancer_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [
    "ALL",
    "Brain Cancer",
    "Breast Cancer",
    "Cervical Cancer",
    "Kidney Cancer",
    "Lung and Colon Cancer",
    "Lymphoma",
    "Oral Cancer"
]

def update_leds(confidence):
    green_led.off()
    yellow_led.off()
    red_led.off()

    if confidence >= 90:
        green_led.on()
    elif confidence >= 70:
        yellow_led.on()
    else:
        red_led.on()

def beep():
    buzzer.on()
    time.sleep(0.2)
    buzzer.off()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

output = np.zeros(len(class_names))
top_indices = [0, 0, 0]
confidence = 0
predicted_class = "Waiting for Scan"
status = "Press Button to Scan"
color = (255, 255, 255)

confidence_history = []

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    if button.is_pressed:
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_index = np.argmax(output)
        confidence = output[pred_index] * 100
        predicted_class = class_names[pred_index]

        top_indices = output.argsort()[-3:][::-1]

        confidence_history.append(confidence)
        if len(confidence_history) > 10:
            confidence_history.pop(0)

        if confidence >= 90:
            color = (0, 255, 0)
            status = "High Confidence"
        elif confidence >= 70:
            color = (0, 255, 255)
            status = "Medium Confidence"
        else:
            color = (0, 0, 255)
            status = "Low Confidence"

        update_leds(confidence)
        beep()

        filename = datetime.now().strftime("scan_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(filename, frame)

        time.sleep(0.5)

    text = f"{predicted_class}: {confidence:.2f}%"

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, status, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    bar_x, bar_y = 20, 110
    bar_width = 300
    filled_width = int((confidence / 100) * bar_width)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + 25),
                  (255, 255, 255), 2)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + filled_width, bar_y + 25),
                  color, -1)

    cv2.putText(frame, "Top 3 Predictions:", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    y = 205
    for i, index in enumerate(top_indices):
        class_name = class_names[index]
        score = output[index] * 100

        cv2.putText(frame, f"{i+1}. {class_name}: {score:.2f}%",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)
        y += 30

    cv2.putText(frame, "Educational AI Prototype - Not for Medical Diagnosis",
                (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2)

    graph_x, graph_y = 350, 80
    cv2.putText(frame, "Confidence History",
                (graph_x, graph_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1)

    for i in range(1, len(confidence_history)):
        x1 = graph_x + (i - 1) * 20
        y1 = graph_y + 100 - int(confidence_history[i - 1])

        x2 = graph_x + i * 20
        y2 = graph_y + 100 - int(confidence_history[i])

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("AI CANCER Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()

green_led.off()
yellow_led.off()
red_led.off()
buzzer.off()
