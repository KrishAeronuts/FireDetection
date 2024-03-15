from io import BytesIO
from flask import Flask, request,send_file
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
model = YOLO('best.pt')

@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    frame = request.files['frame'].read()
    nparr = np.fromstring(frame, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img, device="cpu", conf=0.4)
    annotated_frame = results[0].plot(conf=False)

    cv2.imwrite('result.jpg', annotated_frame)

    return 'Frame received and processed successfully!'

@app.route('/send_frame', methods=['POST'])
def send_frame():
    # Read the image file using OpenCV
    image = cv2.imread("result.jpg")

    # Convert the image to JPEG format
    _, buffer = cv2.imencode('.jpg', image)

    # Return the image
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()
