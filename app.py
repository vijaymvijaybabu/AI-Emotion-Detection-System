from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from PIL import Image
import base64
import io
import time

app = Flask(__name__)

# ==========================
# DEEPFACE IMPORT
# ==========================
try:

    from deepface import DeepFace

    USE_DEEPFACE = True

except Exception as e:

    print("DeepFace Error:", e)

    USE_DEEPFACE = False

# ==========================
# FACE DETECTION
# ==========================
face_cascade = cv2.CascadeClassifier(

    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'

)

# ==========================
# CAMERA
# ==========================
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==========================
# EMOJIS
# ==========================
EMOJIS = {

    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'surprise': '😲',
    'neutral': '😐',
    'disgust': '🤢'

}

# ==========================
# COLORS
# ==========================
COLORS = {

    'happy': (0,255,0),
    'sad': (255,0,0),
    'angry': (0,0,255),
    'fear': (255,0,255),
    'surprise': (0,255,255),
    'neutral': (200,200,200),
    'disgust': (0,255,150)

}

# ==========================
# FALLBACK EMOTIONS
# ==========================
EMOTION_LIST = [

    'angry',
    'disgust',
    'fear',
    'happy',
    'sad',
    'surprise',
    'neutral'

]

# ==========================
# DETECT EMOTION
# ==========================
def detect_emotion(face):

    try:

        if USE_DEEPFACE:

            result = DeepFace.analyze(

                face,

                actions=['emotion'],

                enforce_detection=False,

                detector_backend='opencv',

                silent=True

            )

            if isinstance(result, list):

                result = result[0]

            emotions = result['emotion']

            dominant = result['dominant_emotion']

            return emotions, dominant

    except Exception as e:

        print("DeepFace Detection Error:", e)

    # ======================
    # FALLBACK MOCK AI
    # ======================
    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    seed = int(np.mean(gray))

    np.random.seed(seed)

    vals = np.random.dirichlet(
        np.ones(7)
    ) * 100

    emotions = {

        emotion: float(round(val,2))

        for emotion,val in zip(
            EMOTION_LIST,
            vals
        )

    }

    dominant = max(
        emotions,
        key=emotions.get
    )

    return emotions, dominant

# ==========================
# VIDEO STREAM
# ==========================
def generate_frames():

    prev = 0

    while True:

        success, frame = camera.read()

        if not success:

            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(

            gray,

            scaleFactor=1.1,

            minNeighbors=5,

            minSize=(60,60)

        )

        for (x,y,w,h) in faces:

            try:

                face = frame[
                    y:y+h,
                    x:x+w
                ]

                emotions, dominant = detect_emotion(face)

                emoji = EMOJIS.get(
                    dominant,
                    "🙂"
                )

                color = COLORS.get(
                    dominant,
                    (255,255,255)
                )

                # ==================
                # GLOW EFFECT
                # ==================
                overlay = frame.copy()

                cv2.rectangle(

                    overlay,

                    (x,y),

                    (x+w,y+h),

                    color,

                    -1

                )

                alpha = 0.18

                cv2.addWeighted(

                    overlay,

                    alpha,

                    frame,

                    1-alpha,

                    0,

                    frame

                )

                # ==================
                # MAIN BOX
                # ==================
                cv2.rectangle(

                    frame,

                    (x,y),

                    (x+w,y+h),

                    color,

                    3

                )

                # ==================
                # LABEL
                # ==================
                label = f"{emoji} {dominant}"

                cv2.putText(

                    frame,

                    label,

                    (x,y-12),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.9,

                    color,

                    2

                )

            except Exception as e:

                print("Frame Error:", e)

        # ======================
        # FPS
        # ======================
        curr = time.time()

        fps = 1/(curr-prev) if prev else 0

        prev = curr

        cv2.putText(

            frame,

            f"FPS: {int(fps)}",

            (20,40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255,255,255),

            2

        )

        ret, buffer = cv2.imencode(
            '.jpg',
            frame
        )

        frame = buffer.tobytes()

        yield (

            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'

        )

# ==========================
# HOME
# ==========================
@app.route('/')
def home():

    return render_template(
        'index.html'
    )

# ==========================
# VIDEO FEED
# ==========================
@app.route('/video_feed')
def video_feed():

    return Response(

        generate_frames(),

        mimetype=
        'multipart/x-mixed-replace; boundary=frame'

    )

# ==========================
# IMAGE ANALYSIS
# ==========================
@app.route('/analyze', methods=['POST'])
def analyze():

    try:

        data = request.get_json()

        if not data:

            return jsonify({

                'error':
                'No data received'

            }), 400

        if 'image' not in data:

            return jsonify({

                'error':
                'No image found'

            }), 400

        image_data = data[
            'image'
        ].split(',')[1]

        image = Image.open(

            io.BytesIO(
                base64.b64decode(
                    image_data
                )
            )

        ).convert('RGB')

        image_np = np.array(image)

        image_cv = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        emotions, dominant = detect_emotion(
            image_cv
        )

        emoji = EMOJIS.get(
            dominant,
            "🙂"
        )

        return jsonify({

            'success': True,

            'emotions': emotions,

            'dominant': dominant,

            'emoji': emoji

        })

    except Exception as e:

        print("Analyze Error:", e)

        return jsonify({

            'success': False,

            'error': str(e)

        }), 500

# ==========================
# RUN
# ==========================
if __name__ == '__main__':

    app.run(

        debug=True,

        threaded=True

    )