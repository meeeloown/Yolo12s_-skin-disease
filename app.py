from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Konfigurasi folder
UPLOAD_FOLDER = r'C:\VScode\Python\YOLOv\static\uploads'
OUTPUT_FOLDER = r'C:\VScode\Python\YOLOv\static\results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Pastikan folder ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model YOLO
model_path = r"C:\VScode\Python\YOLOv\runs-20250525T191005Z-1-001\runs\detect\MekaDis_Yolo12s\weights\best.pt"
model = YOLO(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = secure_filename(image.filename)
            input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Normalisasi dan simpan gambar input
            img = Image.open(image)
            img = img.convert("RGB")
            img_resized = img.resize((640, 640))
            img_resized.save(input_image_path)

            try:
                # Deteksi objek
                results = model.predict(source=input_image_path)

                # Simpan hasil gambar deteksi
                output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                results[0].save(filename=output_image_path)

                # Ambil confidence score dan label
                detections = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = model.names[cls_id]
                    detections.append({
                        'label': label,
                        'confidence': round(confidence * 100, 2)
                    })

                # Hitung rata-rata confidence
                if detections:
                    avg_confidence = round(sum(d['confidence'] for d in detections) / len(detections), 2)
                else:
                    avg_confidence = 0.0

                # Kirim ke template
                input_image_url = '/static/uploads/' + filename
                output_image_url = '/static/results/' + filename

                return render_template('index.html',
                                       input_image=input_image_url,
                                       output_image=output_image_url,
                                       detections=detections,
                                       avg_confidence=avg_confidence)

            except Exception as e:
                return f"❌ Terjadi kesalahan saat memproses gambar: {str(e)}"

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
