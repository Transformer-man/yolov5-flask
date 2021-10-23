import cv2
import time
from flask import Flask, request, Response,render_template
import json
from cam.base_camera import BaseCamera

from models.de import detect,get_model
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
app = Flask(__name__)
class_names = [c.strip() for c in open(r'cam/coco.names').readlines()]
file_name = ['jpg','jpeg','png']

yolov5_model = get_model()

@app.route('/images', methods= ['POST'])
def get_image():
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    if image_name.split(".")[-1] in file_name:
        img = cv2.imread(image_name)
        img = detect(yolov5_model,img)
        _, img_encoded = cv2.imencode('.jpg', img)
        response = img_encoded.tobytes()
        os.remove(image_name)
        try:
            return Response(response=response, status=200, mimetype='image/jpg')
        except:
            return render_template('index1.html')
@app.route('/')
def upload_file():
   return render_template('index1.html')
if __name__ == '__main__':
    #    Run locally
    app.run(debug=True, host='127.0.0.1', port=5000)
    #Run on the server
    # app.run(debug=True, host = '0.0.0.0', port=5000)
