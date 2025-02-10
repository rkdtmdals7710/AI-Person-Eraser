from ultralytics import SAM, YOLO
import numpy as np
import cv2
from flask import Flask, request, render_template, make_response

app = Flask(__name__)

object_detection = YOLO('yolo11n.pt')
sam = SAM('sam_b.pt')
        

@app.route('/image', methods=['POST'])
def getMask():
    if request.method == 'POST':
        file = request.files["file"]
        
        byte = file.read()
        img_ndarray = np.frombuffer(byte, dtype=np.uint8)
        img = cv2.imdecode(img_ndarray, cv2.IMREAD_COLOR)
      
        h,w = img.shape[:2]

        points = (w/2, h/2) 


        object_detection_result = object_detection(img, classes=[0])
        bbox_list = object_detection_result[0].boxes.xyxy.cpu().numpy().tolist()

        for bbox in bbox_list :
            xmin, ymin, xmax, ymax = bbox[0:4]
            if xmin <= points[0] <= xmax and ymin <= points[1] <= ymax :
                sam_result = sam(img, bboxes=[xmin, ymin, xmax, ymax])
                
                mask_line = sam_result[0].masks.xy[0].astype(np.int32)
                
                pts = np.array(mask_line, np.int32)
                
                mask = np.full((h, w), 255, dtype=np.uint8)
                
                cv2.polylines(mask, [pts], isClosed=True, color=0, thickness=1)
                
                flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
                
                seed_point = (int(points[0]), int(points[1]))
                
                cv2.floodFill(mask, flood_mask, seed_point, 0)    
      

        ret, result_img = cv2.imencode('.jpg', mask)
        response = make_response(result_img.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
        
        return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7777, debug=True)
