from ultralytics import SAM, YOLO
import numpy as np
import cv2

imgpath = 'sample.jpg'

img = cv2.imread(imgpath)

h,w = img.shape[:2]

points = (w/2, h/2) # 포인트

object_detection = YOLO('yolo11n.pt')

sam = SAM('sam_b.pt')

object_detection_result = object_detection('sample.jpg', classes=[0])

bbox_list = object_detection_result[0].boxes.xyxy.cpu().numpy().tolist()

for bbox in bbox_list :
    xmin, ymin, xmax, ymax = bbox[0:4]
    if xmin <= points[0] <= xmax and ymin <= points[1] <= ymax :
        sam_result = sam('sample.jpg', bboxes=[xmin, ymin, xmax, ymax], save=True)
        
        mask_line = sam_result[0].masks.xy[0].astype(np.int32)
        
        pts = np.array(mask_line, np.int32)
        
        mask = np.full((h, w), 255, dtype=np.uint8)
        
        cv2.polylines(mask, [pts], isClosed=True, color=0, thickness=1)
        
         # floodfill을 위한 마스크는 원본보다 2픽셀 커야 함
        flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
        
        # 내부의 한 점을 시드 포인트로 사용
        seed_point = (int(points[0]), int(points[1]))
        
        # floodfill 수행 (흰색으로 채움)
        cv2.floodFill(mask, flood_mask, seed_point, 0)
        
        cv2.imwrite('mask.jpg', mask)
        
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        
        cv2.imwrite('masked_img.jpg', masked_img)


