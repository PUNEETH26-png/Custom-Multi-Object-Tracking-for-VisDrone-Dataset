import cv2
def gt_vs_preds(results,gts,img):
    ''' 
    results : results obtained from YOLO model directly 
    gts : ground truth bounding boxes of single frame in a video sequence
    img : image of type numpy
    Displays the image with ground truth bounding boxes and predicted bounding boxes
    !!! this version of code is deleted so the further invocations of this function may give 
    errros or wrong results, 
    '''
    for box in results[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0][0],box.xyxy[0][1],box.xyxy[0][2],box.xyxy[0][3]
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    for gt in gts["bounding_box"]:
        x1,y1,x2,y2 = gt[0],gt[1],gt[0]+gt[2],gt[1]+gt[3]
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow('image',img)
    cv2.waitKey(1)
def visualize_preds(tracking_data,img):
    for key,value in tracking_data.items():
        for box,id,labels in zip(value["boxes"],value["track_ids"],value["labels"]):
            x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            text = f"ID: {labels+str(id)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            color = (0, 255, 0) 
            thickness = 2
            cv2.putText(img, text, ((int(x1)+int(x2))//2,(int(y1)+int(y2))//2), font, font_scale, color, thickness)
    cv2.imshow('image',img)
    cv2.waitKey(40)