from ultralytics import YOLO
from datetime import datetime
from utils import *
from tracking import *
from visualize import *
cols = {
    "frame_id" : 0,
    "target_id" : 1,
    "x" : 2,
    "y" : 3,
    "w" : 4,
    "h" : 5,
    "score" : 6,
    "class_id" : 7,
    "truncation" : 8,
    "occlusion" : 9
}
path_ann = "VisDrone2019-VID-val/annotations/"
path_seq = "VisDrone2019-VID-val/sequences/"
model = YOLO("yolov8n.pt")
video_sequences = get_dataset(path_seq,path_ann,cols)
first_frame = True
tracking_data = {}
track_id = 0
for img_path,gts in video_sequences[2]:
    img = cv2.imread(img_path)
    results = model(img,verbose=False)
    class_labels = results[0].names
    bbs_with_centres = compute_centres(results)
    if first_frame:
        first_frame = False
        tracked_data,track_count = assign_ids(bbs_with_centres,results[0].names,3)
    else:
        tracked_data,track_count = track_ids(bbs_with_centres,tracked_data,track_count,results[0].names,3)
    visualize_preds(tracked_data,img)
cv2.destroyAllWindows()