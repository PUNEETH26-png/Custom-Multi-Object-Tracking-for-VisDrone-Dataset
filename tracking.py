import math
import numpy as np
from scipy.optimize import linear_sum_assignment
def measure_distance(centre_with_ids,centre):
    x1,y1,x2,y2 = centre_with_ids[0],centre_with_ids[1],centre[0],centre[1]
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
def compute_centres(results):
    '''
    results are from yolo model output directly
    computes centre for each bounding box
    returns a list where each element is a tuple containing bounidng box, centre and 
    class label predicted from yolo model
    centres[i] : ([x1,y1,x2,y2],centre,cls_idx)
    [x1,y1,x2,y2] : bounding box co ordinates
    centre : centre of bb
    cls_idx : class label index of each predicted bounding box
    '''
    centres = []
    for box in results[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0][0].item(),box.xyxy[0][1].item(),box.xyxy[0][2].item(),box.xyxy[0][3].item()
        centre,cls_idx = ((x1+x2)//2,(y1+y2)//2),int(box.cls.item())
        centres.append(([x1,y1,x2,y2],centre,cls_idx))
    return centres
def assign_ids(bbs_with_centres,class_labels,TTL):
    tracking_data = {}
    track_ids_count = 0
    for item in bbs_with_centres:
        box = item[0]
        centre = item[1]
        label_index = item[2]
        if label_index not in tracking_data.keys():
            tracking_data[label_index] = {
                "centre" : [],
                "track_ids" : [],
                "boxes" : [],
                "labels" : [],
                "TTL" : []
            }
        tracking_data[label_index]["centre"].append(centre)
        tracking_data[label_index]["track_ids"].append(track_ids_count)
        tracking_data[label_index]["boxes"].append(box)
        tracking_data[label_index]["labels"].append(class_labels[label_index])
        tracking_data[label_index]["TTL"].append(TTL)
        track_ids_count += 1
    return tracking_data,track_ids_count
def track_ids(bbs_with_centres,tracking_data,track_count,class_labels,TTL):
    '''
    Arguments :
    bb_with_centres : list of elements where each element is a tuple of 4 elements
                      bb_with_centres[i] = (bounding_box,centre,cls_idx)
                      bounding_box = [x1,y1,x2,y2]
                      centre       = [(x1+x2)//2,(y1+y2)//2]
                      cls_idx      = index of class predicted bby YOLO model
    tracking_data   : dictionary
                    : (key)   : cls_idx predicted by YOLO model
                    : (value) : dictionary with keys {centre,track_ids,boxes,class_labels}
    track_count     : Number of objects currently the algorithm is tracking
    class_labels    : list of class labels the model is trained on

    returns :

    tracking_data   : updated tracking data 
    track_count     : updated track_count

    the function adds new objects which are not present in previous frame
    the function checks for ttl values and removes the objects which are abscent in current frame and present in previous frame

    '''
    pred_dictionary = {}
    for item in bbs_with_centres:
        box = item[0]
        centre = item[1]
        label_index = item[2]
        if label_index not in pred_dictionary.keys():
            pred_dictionary[label_index] = {"centre" : [],"boxes" : [],"labels" : []}
        pred_dictionary[label_index]["centre"].append(centre)
        pred_dictionary[label_index]["boxes"].append(box)
        pred_dictionary[label_index]["labels"].append(class_labels[label_index])
    new_classes = {}
    for key,value in pred_dictionary.items():
        if key not in tracking_data.keys(): #new class label arrived
            new_track_ids = [i+track_count for i in range(len(value["centre"]))]
            ttl_vals = [TTL for _ in range(len(value["centre"]))]
            new_classes[key] = {"centre" : value["centre"],"boxes" : value["boxes"],"labels" : value["labels"],"track_ids" : new_track_ids,"TTL" : ttl_vals}
            track_count += len(value["centre"])
            continue
        prev_frame_data = tracking_data[key]
        num_prev = len(prev_frame_data["centre"])
        num_curr = len(value["centre"])
        Cost_Matrix = [] # cost matrix is calculated for single class label 
        for i in range(len(prev_frame_data["centre"])):
            new_row = []
            for j in range(len(value["centre"])):
                new_row.append(measure_distance(prev_frame_data["centre"][i],value["centre"][j]))
            Cost_Matrix.append(new_row)
        Cost_Matrix_np = np.array(Cost_Matrix)
        track_centre_idxs,pred_centre_idxs = linear_sum_assignment(Cost_Matrix_np)
        missing_objects = [i for i in range(num_prev) if i not in track_centre_idxs]
        new_objects = [i for i in range(num_curr) if i not in pred_centre_idxs]
        if len(new_objects)>0:
            new_track_ids = [i+track_count for i in range(len(new_objects))]
            new_obj_centres = [value["centre"][new_objects[i]] for i in range(len(new_objects))]
            new_obj_boxes = [value["boxes"][new_objects[i]] for i in range(len(new_objects))]
            new_obj_labels = [value["labels"][new_objects[i]] for i in range(len(new_objects))]
            new_obj_ttl = [TTL for i in range(len(new_objects))]
            new_classes[key] = {"centre" : new_obj_centres,"boxes" : new_obj_boxes,"labels" : new_obj_labels,"track_ids" : new_track_ids,"TTL" : new_obj_ttl}
            track_count += len(new_obj_centres)
        if len(missing_objects)>0:
            for i in range(len(missing_objects)):
                if prev_frame_data["TTL"][missing_objects[i]]>0:
                    prev_frame_data["TTL"][missing_objects[i]] -= 1

        for a,b in zip(track_centre_idxs,pred_centre_idxs):
            tracking_data[key]["centre"][a] = value["centre"][b]
            tracking_data[key]["boxes"][a] = value["boxes"][b]
            tracking_data[key]["TTL"][a] = TTL

    for key, value in new_classes.items():
        if key in tracking_data:
            for k,v in new_classes[key].items():
                tracking_data[key][k].extend(v)
        else:
            tracking_data[key] = value
    for key,value in tracking_data.items():
        new_value = {}
        for k,v in value.items():
            for i in range(len(v)):
                if value["TTL"][i]!=0:
                    if k not in new_value:
                        new_value[k] = []
                    new_value[k].append(v[i])
        tracking_data[key] = new_value
    return tracking_data,track_count