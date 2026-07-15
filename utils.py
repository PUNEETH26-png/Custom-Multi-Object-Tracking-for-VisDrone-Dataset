import os
def annotation_row(str):
    ''' 
    str : raw row taken directly from file
    processes it and returns the row as a list of values where each index represents some meaning
    refer 'cols' dictionary to know the meanings of each index 
    '''
    str_new = str.removesuffix('\n')
    row = str_new.split(',')
    return row
def read_file(path):
    ''' 
    path : path to be opened
    returns the data in the form of list where each element in list is a row 
    '''
    with open(path,"r") as f:
        text = f.readlines()
    return text
def get_bb_values(cols,row):
    '''
    cols : dictionary where keys are column names are values are integer indexes 
    row  : one row from large number of rows of single frame
    '''
    x,y,w,h = int(row[cols["x"]]),int(row[cols["y"]]),int(row[cols["w"]]),int(row[cols["h"]])
    return [x,y,w,h]


def get_ann_dict(annotation_rows,cols):
    '''
    annotation_rows : takes annotation details of one entire video sequence
    returns the dictionary where keys are frame ids 
    and values are bbs and other data of the video
    '''
    ann_dict = {}
    for row in annotation_rows:
        if row[cols["frame_id"]] not in ann_dict:
            ann_dict[row[cols["frame_id"]]] = {}
            ann_dict[row[cols["frame_id"]]]["bounding_box"] = []
            ann_dict[row[cols["frame_id"]]]["target_id"] = []
        ann_dict[row[cols["frame_id"]]]["bounding_box"].append(get_bb_values(cols,row))
        ann_dict[row[cols["frame_id"]]]["target_id"].append(row[cols["target_id"]])
    return ann_dict
def scale_dims(w,h,scale_factor):
    return w//scale_factor,h//scale_factor
def scale_bbs(bbs,scale_factor):
    bbs_new = []
    for p in bbs:
        bbs_new.append(p//scale_factor)
    return bbs_new

def get_dataset(path_seq,path_ann,cols): 
    '''
    returns the entire dataset
    returns list 
    each element in the list is also a list which represents a single video and its annotations
    each element's structure in a single video is as follows
    (path_to_frame_i,frame_info)
    example 
    ("VisDrone2019-VID-val/sequences/uav0000086_00000_v/0000001.jpg",
    {"bounding_box" : [[x1,y1,w1,h1].[x2,y2,w2,h2],...[xn,yn,wn,hn]],"target_ids" : [1,2,3,4,...n]})
    '''
    video_sequences = []
    for vid,ann_list in zip(os.listdir(path_seq),os.listdir(path_ann)):
        anns_raw = read_file(path_ann+ann_list)
        anns = [annotation_row(a) for a in anns_raw]
        ann_dict = get_ann_dict(anns,cols)
        video_sequence = []
        for i,frame in enumerate(os.listdir(path_seq+vid)):
            video_sequence.append((path_seq+vid+"/"+frame,ann_dict[str(i+1)]))
        video_sequences.append(video_sequence)
    return video_sequences