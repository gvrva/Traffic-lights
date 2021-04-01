import cv2 as cv
import torch
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import copy

def affect(boxes, width):
    """
    Determines the effect of all traffic lights in the frame on the movement of the vehicle.
    Returns a list of boolean values (True - affect, False - does not affect)according to the order
    of the traffic lights in boxes.
    
    boxes: list of bounding boxes with shape (N,4) and integer values.
    width: width of frame, int.
    
    Only one traffic light is determined as affecting.  
    
    """
    distances = []
    if len(boxes)==0:
        return []
    areas = []
    for box in boxes:
        areas.append((box[2]-box[0])*(box[3]-box[1]))
    max_area = max(areas)
    for i, box in enumerate(boxes):
        if areas[i]<max_area/4.:
            d = 2000
        else:
            d = abs((box[2]+box[0])/2.-width/2.)
            if d == 0:
                d = 1
        distances.append(d)
    affect_index_1 = distances.index(min(distances))
    if len(distances)==1:
        return [True]
    distances_new = copy.deepcopy(distances)
    distances_new[affect_index_1]=2000
    affect_index_2 = distances.index(min(distances_new))
    # if the nearest traffic light is under the second one and they are not far from each other,
    # it means that the nearest traffic light is far and cannot affect
    if boxes[affect_index_1][1]>=boxes[affect_index_2][3] and distances[affect_index_2]<=distances[affect_index_1]*1.5:
        affect_index = affect_index_2
    else:
        affect_index = affect_index_1
    affect_array = []
    for i in range(len(distances)):
        if i == affect_index:
            affect_array.append(True)
        else:
            affect_array.append(False)
    return affect_array

def interpolation(dict_predict, frame_width, frame_height):
    """
    Smooths results of detection and deletes unnecessary occasional detections.
    Returns new dictionary of detections.
    
    dict_predict: initial dictionary of traffic light detections for every frame of video, dict.
    frame_width: width of video frames, int.
    frame_height: height of video frames, int.
    
    Algorithm
    1. Makes the dictionary for tracking of traffic lights.
    2. Interpolate coordinates of traffic lights between every 20th frame.
    3. Determine does the traffic light affect on the car.
    
    """
    interpolated_predict = copy.deepcopy(dict_predict)
    
    # dictionary for tracking the traffic lights
    # {id_1: [[coords_1, frame number_1, id in dict_predictions_1 (j)],
    #       [coords_2, frame number_2, id in dict_predictions_2 (j)]],
    #  id_2: [[coords_1, frame number_1, id in dict_predictions_1 (j)],
    #       [coords_2, frame number_2, id in dict_predictions_2 (j)]],
    #        ...}
    dict_key_points = {}
    
    
    # dictionary of coordinates of current traffic lights
    curr_points = {}
    
    # dictionary of traffic light ids in initial dict_predict
    dict_j = {}
    
    frame_numbers = list(map(str,sorted(list(map(int,interpolated_predict.keys())))))
    
    next_id = 0
    for frame_number in frame_numbers:
        updated_tl = set() #set of ids of updated or new traffic ligths
        new_tl = set()
        for key in interpolated_predict[frame_number].keys():
            box = interpolated_predict[frame_number][key]
            coord = box['coords']
            old_point = False    # flag of previously detected traffic light
            if coord[0]<=2 or coord[2]<=2 or coord[1]>=frame_width-2 or coord[3]>=frame_height-2:
                continue
            cx = (coord[0]+coord[2])/2.
            cy = (coord[1]+coord[3])/2.
            
            # updating current points of traffic lights
            for point_key in curr_points.keys():
                if curr_points[point_key][2]>cx and curr_points[point_key][0]<cx and curr_points[point_key][3]>cy and curr_points[point_key][1]<cy and int(frame_number)-int(curr_points[point_key][4])==20:
                    old_point = True
                    curr_points[point_key] = [coord[0], coord[1], coord[2], coord[3], frame_number]                            
                    dict_j[point_key][frame_number] = key
                    updated_tl.add(point_key)
            
            # if traffic light is new, then it adds to curr_points with new id.
            if old_point == False:
                curr_points[next_id] = [coord[0], coord[1], coord[2], coord[3], frame_number]
                dict_key_points[next_id] = []
                dict_j[next_id] = {}
                dict_j[next_id][frame_number] = key
                updated_tl.add(next_id)
                new_tl.add(next_id)
                next_id+=1
            
        # if the box is not updated in 20 frames and not new, it is deleted from curr_points
        # and it becomes the last point of its traffic light
        first_last_boxes = list(set(curr_points.keys())-updated_tl)
        for key in first_last_boxes:
            dict_key_points[key].append(curr_points[key])
            curr_points.pop(key)
        
        # new points are added to tracking dictionary
        for key in list(new_tl): 
            dict_key_points[key].append(curr_points[key])
                
    # interpolation
    for key in dict_key_points.keys(): # key - traffic light id
        key_boxes = dict_key_points[key]
            
        frame_count = int(key_boxes[-1][4])-int(key_boxes[0][4])
        if frame_count == 0:
            continue
                
        x = []
        y = []
            
        width = []
        height_coef = (key_boxes[i][3]-key_boxes[i][1])/float(key_boxes[i][2]-key_boxes[i][0])
            
        frame_number = int(key_boxes[0][4])
        for i in range(frame_count):
            if i<2:
                x.append((key_boxes[i][2]+key_boxes[i][0])/2.)
                y.append((key_boxes[i][3]+key_boxes[i][1])/2.)
                width.append(key_boxes[i][2]-key_boxes[i][0])
                continue

            x.append((key_boxes[i][2]+key_boxes[i][0])/2.)
            y.append((key_boxes[i][3]+key_boxes[i][1])/2.)
            width.append(key_boxes[i][2]-key_boxes[i][0])
            
            if i<=30:
                dx = np.mean(np.diff(np.array(x)))
                dy = np.mean(np.diff(np.array(y)))
                dw = np.mean(np.diff(np.array(width)))

            else:
                dx = np.mean(np.diff(np.array(x[i-30:])))
                dy = np.mean(np.diff(np.array(y[i-30:])))
                dw = np.mean(np.diff(np.array(width[i-30:])))

            if i<=30:
                key_index = 0
                l = i+1
            else:
                key_index = i-30
                l = 30
            
            start_cx = (key_boxes[key_index][2]-key_boxes[key_index][0])/2.
            start_cy = (key_boxes[key_index][3]-key_boxes[key_index][1])/2.
            start_width = key_boxes[key_index][2]-key_boxes[key_index][0]

            new_x_left = int((start_cx + dx*l)-(start_width+dw)/2.)
            new_y_left = int((start_cy + dy*l)-(start_width+dw)*height_coef/2.)
            new_x_right = int((start_cx + dx*l)+(start_width+dw)/2.)
            new_y_right = int((start_cy + dy*l)+(start_width+dw)*height_coef/2.)

            interpolated_predict[str(int(key_boxes[0][4])+i)][dict_j[key][str(int(key_boxes[0][4])+i)]]['coords']=[new_x_left, new_y_left, new_x_right, new_y_right]

    
    # determine what traffic light affects
    start_frame = None
    color = None
    count = 0
    pause = 0
    prev_box = None
    for frame in interpolated_predict.keys():
        box_keys = list(interpolated_predict[frame].keys())
        boxes = []
        affect_keys = []
        for box_key in box_keys:
            if interpolated_predict[frame][box_key]['state'] != "unknown":
                boxes.append(interpolated_predict[frame][box_key]['coords'])
                affect_keys.append(box_key)
            else:
                interpolated_predict[frame][box_key]['affect']= False

        if pause>1:
            for i, box_key in enumerate(affect_keys):
                interpolated_predict[frame][box_key]['affect'] = False
            pause -= 1
            continue
        
        affect_list = affect(boxes, frame_width)

        for i, box_key in enumerate(affect_keys):
            interpolated_predict[frame][box_key]['affect'] = affect_list[i]
            if affect_list[i]==True:
                if start_frame == None:
                    start_frame = frame
                    color = interpolated_predict[frame][box_key]['state']
                    count = 1
                    prev_box = interpolated_predict[frame][box_key]['coords']
                elif color == interpolated_predict[frame][box_key]['state']:
                    count +=1
                    prev_box = interpolated_predict[frame][box_key]['coords']
                elif color != interpolated_predict[frame][box_key]['state']:
                    cx = (interpolated_predict[frame][box_key]['coords'][0]+interpolated_predict[frame][box_key]['coords'][2])/2.
                    cy = (interpolated_predict[frame][box_key]['coords'][1]+interpolated_predict[frame][box_key]['coords'][3])/2.
                    
                    # if the same traffic light changes the color it continue to be detected as affect True
                    # without further delay in detecting affect
                    if cx > prev_box[0] and cx < prev_box[1] and cy > prev_box[1] and cy < prev_box[3]:
                        start_frame = frame
                        color = interpolated_predict[frame][box_key]['state']
                        count = 1
                        prev_box = interpolated_predict[frame][box_key]['coords']
                    else:
                        if count>30:
                            interpolated_predict[frame][box_key]['affect'] = False
                            start_frame = None
                            color = None
                            count = 0
                            pause = 15
                        else:
                            interpolated_predict[frame][box_key]['affect'] = False
                            start_frame = None
                            color = None
                            count = 0
                            pause = 30
    
    return interpolated_predict

def check_similarity(prev_image, curr_image, boxes):
    """
    Calculates the similarity between current frame and previous one in bounding boxes parts.
    Returns True if images are similar and False if images differs
    
    prev_image:  previous image, numpy array.
    curr_image:  current image, numpy array.
    boxes: array of boxes from previous image with shape (N,4) and int values.
    
    """
    prev = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
    curr = cv.cvtColor(curr_image, cv.COLOR_BGR2GRAY)
    for box in boxes:

        cropped_current_image = curr[box[1]:box[3],box[0]:box[2]]
        cropped_previous_image = prev[box[1]:box[3],box[0]:box[2]]

        cropped_current_image_norm = cropped_current_image/255.
        cropped_previous_image_norm = cropped_previous_image/255.

        similarity_rate = abs(np.mean(cropped_current_image_norm-cropped_previous_image_norm))
        
        if similarity_rate<0.02:
            continue
        else:
            return False
    return True

def video_predict(path, json_path, model):
    """
    Prediction of traffic lights, its color and affect status on video.
    No return.
    
    path: path to video, str.
    json_path: path to output json file, str.
    model: network model.
    
    Example:
    video_predict("phase_1/video_3.MP4", "phase_1/video_3.json", model)
    
    """
    cap = cv.VideoCapture(path)
    frame_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    dict_predictions = {}
    model.eval()
    i=1
    colors = {2: "red", 1: "yellow", 0: "green"}
    image_previous = None
    boxes_previous = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # makes prediction of every second frame
        # if number of frame is even, predictions are copied from previous frame
        if i%2==0:
            dict_predictions[str(i)] = copy.deepcopy(dict_predictions[str(i-1)])
            i+=1
            continue
        
        # calculation of similarity
        # if images are similar in bounding box part, predictions are copied from previous frame
        if (image_previous is not None) and (boxes_previous is not None):
            similar = check_similarity(image_previous, frame, boxes_previous)
            if similar:
                dict_predictions[str(i)] = copy.deepcopy(dict_predictions[str(i-1)])
                i+=1
                continue
        
        # prediction for current frame
        with torch.no_grad():
            image_previous = frame.copy()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            prediction = model([frame], size = frame_width)
            dict_predictions[str(i)]={}

            confidence = prediction.xyxy[0].cpu().numpy().astype('float')[:,4]

            boxes = prediction.xyxy[0].cpu().numpy().astype('int')
            
            if len(boxes)>0:
                boxes_previous = boxes.copy()
            else:
                boxes_previous = None
            
            for j, box in enumerate(boxes):
                if confidence[j]<0.4:
                    continue
                dict_predictions[str(i)][str(j)] = {}
                curr_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

                dict_predictions[str(i)][str(j)]["coords"] = curr_box

                color = box[-1]
                if color not in colors.keys():
                    dict_predictions[str(i)][str(j)]["state"] = "unknown"
                else:
                    dict_predictions[str(i)][str(j)]["state"] = copy.deepcopy(colors[color])

        i+=1

    # smoothing of bounding boxes and deleting unnecessary bounding boxes
    final_predictions = interpolation(dict_predictions, frame_width, frame_height)
    
    with open(json_path, 'w') as outfile:
        json.dump(final_predictions, outfile)

def video_display(source_video_path, target_video, boxes_path):
    """
    Diaplay predicted bounding boxes on the source video.
    No return.
    
    source_video_path: path to the source video.
    target_video: path to target video.
    boxes_path: path to json file with predictions.
    
    Example:
    video_display("phase_1/video_3.MP4", "phase_1/video_3_boxes.MP4", "phase_1/video_3.json")
    
    """
    cap = cv.VideoCapture(source_video_path)
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    i=1
    out = cv.VideoWriter(target_video,cv.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    with open(boxes_path) as json_file:
        data = json.load(json_file)
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = []
        colors = []
        affects = []
        if str(i) in data.keys():
            for j in data[str(i)].keys():
                box = data[str(i)][j]["coords"]
                boxes.append(list(map(int, box)))
                state = data[str(i)][j]["state"]
                if state == 'red':
                    colors.append((0,0,255))
                if state == 'yellow':
                    colors.append((0,255,255))
                if state == 'green':
                    colors.append((0,255,0))
                if state == 'unknown':
                    colors.append((255,255,255))
                affects.append(data[str(i)][j]["affect"])
            
        for box, color, affect in zip(boxes, colors, affects):
            
            cv.rectangle(frame, (box[2], box[3]), (box[0], box[1]), color, 2)
            if affect == True:
                text_color = tuple([50 if c==255 else c for c in color])
                frame = cv.putText(frame, 'affect', (box[0], box[3]+20), fontFace = cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale = 0.6, thickness = 2, color = text_color)

        out.write(frame)
        i += 1
        
    out.release()
