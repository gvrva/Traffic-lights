#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import torch
import os
import numpy as np
import json


# In[2]:


def affect(boxes, height, width):
    distances = []
    areas = []
    for box in boxes:
        areas.append((box[3]*width)*(box[4]*height))
    max_area = max(areas)
    for i, area in enumerate(areas):
        if max_area/area>=3:
            areas[i]=0
            distances.append(1000)
        else:
            d = abs(boxes[i][1]-0.5)
            distances.append(d)
    affect_index = distances.index(min(distances))
    affect_array = []
    for i in range(len(distances)):
        if i == affect_index:
            affect_array.append(True)
        else:
            affect_array.append(False)
    return affect_array


# In[3]:


#без сглаживания
def txt_to_json(video_name, path_to_txt, path_to_video, json_path):
    
    #video_name - имя текущего видео
    #path_to_txt - путь, где хранится txt файлы после yolo, в цонце должен быть обратный слеш, например,'labels/'
    #path_to_video - путь к исходному видео
    #json_path - путь и имя файла, где сохранится json

    cap = cv.VideoCapture(path_to_video) # Вывод с видео файла
    frame_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(frame_width, frame_height)

    all_txt = list(sorted(os.listdir(path_to_txt)))
    txt = [t for t in all_txt if video_name in t]
    dict_predictions = {}
    colors = {2: "red", 1: "yellow", 0: "green"}

    for t in txt:
        i = int(t[len(video_name)+1:-4])
        if path_to_txt[-1] == '/':
            file_name = path_to_txt+t
        else:
            file_name = path_to_txt+'/'+t
        with open(file_name) as f:
            boxes = f.readlines()

            boxes = np.array([list(map(float,box[:-1].split())) for box in boxes])

            affect_list = affect(boxes, frame_height, frame_width)
            dict_predictions[i] = []
            for j, box in enumerate(boxes):
                box[0] = int(box[0])

                width = box[3]*frame_width
                height = box[4]*frame_height

                x_left = box[1]*frame_width-width/2
                y_left = box[2]*frame_height-height/2
                x_right = box[1]*frame_width+width/2
                y_right = box[2]*frame_height+height/2

                dict_predictions[i].append({j:{}})
                curr_box = list(map(str,[int(x_left), int(y_left), int(x_right), int(y_right)]))

                dict_predictions[i][j][j]["coords"] = curr_box

                color = box[0]
                if color not in colors.keys():
                    dict_predictions[i][j][j]["state"] = "unknown"
                else:
                    dict_predictions[i][j][j]["state"] = colors[color]

                dict_predictions[i][j][j]["affect"] = str(affect_list[j])
    with open(json_path, 'w') as outfile:
        json.dump(dict_predictions, outfile)


# In[5]:


def video_display(source_video_path, target_video, vid_boxes, fps = 24):
    cap = cv.VideoCapture(source_video_path)
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    i=0
    out = cv.VideoWriter(target_video,cv.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    with open(vid_boxes) as json_file:
        data = json.load(json_file)
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = []
        colors = []
        affects = []
        if str(i) in data.keys():
            for j, tl_dict in enumerate(data[str(i)]):
                box = tl_dict[str(j)]["coords"]
                boxes.append(list(map(int, box)))
                state = tl_dict[str(j)]["state"]
                if state == 'red':
                    colors.append((0,0,255))
                if state == 'yellow':
                    colors.append((0,255,255))
                if state == 'green':
                    colors.append((0,255,0))
                if state == 'unknown':
                    colors.append((255,255,255))
                affects.append(tl_dict[str(j)]["affect"])
            
        for box, color, affect in zip(boxes, colors, affects):
            
            cv.rectangle(frame, (box[2], box[3]), (box[0], box[1]), color, 2)
            if affect == 'True':
                frame = cv.putText(frame, 'affect', (box[0], box[3]+20), fontFace = cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale = 0.5, thickness = 1, color = color)

        out.write(frame)
        i += 1
        
    out.release()


# ## Пример использования

# In[ ]:


txt_to_json('video_3', 'labels/', 'phase_1/video_3.MP4', 'video_3_json.txt')


# In[ ]:


video_display("phase_1/video_3.MP4", "phase_1/video_3_boxes.MP4", "video_3_json.txt", fps = 24)

