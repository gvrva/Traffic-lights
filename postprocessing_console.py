#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import torch
import os
import numpy as np
import json


# In[2]:


def affect(boxes, width):
    distances = []
    areas = []
    for box in boxes:
        areas.append((int(box[3])-int(box[1]))*(int(box[2]) - int(box[0])))
    max_area = max(areas)
    for i, area in enumerate(areas):
        if max_area/area>=3.5:
            areas[i]=0
            distances.append(1000)
        else:
            d = abs((int(boxes[i][2])+int(boxes[i][0]))/2.-width/2.)
            if d == 0:
                d = 1
            distances.append(d)
    affect_index_1 = distances.index(min(distances))
    if len(distances)==1:
        return [True]
    distances_new = distances.copy()
    distances_new[affect_index_1]=2000
    affect_index_2 = distances.index(min(distances_new))
    if boxes[affect_index_1][2]>=boxes[affect_index_2][2] and distances[affect_index_2]/distances[affect_index_1]<=1.5:
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


# In[ ]:


def interpolation(dict_predict, frame_width, frame_height):
    interpolated_predict = copy.deepcopy(dict_predict)
    
    # время выполнения без интерполяции 508 ms
    # пересчет координат после всего
    
    # словарь для каждого светофора с координатами каждые 20 кадров
    # {id_1: [[coords_1, frame number_1, id in dict_predictions_1 (j)],
    #       [coords_2, frame number_2, id in dict_predictions_2 (j)]],
    #  id_2: [[coords_1, frame number_1, id in dict_predictions_1 (j)],
    #       [coords_2, frame number_2, id in dict_predictions_2 (j)]]}
    dict_key_points = {}
    
    # хранит координаты светофоров предудыщего кадра, если светофор отсутствует в новом кадре, то эти координаты записываются
    # в список соответствующего светофора в dict_key_points
    curr_points = {}
    
    # словарь хранит последовательность индексов j для каждого светофора на случай, если в каждом кадре он имеет разые индексы
    # это позволит не вычислять при пересчете координат 
    dict_j = {}
    
    frame_numbers = interpolated_predict.keys()
    
    next_id = 0
    for frame_number in sorted(list(frame_numbers)):
        updated_tl = set()
        new_tl = set()
        for key in interpolated_predict[frame_number].keys():
            box = interpolated_predict[frame_number][key]
            coord = list(map(int, box['coords']))
            old_point = False
            if coord[0]<=2 or coord[2]<=2 or coord[1]>=frame_width-2 or coord[3]>=frame_height-2:
                continue
            cx = (coord[0]+coord[2])/2.
            cy = (coord[1]+coord[3])/2.
            
            # обновление текущих точек
            for point_key in curr_points.keys():
                if curr_points[point_key][2]>cx and curr_points[point_key][0]<cx and curr_points[point_key][3]>cy and curr_points[point_key][1]<cy and frame_number-curr_points[point_key][4]<=15:
                    old_point = True
                    curr_points[point_key] = [coord[0], coord[1], coord[2], coord[3], frame_number]                            
                    dict_j[point_key][frame_number] = key
                    updated_tl.add(point_key)
            
            # если данный бокс новый, то old_point останется False, тогда нужно добавить в curr_points с новым id.
            if old_point == False:
                curr_points[next_id] = [coord[0], coord[1], coord[2], coord[3], frame_number]
                dict_key_points[next_id] = []
                dict_j[next_id] = {}
                dict_j[next_id][frame_number] = key
                updated_tl.add(next_id)
                new_tl.add(next_id)
                next_id+=1
            
        # проверка на необновленные боксы
        # если бокс не обновлен, то добавляеся в dict_key_points
        first_last_boxes = list(set(curr_points.keys())-updated_tl)
        for key in first_last_boxes:
            if frame_number - curr_points[key][4]>15:
                dict_key_points[key].append(curr_points[key])
                curr_points.pop(key)
            
        for key in list(new_tl): 
            dict_key_points[key].append(curr_points[key])
            
        # теперь запись оставшихся боксов в dict_key_points, если кадр 30-ый
        if frame_number%30==0:
            for key in list(set(curr_points.keys()-new_tl)):
                dict_key_points[key].append(curr_points[key])
                
    # перезапись боксов в dict_predictions
    for key in dict_key_points.keys():
        # key - номер светофора
        key_boxes = dict_key_points[key]
            
        start_width = key_boxes[0][2]-key_boxes[0][0]
        start_height = key_boxes[0][3]-key_boxes[0][1]
        
        if key_boxes[-1][4]-key_boxes[0][4]<2:
            for kb in range(len(key_boxes)):
                if key_boxes[kb][4] in interpolated_predict.keys():
                    if dict_j[key][key_boxes[kb][4]] in interpolated_predict[key_boxes[kb][4]].keys():
                        interpolated_predict[key_boxes[kb][4]].pop(dict_j[key][key_boxes[kb][4]])
                        if len(interpolated_predict[key_boxes[kb][4]])==0:
                            interpolated_predict.pop(key_boxes[kb][4])
            continue
        
        for kb in range(len(key_boxes)-1):
            
            if 2*(key_boxes[kb+1][4]-key_boxes[kb][4])==0:
                side_change = 1        
            else:
                area_start = (key_boxes[kb][2]-key_boxes[kb][0])*(key_boxes[kb][3]-key_boxes[kb][1])
                area_end = (key_boxes[kb+1][2]-key_boxes[kb+1][0])*(key_boxes[kb+1][3]-key_boxes[kb+1][1])
                side_change = (area_end/area_start)**(1/(2*(key_boxes[kb+1][4]-key_boxes[kb][4])))
            
            
            frame_count = key_boxes[kb+1][4]-key_boxes[kb][4]
            if frame_count == 0:
                continue
            # покадровое изменение каждой координаты светофора
            dx = ((key_boxes[kb+1][2]+key_boxes[kb+1][0])/2-(key_boxes[kb][2]+key_boxes[kb][0])/2)/frame_count
            dy = ((key_boxes[kb+1][3]+key_boxes[kb+1][1])/2-(key_boxes[kb][3]+key_boxes[kb][1])/2)/frame_count
            
            start_cx = (key_boxes[kb][2]+key_boxes[kb][0])/2
            start_cy = (key_boxes[kb][3]+key_boxes[kb][1])/2
            
            width = key_boxes[kb][2]-key_boxes[kb][0]
            height = start_height*(width/start_width)
            
            frame_number = key_boxes[kb][4]
            for i in range(frame_count):
                new_x_left = (start_cx + dx*i)-(width*(side_change**i))/2
                new_y_left = (start_cy + dy*i)-(height*(side_change**i))/2
                new_x_right = (start_cx + dx*i)+(width*(side_change**i))/2
                new_y_right = (start_cy + dy*i)+(height*(side_change**i))/2
                if frame_number+i in interpolated_predict.keys():
                    if frame_number+i in dict_j[key].keys():
                        interpolated_predict[frame_number+i][dict_j[key][frame_number+i]]['coords']=list(map(str,[int(new_x_left), int(new_y_left), int(new_x_right), int(new_y_right)]))
                    else:
                        max_key = max(list(interpolated_predict[frame_number+i].keys()))
                        interpolated_predict[frame_number+i][max_key+1] = copy.deepcopy(interpolated_predict[frame_number+i-1][dict_j[key][frame_number+i-1]])
                        interpolated_predict[frame_number+i][max_key+1]['coords']=list(map(str,[int(new_x_left), int(new_y_left), int(new_x_right), int(new_y_right)]))
                        dict_j[key][frame_number+i] = max_key+1
                else:
                    interpolated_predict[frame_number+i] = {}
                    interpolated_predict[frame_number+i][0] = copy.deepcopy(interpolated_predict[frame_number+i-1][dict_j[key][frame_number+i-1]])
                    interpolated_predict[frame_number+i][0]['coords']=list(map(str,[int(new_x_left), int(new_y_left), int(new_x_right), int(new_y_right)]))
                    dict_j[key][frame_number+i] = 0
            
            if kb == len(key_boxes)-2:
                frame_number = key_boxes[kb+1][4]
                new_x_left = (start_cx + dx*frame_count)-(width*(side_change**frame_count))/2
                new_y_left = (start_cy + dy*frame_count)-(height*(side_change**frame_count))/2
                new_x_right = (start_cx + dx*frame_count)+(width*(side_change**frame_count))/2
                new_y_right = (start_cy + dy*frame_count)+(height*(side_change**frame_count))/2
                interpolated_predict[frame_number][dict_j[key][frame_number]]['coords']=list(map(str,[int(new_x_left), int(new_y_left), int(new_x_right), int(new_y_right)]))
    # расчет affect
    for frame in interpolated_predict.keys():
        box_keys = list(interpolated_predict[frame].keys())
        boxes = []
        for box_key in box_keys:
            boxes.append(interpolated_predict[frame][box_key]['coords'])
        affect_list = affect(boxes, frame_width)
        for i, box_key in enumerate(box_keys):
            interpolated_predict[frame][box_key]['affect'] = str(affect_list[i])
    
    return interpolated_predict


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

    all_txt = list(sorted(os.listdir(path_to_txt)))
    txt = [t for t in all_txt if video_name in t]
    dict_predictions = {}
    colors = {2: "red", 1: "yellow", 0: "green"}

    for t in txt:
        i = int(t[len(video_name)+1:-4])
        with open(path_to_txt+t) as f:
            boxes = f.readlines()

            boxes = np.array([list(map(float,box[:-1].split())) for box in boxes])

            dict_predictions[i] = {}
            for j, box in enumerate(boxes):
                box[0] = int(box[0])

                width = box[3]*frame_width
                height = box[4]*frame_height

                x_left = box[1]*frame_width-width/2
                y_left = box[2]*frame_height-height/2
                x_right = box[1]*frame_width+width/2
                y_right = box[2]*frame_height+height/2

                dict_predictions[i][j] = {}
                curr_box = list(map(str,[int(x_left), int(y_left), int(x_right), int(y_right)]))

                dict_predictions[i][j]["coords"] = curr_box

                color = box[0]
                if color not in colors.keys():
                    dict_predictions[i][j]["state"] = "unknown"
                else:
                    dict_predictions[i][j]["state"] = colors[color]
    
    final_predictions = interpolation(dict_predictions, frame_width, frame_height)
    
    with open(json_path, 'w') as outfile:
        json.dump(final_predictions, outfile)
    return final_predictions


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
            if affect == 'True':
                text_color = tuple([70 if c==255 else c for c in color])
                frame = cv.putText(frame, 'affect', (box[0], box[3]+20), fontFace = cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale = 0.6, thickness = 2, color = text_color)

        out.write(frame)
        i += 1
        
    out.release()


# ## Пример использования

# In[ ]:


predictions = txt_to_json('video_3', 'labels/', 'phase_1/video_3.MP4', 'video_3_json_interpolation.txt')


# In[ ]:


video_display("phase_1/video_3.MP4", "phase_1/video_3_boxes.MP4", "video_3_json_interpolation", fps = 30)

