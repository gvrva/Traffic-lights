#!/usr/bin/env python
# coding: utf-8
#Подготовка данных
# Функция data_loader загружает батчи нужного размера для трейна и теста. Можно регулировать размер батча, трансформы, если они нам понадобятся, и долю тестовой выборки в датасете. Пример использования этой функции в разделе Test. Подключаемый файл data_preparation.py

# Теперь по данным, нужно загрузить датасет LISA в ту же папку, что и ноутбуки все. Скачивается архив, распаковываешь его в папку с названием LISA, которая находится там же, где и ноутбуки.
# ![image.png](attachment:image.png)

# В папке LISA должны лежать вот эти файлы
# ![image.png](attachment:image.png)

# Больше ничего с файлами делать не требуется

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from torchvision import transforms
import torchvision
import json
import albumentations as A


class LISADataset(object):
    def __init__(self, transforms = None):
        # загрузка датасета аннотаций для bounding box'ов
        for i in range(1,14):
            folder = "dayClip"+str(i)
            annotations_path = "LISA/Annotations/Annotations/dayTrain/"+folder+"/frameAnnotationsBOX.csv"
            df_i = pd.read_csv(annotations_path, sep = ';')
    
            images_path = "LISA/dayTrain/dayTrain/"+folder+"/frames"
            imgs_temp = list(sorted(os.listdir(images_path)))
            imgs_i = [imgs_temp[i] for i in df_i["Origin frame number"].unique()]
    
            if i==1:
                self.df = df_i.copy()
                self.imgs = imgs_i
            else:
                self.df = pd.concat([self.df, df_i], ignore_index = True)
                self.imgs = self.imgs + imgs_i
        self.transforms = transforms

    def __getitem__(self, idx):
        # load images
        img_folder = self.imgs[idx][:9]
        if img_folder[-1]=='-':
            img_folder = img_folder[:-1]
            
        img_path = os.path.join("LISA\dayTrain\dayTrain", img_folder, "frames", self.imgs[idx])
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)/255
        
        label_dict = {"go": 3, "warning": 2, "stop": 1, "stopLeft": 1, "goLeft": 3, "warningLeft": 2}

        # get bounding box coordinates for each mask
        frame_df = self.df[self.df["Filename"]=="dayTraining/"+self.imgs[idx]]
        num_objs = len(frame_df)
        boxes = []
        labels = []
        
        x_max = list(frame_df["Upper left corner X"])
        x_min = list(frame_df["Lower right corner X"])
        y_max = list(frame_df["Upper left corner Y"])
        y_min = list(frame_df["Lower right corner Y"])
        light = list(frame_df["Annotation tag"])
        
        for i in range(num_objs):
            boxes.append([x_max[i], y_max[i], x_min[i], y_min[i]])
            labels.append(label_dict[light[i]])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        
        if self.transforms:
        
            sample = {
                'image': img,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            if len(sample['bboxes'])>0:
                img = sample['image']
                target['boxes'] = torch.as_tensor(sample['bboxes'],dtype=torch.int64)
                target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        target['area'] = area
        
        img = torch.tensor(img, dtype=torch.float32)
        img = torch.reshape(img, (3, img.shape[0], img.shape[1]))
        
        return img, target

    def __len__(self):
        return len(self.imgs)



def collate_fn(batch):
    return tuple(zip(*batch))


def getTrainTransform():
    return A.Compose([
        A.RandomCrop(width=300, height=300, p = 0.2),
        A.HorizontalFlip(p=0.5)
    ], bbox_params=A.BboxParams(format = 'pascal_voc', label_fields=['labels'], min_visibility = 0.5))


def data_loader(batch_size, transform = None, test_size = 0.2):
    
    dataset_train = LISADataset(transforms = transform)
    dataset_test = LISADataset()
    indices = torch.randperm(len(dataset_train)).tolist()
    
    t = round(len(dataset_train)*test_size)
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-t])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-t:])
    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    return data_loader_train, data_loader_test
    


#без сглаживания
def video_predict(path, json_path, model, device):
    
    #path - путь к видео
    #json_path - путь, где сохранится json файл в формате "folder(если нужна папка)/file_name.txt"
    #model - модель
    #device - видеокарта
    
    cap = cv.VideoCapture(path) # Вывод с видео файла
    dict_predictions = {}
    model.eval()
    i=0
    colors = {1: "red", 2: "yellow", 3: "green"}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with torch.no_grad():
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)/255
            frame = torch.tensor(frame, dtype=torch.float32)
            frame = [torch.reshape(frame, (3, frame.shape[0], frame.shape[1])).to(device)]
            
            prediction = model(frame)
            dict_predictions[i]=[]
            for j, box in enumerate(prediction[0]['boxes']):
                dict_predictions[i].append(j)
                dict_predictions[i][j] = {}
                curr_box = list(map(str,box.cpu().numpy().astype(np.int32)))
                
                dict_predictions[i][j]["coords"] = curr_box
                
                color = int(prediction[0]['labels'][j].cpu())
                if color not in colors.keys():
                    dict_predictions[i][j]["state"] = "unknown"
                else:
                    dict_predictions[i][j]["state"] = colors[color]
                
                #определить влияние на полосу
        i+=1
        
    with open(json_path, 'w') as outfile:
        json.dump(dict_predictions, outfile)


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
        for tl_dict in data[str(i)]:
            box = tl_dict["coords"]
            boxes.append(list(map(int, box)))
            state = tl_dict["state"]
            if state == 'red':
                colors.append((0,0,255))
            if state == 'yellow':
                colors.append((0,255,255))
            if state == 'green':
                colors.append((0,255,0))
            if state == 'unknown':
                colors.append((255,255,255))
            
        for box, color in zip(boxes, colors):
            
            cv.rectangle(frame, (box[2], box[3]), (box[0], box[1]), color, 3)

        out.write(frame)
        i += 1
        
    out.release()


def crop_image(image, box):
    cropped_image = image[box[1]:box[3], box[0]:box[2], :]
    return cropped_image



#учет положения огонька
def traffic_light_color(image, box):
    traf_light = crop_image(image, box)
    traf_light = cv.cvtColor(traf_light, cv.COLOR_BGR2RGB)
    traf_light_gray = cv.cvtColor(traf_light, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(traf_light_gray,120,255,cv.THRESH_BINARY)
    
    thresh = np.array(thresh)
    
    height = thresh.shape[0]
    width = thresh.shape[1]
    thresh_copy = thresh.copy()
    for i in range(height):
        if sum(thresh[i])>255*width*0.6:
            thresh_copy[i] = thresh_copy[i]*0
    for i in range(width):
        if sum(thresh[:,i])>255*height*0.6:
            thresh_copy[:,i] = thresh_copy[:,i]*0

    cy = int(height/3)
    r = thresh_copy[:cy].sum()
    y = thresh_copy[cy:2*cy].sum()
    g = thresh_copy[2*cy:].sum()
    
    light = [r,y,g]
    color_id = light.index(max(light))
    
#     print(thresh)
#     print(r,y,g)
#     print()
    
    if color_id == 0:
        return 'red'
    if color_id == 1:
        return 'yellow'
    if color_id == 2:
        return 'green'
