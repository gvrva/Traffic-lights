#!/usr/bin/env python
# coding: utf-8

# # Подготовка данных

# Функция data_loader загружает батчи нужного размера для трейна и теста. Можно регулировать размер батча, трансформы, если они нам понадобятся, и долю тестовой выборки в датасете. Пример использования этой функции в разделе Test. Подключаемый файл data_preparation.py
# 
# Подключение файла:
# - import data_preparation

# Теперь по данным, нужно загрузить датасет LISA в ту же папку, что и ноутбуки все. Скачивается архив, распаковываешь его в папку с названием LISA, которая находится там же, где и ноутбуки.
# ![image.png](attachment:image.png)

# В папке LISA должны лежать вот эти файлы
# ![image.png](attachment:image.png)

# Больше ничего с файлами делать не требуется

# In[1]:


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from torchvision import transforms
import torchvision
import json


# In[19]:


cap = cv.VideoCapture("phase_1/video_0.MP4") # Вывод с видео файла
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(length)
print("width",width)
print("height",height)


# In[3]:


print(torch.__version__)


# ## Создание генератора датасета

# ### Создание класса датасета

# In[4]:


class LISADataset(object):
    def __init__(self):
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

    def __getitem__(self, idx):
        # load images
        print(idx)
        img_folder = self.imgs[idx][:9]
        if img_folder[-1]=='-':
            img_folder = img_folder[:-1]
        print(self.imgs[idx])
            
        img_path = os.path.join("LISA\dayTrain\dayTrain", img_folder, "frames", self.imgs[idx])
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)/255

        # get bounding box coordinates for each mask
        num_objs = len(self.df[self.df["Filename"]=="dayTraining/"+self.imgs[idx]])
        boxes = []
        for i in range(num_objs):
            x_left = list(self.df[self.df["Filename"]=="dayTraining/"+self.imgs[idx]]["Upper left corner X"])[i]
            x_right = list(self.df[self.df["Filename"]=="dayTraining/"+self.imgs[idx]]["Lower right corner X"])[i]
            y_left = list(self.df[self.df["Filename"]=="dayTraining/"+self.imgs[idx]]["Upper left corner Y"])[i]
            y_right = list(self.df[self.df["Filename"]=="dayTraining/"+self.imgs[idx]]["Lower right corner Y"])[i]
            boxes.append([x_left, y_left, x_right, y_right])
        print(boxes)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        
        img = torch.tensor(img, dtype=torch.float32)
        return img, target

    def __len__(self):
        return len(self.imgs)


# In[5]:


def collate_fn(batch):
    return tuple(zip(*batch))


# In[6]:


def data_loader(batch_size, transform = None, test_size = 0.2):
    
    dataset_train = LISADataset()
    dataset_test = LISADataset()
    indices = torch.randperm(len(dataset_train)).tolist()
    
    t = round(len(dataset_train)*test_size)
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-t])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-t:])
    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    return data_loader_train, data_loader_test
    


# # Предсказание для видео

# In[32]:


def video_predict(path, model):
    cap = cv.VideoCapture(path) # Вывод с видео файла
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    dict_predictions = {}
    model.eval()
    for i in range(3):
        ret, frame = cap.read()
        with torch.no_grad():
            frame = torch.tensor(frame, dtype=torch.float32)
            frame = [torch.reshape(frame, (3, frame.shape[0], frame.shape[1])).to(device)]
            prediction = model(frame)
            dict_predictions[i]=[]
            for j, box in enumerate(prediction[0]['boxes']):
                dict_predictions[i].append({j:list(map(str,box.cpu().numpy().astype(np.int32)))})
    json_file_name = 'test_pred_'+path[8:15]+'.txt'
    with open(json_file_name, 'w') as outfile:
        json.dump(dict_predictions, outfile)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Test

# Тут я тестирую созданные функции, можно не обращать внимание

# In[6]:


torch.cuda.empty_cache()


# In[7]:


device = torch.device('cuda:0')


# In[8]:


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = LISADataset()
train, test = data_loader(batch_size = 2)
# For Training
images,targets = next(iter(train))
images = list(torch.reshape(image, (3, image.shape[0], image.shape[1])).to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
model.to(device)
model.train()


# ## Предсказания для видео

# In[32]:


def video_predict(path, model):
    cap = cv.VideoCapture(path) # Вывод с видео файла
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    dict_predictions = {}
    model.eval()
    for i in range(3):
        ret, frame = cap.read()
        with torch.no_grad():
            frame = torch.tensor(frame, dtype=torch.float32)
            frame = [torch.reshape(frame, (3, frame.shape[0], frame.shape[1])).to(device)]
            prediction = model(frame)
            dict_predictions[i]=[]
            for j, box in enumerate(prediction[0]['boxes']):
                dict_predictions[i].append({j:list(map(str,box.cpu().numpy().astype(np.int32)))})
    json_file_name = 'test_pred_'+path[8:15]+'.txt'
    with open(json_file_name, 'w') as outfile:
        json.dump(dict_predictions, outfile)


# In[33]:


video_predict("phase_1/video_0.MP4", model) #пример использования


# In[ ]:





# In[ ]:





# In[ ]:


# For inference
model.eval()
i = 0
with torch.no_grad():
    images,targets = next(iter(test))
    images = list(torch.reshape(image, (3, image.shape[0], image.shape[1])).to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    predictions = model(images)           # Returns predictions


# сначала полностью заполняю словарь для ВСЕХ кадров видео, а потом уже записываю его в файл

# In[82]:


print(predictions)


# In[41]:


dict_predictions = {}
dict_predictions[0]=[]
for j, box in enumerate(predictions[0]['boxes']):
    dict_predictions[0].append({j:list(map(str,box.cpu().numpy().astype(np.int32)))})
    # тут добавить определение цвета и влияния светофора позже

with open('test_pred.txt', 'w') as outfile:
    json.dump(dict_predictions, outfile)


# In[77]:


with open('test_pred.txt') as json_file:
    data = json.load(json_file)
    #для одного кадра
    boxes = []
    for box in data[str(0)]:
        value = list(box.values())[0]
        boxes.append(list(map(int, value)))
    displayImage(images[0].cpu().numpy(), boxes, False)


# In[78]:


image = images[0].cpu().numpy()


# In[79]:


def displayImage(image, boxes, to_cpu):
    if to_cpu:
        boxes = boxes.cpu().numpy().astype(np.int32)
    image_temp = np.reshape(image, (image.shape[1],image.shape[2], image.shape[0]))
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv.rectangle(image_temp,
                      (box[2], box[3]),
                      (box[0], box[1]),
                      (220, 0, 0), 2)

    ax.set_axis_off()
    ax.imshow(image_temp)

    plt.show()


# In[81]:


displayImage(image, predictions[0]['boxes'], True)


# # Препроцессинг видео

# In[ ]:


ret, frame = cap.read()


# In[8]:


frame.shape


# In[5]:


cv.imshow("frame", frame)
cv.waitKey()
cv.destroyAllWindows()


# In[ ]:




