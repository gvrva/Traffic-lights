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
import albumentations as A


# In[19]:


# cap = cv.VideoCapture("phase_1/video_0.MP4") # Вывод с видео файла
# length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
# width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# print(length)
# print("width",width)
# print("height",height)


# In[2]:


print(torch.__version__)


# ## Создание генератора датасета

# ### Создание класса датасета

# In[55]:


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


# In[56]:


def collate_fn(batch):
    return tuple(zip(*batch))


# In[58]:


def getTrainTransform():
    return A.Compose([
        A.RandomCrop(width=300, height=300, p = 0.2),
        A.HorizontalFlip(p=0.5)
    ], bbox_params=A.BboxParams(format = 'pascal_voc', label_fields=['labels'], min_visibility = 0.5))


# In[59]:


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
    


# # Предсказание для видео

# In[60]:


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


# # Создание видео с box'ами

# In[67]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Test

# Тут я тестирую созданные функции

# In[6]:


torch.cuda.empty_cache()


# In[18]:


device = torch.device('cuda:0')


# ### тут пример того, как преобразовать данные перед передачей в модель

# In[62]:


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

train, test = data_loader(batch_size = 1, getTrainTransform(), test_size = 0.2)

model.to(device)

#пример трейна
model.train()
for data, target in train:
    #Вот эти две строчки - пример того, как нужно преобразовать данные
    data = list(d.to(device) for d in data)
    target = [{k: v.to(device) for k, v in t.items()} for t in target]
    
    
    out = model(data, target)
    break

#пример теста
model.eval()
with torch.no_grad():
    for data, target in test:
        #В тесте то же самое
        data = list(d.to(device) for d in data)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        
        
        out = model(data)
        break


# In[63]:


print(out)


# In[ ]:





# ## Предсказания для видео

# In[64]:


get_ipython().run_cell_magic('time', '', 'video_predict("phase_1/video_0.MP4", "phase_1/video_0.txt", model, device) #пример использования')


# In[66]:


video_display("phase_1/video_0.MP4", "phase_1/video_0_boxes.MP4", "phase_1/video_0.txt", fps = 1)


# In[ ]:





# In[ ]:





# In[ ]:





# ### мой черновик, тут ничего интересного

# In[43]:


display_video("train_video_0.avi", "func_train_video_boxes.avi", "train_true_boxes.txt")


# In[48]:


def crop_image(image, box):
    cropped_image = image[box[1]:box[3], box[0]:box[2], :]
    return cropped_image


# In[115]:


#собираем все вместе
# def traffic_light_color(image, box):
#     traf_light = crop_image(image, box)
#     traf_light = cv.cvtColor(traf_light, cv.COLOR_BGR2RGB)
#     traf_light_gray = cv.cvtColor(traf_light, cv.COLOR_BGR2GRAY)
#     ret,thresh = cv.threshold(traf_light_gray,130,255,cv.THRESH_BINARY)
#     black = np.zeros(thresh.shape, np.uint8)
#     mean_val_out = cv.mean(traf_light,mask = thresh)
    
#     red = (255,0,0)
#     green = (0,255,0)
#     yellow = (255,255,0)

#     red_dist = ((mean_val_out[0]-red[0])**2+(mean_val_out[1]-red[1])**2+(mean_val_out[2]-red[2])**2)**0.5
#     yellow_dist = ((mean_val_out[0]-yellow[0])**2+(mean_val_out[1]-yellow[1])**2+(mean_val_out[2]-yellow[2])**2)**0.5
#     green_dist = ((mean_val_out[0]-green[0])**2+(mean_val_out[1]-green[1])**2+(mean_val_out[2]-green[2])**2)**0.5
    
#     distances = [red_dist, yellow_dist, green_dist]
#     color_id = distances.index(min(distances))
#     if color_id == 0:
#         return 'red'
#     if color_id == 1:
#         return 'yellow'
#     if color_id == 2:
#         return 'green'


# In[49]:


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


# In[ ]:


# #учет положения огонька по контурам
# def traffic_light_color(image, box):
#     traf_light = crop_image(image, box)
#     traf_light = cv.cvtColor(traf_light, cv.COLOR_BGR2RGB)
#     traf_light_gray = cv.cvtColor(traf_light, cv.COLOR_BGR2GRAY)
#     ret,thresh = cv.threshold(traf_light_gray,130,255,cv.THRESH_BINARY)
    
#     thresh = np.array(thresh)
    
#     height = thresh.shape[0]
#     width = thresh.shape[1]
#     thresh_copy = thresh.copy()
#     for i in range(height):
#         if sum(thresh[i])>255*width*0.5:
#             thresh_copy[i] = thresh_copy[i]*0
#     for i in range(width):
#         if sum(thresh[:,i])>255*height*0.5:
#             thresh_copy[:,i] = thresh_copy[:,i]*0

#     cy = int(height/3)
#     r = thresh_copy[:cy].sum()
#     y = thresh_copy[cy:2*cy].sum()
#     g = thresh_copy[2*cy:].sum()
    
#     light = [r,y,g]
#     color_id = light.index(max(light))
    
# #     print(thresh)
# #     print(r,y,g)
# #     print()
    
#     if color_id == 0:
#         return 'red'
#     if color_id == 1:
#         return 'yellow'
#     if color_id == 2:
#         return 'green'


# In[46]:


images_path = "LISA/dayTrain/dayTrain/dayClip1/frames"
imgs = list(sorted(os.listdir(images_path)))
out = cv.VideoWriter('train_videos_tests/clip1.avi',cv.VideoWriter_fourcc(*'DIVX'), 24, (1280, 960))

length = len(imgs)
df = pd.read_csv("LISA/Annotations/Annotations/dayTrain/dayClip1/frameAnnotationsBOX.csv", sep = ';')
dict_box = {}

for i in range(length):
    img = cv.imread("LISA/dayTrain/dayTrain/dayClip1/frames/"+imgs[i])
    out.write(img)
    
    dict_box[i]=[]
    num_objs = len(df[df["Origin frame number"]==i])
    
    x_left = list(df[df["Origin frame number"]==i]["Upper left corner X"])
    x_right = list(df[df["Origin frame number"]==i]["Lower right corner X"])
    y_left = list(df[df["Origin frame number"]==i]["Upper left corner Y"])
    y_right = list(df[df["Origin frame number"]==i]["Lower right corner Y"])
    
#     print(i)
    for j in range(num_objs):
        dict_box[i].append(j)
        dict_box[i][j] = {}
        curr_box = [x_left[j], y_left[j], x_right[j], y_right[j]]
        dict_box[i][j]["coords"] = list(map(str,curr_box))
        
        dict_box[i][j]["state"] = traffic_light_color(img,curr_box)
        
out.release()

    
json_file_name = 'train_videos_tests/clip1_true_boxes.txt'
with open(json_file_name, 'w') as outfile:
    json.dump(dict_box, outfile)


# In[47]:


display_video('train_videos_tests/clip1.avi', 'train_videos_tests/clip1_boxes.avi', 'train_videos_tests/clip1_true_boxes.txt', 10)


# In[46]:


t.shape[0]


# In[126]:


with open('train_videos_tests/clip1_true_boxes.txt') as json_file:
    data = json.load(json_file)
    box_t = list(map(int,data['1704'][1]['coords']))


# In[175]:


box_t


# In[176]:


image_t = cv.imread("LISA/dayTrain/dayTrain/dayClip1/frames/"+imgs[1704])


# In[192]:


plt.imshow(traf_light)


# In[193]:


traf_light = crop_image(image_t, box_t)
traf_light = cv.cvtColor(traf_light, cv.COLOR_BGR2RGB)
traf_light_gray = cv.cvtColor(traf_light, cv.COLOR_RGB2GRAY)
ret,thresh = cv.threshold(traf_light_gray,130,255,cv.THRESH_BINARY)
    
thresh = np.array(thresh)
print(thresh)


# In[184]:


# thresh = np.flip(thresh, axis = 0)


# In[194]:


contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, 4)


# In[195]:


black = np.zeros(thresh.shape,np.uint8)
cnt = contours[0]
contour_img = cv.drawContours(black, [cnt], -1, (255,255,255), -1)
plt.imshow(contour_img,'gray')


# In[196]:


M = cv.moments(cnt)
print(M)


# In[197]:


cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

print(traf_light.shape)
print(cx,cy)


# In[198]:


mean_val = cv.mean(traf_light,mask = contour_img)


# In[199]:


print(mean_val)


# In[202]:


red = (255,0,0)
green = (0,255,0)
yellow = (255,255,0)

red_dist = ((mean_val[0]-red[0])**2+(mean_val[1]-red[1])**2+(mean_val[2]-red[2])**2)**0.5
yellow_dist = ((mean_val[0]-yellow[0])**2+(mean_val[1]-yellow[1])**2+(mean_val[2]-yellow[2])**2)**0.5
green_dist = ((mean_val[0]-green[0])**2+(mean_val[1]-green[1])**2+(mean_val[2]-green[2])**2)**0.5

distances = [red_dist, yellow_dist, green_dist]
print(distances)


# In[ ]:





# In[ ]:





# # Черновики

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




