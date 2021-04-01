# Как использовать обученную yolov5l для своих видео

Для предсказания на пользовательских видео необходимо подключить модель predict.py с помощью 
```python
import predict
```
Таким образом можно использовать функции из этого модуля. Модель yolo для полученных весов строится с помощью команды
```python
model = torch.hub.load('ultralytics/yolov5', 'custom', 'path_to_weight_file', force_reload=True)
```
Пример процесса предсказания и воспроизведения результатов
Таким образом можно использовать функции из этого модуля. Модель yolo для полученных весов строится с помощью команды
```python
import predict
model = torch.hub.load('ultralytics/yolov5', 'custom', 'bosch_2ep.pt', force_reload=True)
predict.video_predict("video_0.MP4", "video_0.json", model)
predict.video_display("video_0.MP4", "video_0_boxes.MP4", "video_0.json")
```
