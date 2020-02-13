import cv2
import time
import torch
from PIL import Image
from torchvision import transforms
from os import listdir
from os.path import isfile, join
import concurrent.futures as futures
from Siamese_MobileNetV2 import Siamese_MobileNetV2

input_width = 224
input_height = 224
face_cascade = cv2.CascadeClassifier('../utils/haarcascade_frontalface_default.xml')
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model = Siamese_MobileNetV2()
model.eval()
images_dir = '/Users/vincentchooi/documents/alarm clock/faces'
img_list = [join(images_dir, f) for f in listdir(images_dir)]

def preprocessAndInferSingle(ind):
    img = img_list[ind]
    init_time = time.time()

    img = cv2.imread(img)
    img = cv2.resize(img, (320, 240))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        w = max(w, h)
        h = w #SQUARE-IFY
        img = img[y:y+h, x:x+w] #CROP IMAGE
        img = cv2.resize(img, (input_width, input_height))
        input_img = Image.fromarray(img) #Need to transpose?
        input_tensor = preprocess(input_img).unsqueeze(0) #(1, 3, 224, 224)
        print(input_tensor.shape)

        with torch.no_grad():
            output = model(input_tensor)

    print('FPS: {}'.format(1/(time.time() - init_time)))

def preprocessAndInferBatch(ind):
    for img in img_list[:ind]:
        init_time = time.time()

        img = cv2.imread(img)
        img = cv2.resize(img, (320, 240))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            w = max(w, h)
            h = w #SQUARE-IFY
            img = img[y:y+h, x:x+w] #CROP IMAGE
            img = cv2.resize(img, (input_width, input_height))
            input_img = Image.fromarray(img) #Need to transpose?
            input_tensor = preprocess(input_img).unsqueeze(0) #(1, 3, 224, 224)

            with torch.no_grad():
                output = model(input_tensor)

        print('FPS: {}'.format(1/(time.time() - init_time)))

#Test multi-processing (doesn't work if Test Synchronous is run first)
print('Test multi-processing')
t1 = time.perf_counter()

with futures.ProcessPoolExecutor() as executor:
    executor.map(preprocessAndInferSingle, [i for i in range(20)])

t2 = time.perf_counter()
print('Time taken: {}s'.format(t2 - t1))

#Test synchronous
print('Test synchronous')
t1 = time.perf_counter()

for i in range(20):
    preprocessAndInferSingle(i)

t2 = time.perf_counter()
print('Time taken: {}s'.format(t2 - t1))