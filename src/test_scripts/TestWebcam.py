import os
import cv2
import time
import torch
import argparse
from PIL import Image
from torchvision import transforms
from Siamese_MobileNetV2 import Siamese_MobileNetV2

def load(model, weight_fn):

    assert os.path.isfile(weight_fn), "{} is not a file.".format(weight_fn)

    state = torch.load(weight_fn)
    weight = state['weight']
    it = state['iterations']
    model.load_state_dict(weight)
    print("Checkpoint is loaded at {} | Iterations: {}".format(weight_fn, it))

def main(args):
    weight_fn = args.weight_fn

    input_width = 224
    input_height = 224
    face_cascade = cv2.CascadeClassifier('../utils/haarcascade_frontalface_default.xml')
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    model = Siamese_MobileNetV2()
    model.eval()

    load(model, weight_fn)

    cap = cv2.VideoCapture(0)

    # ~30FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240);

    while True:
        init_time = time.time()
        _, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        #for x, y, w, h in faces:
            #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            w = max(w, h)
            h = w #SQUARE-IFY
            img = img[y:y+h, x:x+w] #CROP IMAGE
            img = cv2.resize(img, (input_width, input_height))
            input_img = Image.fromarray(img) #Need to transpose?
            input_tensor = preprocess(input_img).unsqueeze(0) #(1, 3, 224, 224)

            '''
            with torch.no_grad():
                output = model(input_tensor)
            '''

        cv2.imshow('img', img)
        print('FPS: {}'.format(1/(time.time() - init_time)))

        k = cv2.waitKey(30) & 0xff

        if k == 27:
            break

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #FORMAT DIRECTORIES
    parser.add_argument("--weight_fn", type=str, required=True, help="Directory: Weights")
    args = parser.parse_args()

    main(args)