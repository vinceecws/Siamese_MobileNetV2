import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from Faces import Faces
import cv2

faces = Faces('./Faces', (224,224), pair=True, crop_face=True)

fig = plt.figure(figsize=(6,2))

'''
for group in range(450//6):
    for idx in range(6):
        image, label, image2, label2 = faces[group*6 + idx]
        image = image.permute(1,2,0).numpy()
        image2 = image2.permute(1,2,0).numpy()
        plt.subplot(121)
        plt.xlabel(label)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.xlabel(label2)
        plt.imshow(image2)
        

    plt.show()
    cv2.waitKey()
    k = cv2.waitKey(30)
'''

for idx in range(len(faces)):
    image, label, image2, label2 = faces[idx]
    image = image.permute(1,2,0).numpy()
    image2 = image2.permute(1,2,0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    cv2.imshow('{}'.format(label), image)
    k = cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.imshow('{}'.format(label2), image2)
    k = cv2.waitKey(1000)
    cv2.destroyAllWindows()

