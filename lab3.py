import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import time

# Create regular pytorch model

def load_model()

    timest = time.time()
    model = alexnet(pretrained = True).eval().cuda()
    print("load time {}".format(time.time() - timest))

    return model


# Load images

def load_images(images_path)

    images = []
    paths = os.listdir(images_path)

    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        images.append(image)

    return images


# Read classes

# classes=[]

# with open('imagenet.txt', 'r') as fd:
#     reader = csv.reader(fd)
#     for row in reader:
#         classes.append(row)


# Transform image

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Predict image

def predict(image, model, trt):

    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    timest = time.time()
    #output = model_trt(input)
    output = model(input)
    print("processing {}".format(time.time()-timest))
    index = output.data.cpu().numpy().argmax()

    return index


# Process image

def process(image, model, trt):

    fig = plt.figure(figsize=(10, 10))
    sub = fig.add_subplot(1,1,1)

    index = predict(image)

    sub.set_title("class " + str(classes[index]))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig('./output/'+str(index)+'.jpg')
    plt.show()



if __name__ == "__main__":

    trt = False

    if sys.argv[1]:
        trt = (sys.argv[1] == "trt")

    model = load_model()

    images = load_images("img/")

    for image in images:
        process(image, model, trt)