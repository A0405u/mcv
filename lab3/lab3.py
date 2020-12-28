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

def load_model():

    print("loading model...")

    timest = time.time()

    model = alexnet(pretrained = True).eval().cuda()

    print("model load time {}".format(time.time() - timest))

    return model


# Load images

def load_images(images_path):

    print("loading images...")

    timest = time.time()

    images = []
    paths = os.listdir(images_path)

    print(paths)

    for path in paths:
        image = cv2.imread(images_path + path, cv2.IMREAD_COLOR)
        images.append(image)

    print("found {} images".format(len(images)))
    print("images load time {}".format(time.time() - timest))

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

    index = predict(image, model, trt)

    sub.set_title("class " + str(classes[index]))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig('./output/'+str(index)+'.jpg')
    plt.show()



if __name__ == "__main__":

    trt = False

    if len(sys.argv) == 2:
        trt = (sys.argv[1] == "trt")

    model = load_model()

    images = load_images("img/")

    print("processing images...")

    for i, image in enumarate(images):

        print("{current} of {all}".format(current = i, all = len(images)))

        process(image, model, trt)