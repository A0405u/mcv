import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
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

    print("model loaded in {}s".format(round(time.time() - timest, 2)))

    return model


# Load images

def load_images(images_path):

    print("loading images...")

    timest = time.time()

    images = []
    paths = os.listdir(images_path)

    print(paths)

    for path in paths:
        image = Image.open(images_path + path)
        images.append(image)

    print("found {} images".format(len(images)))
    print("images loaded in {}s".format(round(time.time() - timest, 2)))

    return images


# Read classes

def load_classes(path):

    print("loading classes...")

    classes=[]

    with open(path, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            classes.append(row)

    print("found {} classes".format(len(classes)))
    print("classes loaded in {}s".format(round(time.time() - timest, 2)))

    return classes


# Transform image

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])


# Predict image

def predict(image, model, trt):

    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    timest = time.time()
    #output = model_trt(input)
    output = model(input)

    print("processing {}".format(round(time.time() - timest, 2)))

    index = output.data.cpu().numpy().argmax()

    return index


# Process image

def process(image, model, trt):

    fig = plt.figure(figsize=(10, 10))
    sub = fig.add_subplot(1,1,1)

    index = predict(image, model, trt)

    sub.set_title(classes[index])
    plt.axis('off')
    plt.imshow(image)
    plt.savefig('out/' + str(index) + '.png')
    plt.show()



if __name__ == "__main__":

    trt = False

    if len(sys.argv) == 2:
        trt = (sys.argv[1] == "trt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model()

    images = load_images("img/")

    classes = load_classes("classes.csv")

    print("processing images...")

    for i, image in enumerate(images):

        print("{current} of {all}".format(current = i, all = len(images)))

        process(image, model, trt)