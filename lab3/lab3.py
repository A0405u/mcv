import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import csv
import sys
import os

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

    for path in paths:
        image = Image.open(images_path + path)
        images.append(image)

    print("found {} images".format(len(images)))
    print("images loaded in {}s".format(round(time.time() - timest, 3)))

    return images


# Read classes

def load_classes(path):

    print("loading classes...")

    timest = time.time()

    classes=[]

    file = open(path)

    for line in file:
        classes.append(line)

    print("found {} classes".format(len(classes)))
    print("classes loaded in {}s".format(round(time.time() - timest, 3)))

    return classes


# Transform image

image_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])


# Predict image

def predict(image, model, trt):

    timest = time.time()

    image_tensor = image_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    input = Variable(image_tensor)
    input = input.to(device)

    #output = model_trt(input)
    output = model(input)

    print("image processed in {}s".format(round(time.time() - timest, 3)))

    return output.data.cpu().numpy().argmax()


# Process image

def process(image, model, trt):

    fig = plt.figure(figsize=(10, 10))
    sub = fig.add_subplot(1,1,1)

    index = predict(image, model, trt)

    print(index)

    sub.set_title(classes[index])
    plt.axis('off')
    plt.imshow(image)
    plt.savefig('out/' + str(index) + '.png')
    # plt.show()



if __name__ == "__main__":

    trt = False

    if len(sys.argv) == 2:
        trt = (sys.argv[1] == "trt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model()

    images = load_images("img/")

    classes = load_classes("classes.txt")

    print("processing images...")

    for i, image in enumerate(images):

        print("processing {current} of {all}...".format(current = i + 1, all = len(images)))

        process(image, model, trt)