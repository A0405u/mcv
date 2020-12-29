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


IMAGES_PATH = "img/"
CLASSES_PATH = "classes.txt"

trt = False


# Select device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create regular pytorch model

print("loading model...")

timest = time.time()

# model = torch.hub.load('pytorch/vision:v0.8.0', 'wide_resnet101_2', pretrained=True).eval().cuda()
model = alexnet(pretrained = True).eval().cuda()

print("model loaded in {}s".format(round(time.time() - timest, 2)))


# Load images

print("loading images...")

timest = time.time()

images = []

paths = os.listdir(IMAGES_PATH)

for path in paths:
    image = Image.open(IMAGES_PATH + path)
    images.append(image)

print("found {} images".format(len(images)))
print("images loaded in {}s".format(round(time.time() - timest, 3)))


# Read classes

print("loading classes...")

timest = time.time()

classes=[]

file = open(CLASSES_PATH)

for line in file:
    classes.append(line)

file.close()

print("found {} classes".format(len(classes)))
print("classes loaded in {}s".format(round(time.time() - timest, 3)))


# Transform image

image_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])


# Predict image

def predict(image):

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

def process(image):

    fig = plt.figure(figsize=(10, 10))
    sub = fig.add_subplot(1,1,1)

    index = predict(image)

    print(classes[index])

    sub.set_title(classes[index])
    plt.axis('off')
    image.thumbnail((320, 240))
    plt.imshow(image)
    plt.savefig('out/' + str(index) + '.png')
    # plt.show()



if __name__ == "__main__":

    if len(sys.argv) == 2:
        trt = (sys.argv[1] == "trt")

    print("processing images...")

    for i, image in enumerate(images):

        print("processing {current} of {all}...".format(current = i + 1, all = len(images)))

        process(image)