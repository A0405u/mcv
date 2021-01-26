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
import sys
import os


IMAGES_PATH = "img/"
CLASSES_PATH = "classes.txt"
MODEL_TRT_PATH = "alexnet_trt.pt"

trt = False

if len(sys.argv) == 2:
    trt = (sys.argv[1] == "trt")

# Select device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load regular model

if not trt:

    print("loading model...")

    timest = time.time()

    # model = torch.hub.load('pytorch/vision:v0.8.0', 'wide_resnet101_2', pretrained=True).eval().cuda()
    model = alexnet(pretrained = True).eval().cuda()

    print("model loaded in {}s".format(round(time.time() - timest, 3)))

# Load TRT

else:

    print("loding trt model...")
    timesttrt = time.time()

    try: # Load from file
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(MODEL_TRT_PATH))

    except FileNotFoundError: # Convert from regular

        print("converting torch to trt...")

        x = torch.ones((1, 3, 224, 224)).cuda()

        timest = time.time()
        model_trt = torch2trt(alexnet(pretrained = True).eval().cuda(), [x])

        torch.save(model_trt.state_dict(), MODEL_TRT_PATH)

        print("converted in {}s".format(round(time.time() - timest, 3)))

    print("loaded in {}s".format(round(time.time() - timesttrt, 3)))

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

    if trt:
        output = model_trt(input)
    else:
        output = model(input)

    print("image processed in {}s".format(round(time.time() - timest, 3)))

    return output.data.cpu().numpy().argmax()


# Process image

def process(image):

    fig = plt.figure(figsize=(5, 5))
    sub = fig.add_subplot(1,1,1)

    index = predict(image)

    print(classes[index])

    sub.set_title(classes[index])
    plt.axis('off')
    image.thumbnail((256, 256))
    plt.imshow(image)

    if trt:
        plt.savefig('out/' + str(index) + str(time.time()) + 'trt.png')
    else:
        plt.savefig('out/' + str(index) + str(time.time()) + '.png')
    # plt.show()


print("processing images...")

for i, image in enumerate(images):

    print("processing {current} of {all}...".format(current = i + 1, all = len(images)))

    process(image)