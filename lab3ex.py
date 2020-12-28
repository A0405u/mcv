import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
from torchvision.models.alexnet import alexnet
from torch.utils.data.sampler import SubsetRandomSampler
import csv
import time
from torch2trt import torch2trt

images=[]

#parameter trt or not
trt = False

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

#choose images
img_paths = ['./data/2_big.jpg', './data/banana_0.jpg', './data/brown_bear.jpg', './data/cat_0.jpg', './data/fruit_18.jpg',
             './data/polar_bear.jpg', './data/strawberry_0.jpg']
for i in img_paths:
    image = Image.open(i)
    images.append(image)

#Transforms image for model input
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model loading
print('Model now loading')
timest = time.time()
model = alexnet(pretrained=True).eval().cuda()
print("elapsed model loading time: {}".format(time.time()-timest))

#torch to tensorRT
if trt:
    print("TRT")
    timest = time.time()
    model_trt = torch2trt(model, [x])
    print("torch2trt time: {}".format(time.time()-timest))

#Read classes because our model is pretrained
classes=[]
with open('imagenet.txt', 'r') as fd:
    reader = csv.reader(fd)
    for row in reader:
        classes.append(row)

#Transforming image using test_transforms and predicting class using model
def predict_image(image):
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

#process our images
def processing(images):
    
    for image in images:
        fig = plt.figure(figsize=(10, 10))
        sub = fig.add_subplot(1,1,1)
        index = predict_image(image)
        sub.set_title("class " + str(classes[index]))
        plt.axis('off')
        plt.imshow(image)
        plt.savefig('./output/'+str(index)+'.jpg')
        plt.show()

processing(images)
