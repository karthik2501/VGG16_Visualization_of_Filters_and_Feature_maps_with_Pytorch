import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms,utils
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image_path = 'imagenet-sample-images-master/n03028079_church.JPEG'
img = Image.open(image_path)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.to(device)

outputs = []
outputs.append(img_tensor)


vgg16 = models.vgg16(weights = "DEFAULT")
vgg16.to(device)
vgg16.eval()


model_weights =[]

model_children = list(vgg16.children())
counter = 0
for child in model_children[0].children():
    img_tensor = child(img_tensor)
    if type(child) == nn.Conv2d:
        counter+=1
        model_weights.append(child.weight)
        outputs.append(img_tensor)


plt.figure(figsize=(40,20))
layer_no = 0
n_filters = 5
for i in range(n_filters):
    filter = model_weights[layer_no][i]
    channels = filter.size(0)
    for c in range(channels):
        plt.subplot(n_filters,channels,i*channels+c+1)
        plt.imshow(filter[c].detach().cpu().numpy(),cmap='gray')
        plt.axis('off')

plt.show()

