import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms,utils
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image_path = 'imagenet-sample-images-master/n03218198_dogsled.JPEG'
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

layer_no = 1
feature_maps = outputs[layer_no].squeeze(0)
no_feature_maps = feature_maps.size(0)

plt.figure(figsize=(40,40))
for i in range(no_feature_maps):
    plt.subplot(no_feature_maps//8,8,i+1)
    plt.imshow(feature_maps[i].detach().cpu().numpy(),cmap = 'gray')
    # plt.title(f"{i+1}")
    plt.axis('off')

# plt.tight_layout()
plt.show()
