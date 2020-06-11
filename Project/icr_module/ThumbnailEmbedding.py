import pickle
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from icr_module.CNN import CNN
from icr_module.ImageLoader import ImageFolderWithPaths
from icr_module.ImageUtils import imshow

import os
import glob
import pandas as pd
from PIL import Image

BASE_FOLDER = "D:\\Data\\thumbnails\\thumbnails\\"
BATCH_SIZE = 1
img_dir = "D:\Data\\thumbnails\\thumbnails\**\\*.jpg"  # Enter Directory of all images
data_path = os.path.join(img_dir)
files = glob.glob(data_path)



def get_scaling():
    min_width, min_height = 5000, 5000
    width_threshold, height_threshold = 50, 50  # Images with lower dimensions would be considered 'empty'.
    malformed_images_count = 0
    images_amount = len(files)
    for file in files:
        jpgfile = Image.open(file)
        height, width = jpgfile.size
        if height <= height_threshold or width <= width_threshold:
            malformed_images_count += 1
        else:
            min_width = min(width, min_width)
            min_height = min(height, min_height)

    print(min_width, min_height)
    print('Percentage of corrupted pictures: %.2f%%' % ((malformed_images_count / images_amount) * 100))

    return min_width, min_height


def load_training(root_path, dir, batch_size, kwargs):
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


transform = transforms.Compose(
    [transforms.Resize((150,200)), transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = ImageFolderWithPaths(root="D:\\Data\\thumbnails\\thumbnails\\", transform=transform)

trainloader = DataLoader(dataset=train_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True)


# get some random training images
dataiter = iter(trainloader)
(images, labels), _ = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

net = CNN(batch_size=BATCH_SIZE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

PRINT_EVERY = 1000
# for epoch in range(2):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         (inputs, labels), _ = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#
#         (outputs, embedding) = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % PRINT_EVERY == (PRINT_EVERY-1):    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / PRINT_EVERY))
#             running_loss = 0.0

print('Finished Training')

# Save the model
PATH = './thumbnail_embedding.pth'
# torch.save(net.state_dict(), PATH)


# load
net = CNN(batch_size=BATCH_SIZE)
net.load_state_dict(torch.load(PATH))

thumbnail_to_embedding = {}
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(trainloader):
        (images, labels), filename = data
        (outputs, embedding) = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        thumbnail_to_embedding[filename[0]] = embedding[0].numpy()
        if i % PRINT_EVERY == (PRINT_EVERY - 1):  # print every 2000 mini-batches
            print(i/len(files))


print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

with open('thumbnail_to_embedding.pickle', 'wb') as handle:
    pickle.dump(thumbnail_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)