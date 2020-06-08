# test.py

from itertools import product
from typing import Tuple

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch import Tensor
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.cm as cm # type: ignore

CATEGORIES = 3
Color = Tuple[float,float,float,float]

def e_k(k: int, n: int) -> np.array:
    a = np.zeros(n)
    a[k] = 1
    return a

def get_category(x: float, y: float) -> int:
    if np.hypot(x,y) < 1: return 0
    elif x > 0 and y > 0: return 1
    else: return 2

def get_tag(x:float, y:float) -> np.array:
    return e_k(get_category(x,y), CATEGORIES)

def get_color(x: float, y: float) -> Color:
    category = get_category(x,y)
    if category == 0:
        return (1,0,0,1)
    elif category == 1:
        return (0,1,0,1)
    elif category == 2:
        return (0,0,1,1)
    else:
        assert False and "unreachable"

def get_region_color(weights: Tensor) -> Color:
    category = torch.argmax(weights)
    weights = nn.Softmax()(weights)
    #rgb = tuple(i.item() for i in weights[:3])
    import pdb
    #pdb.set_trace()
    rgb = (weights[0].item(),
                 weights[1].item(),
                 weights[2].item(),
                 .5)
    #print(rgb)
    return rgb
    if category == 0:
        return (1,.2,.2,.5)
    elif category == 1:
        return (.2,1,.2,.5)
    elif category == 2:
        return (.2,.2,1,.5)
    else:
        assert False and "unreachable"

def tag_samples(a: np.array) -> np.array:
    return np.array([get_tag(x,y) for x,y in a])

def get_data(n: int = 100) -> DataLoader:
    data = []
    for _ in range(n):
        x = 4*(np.random.random()-.5)
        y = 4*(np.random.random()-.5)
        data.append(np.array([x,y]))
    return np.array(data)

class Classifier(nn.Module):
    feature_map: nn.Linear
    connection: nn.Linear
    projection: nn.Linear
    activation_1: nn.Module
    activation_2: nn.Module
    loss_fn: nn.CrossEntropyLoss

    def __init__(self, 
                 features: int = 100,
                 activation_1: nn.Module = nn.Sigmoid(),
                 activation_2: nn.Module = nn.ELU(),
            ):
        super().__init__()
        self.feature_map = nn.Linear(2, features)
        self.connection = nn.Linear(features, features)
        self.projection = nn.Linear(features, CATEGORIES)
        self.activation_1 = activation_1
        self.activation_2 = activation_2
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs):
        # lift 2-d data to feature space
        x = self.feature_map(inputs)
        x = self.activation_1(x)
        # fully connected layer
        x = self.connection(x)
        x = self.activation_2(x)
        # prepare for probability estimation
        x = self.projection(x)
        return x
    
    def get_loss(self, inputs, categories):
        return self.loss_fn(self(inputs), categories)

    def total_loss(self, dataloader: DataLoader) -> float:
        dataset = dataloader.dataset
        weights = self(torch.from_numpy(dataset).float())
        cats = torch.tensor([get_category(x,y) for x,y in dataset]) # type: ignore
        return torch.sum(self.loss_fn(weights,cats)).item()

fig_dir = 'figs'
fig_count = 0
from pathlib import Path

def save_fig():
    global fig_count
    path = Path() / 'figs' / f'{fig_count:05}.pdf'
    plt.savefig(path)
    fig_count += 1

def display(classifier: Classifier, dataloader: DataLoader):
    data = dataloader.dataset # type: ignore
    X = np.linspace(-4,4,64)
    Y = np.linspace(-4,4,64)
    region_data = []
    for x,y in product(X,Y):
        region_data.append([x,y,get_region_color(classifier(torch.tensor([x,y]).float()))])
    region_array = np.array(region_data)
    region_colors = [sm for _,_,sm in region_array]
    plt.scatter(region_array[:,0], region_array[:,1], c=region_colors) # type: ignore
    data_colors = [get_color(x,y) for x,y in data] # type: ignore
    plt.scatter(data[:,0], data[:,1], c=data_colors, edgecolors='k') # type: ignore
    save_fig()
    #plt.show()

def train(classifier: Classifier, dataloader: DataLoader, epochs: int):
    optimizer = optim.SGD(classifier.parameters(), lr=.00001, momentum=0.0)
    optimizer.zero_grad()
    for i in range(epochs):
        for data in dataloader:
            #data_ = torch.from_numpy(tag_samples(data))
            categories = torch.tensor([get_category(x,y) for x,y in data])
            loss = classifier.get_loss(data.float(), categories)
            loss.backward()
            optimizer.step()
            display(classifier, dataloader)
        print(f'total loss: {classifier.total_loss(dataloader):8.4f}')
        #if i % 50 == 0:
            #display(classifier, dataloader)
    display(classifier, dataloader)

if __name__ == '__main__':
    data = get_data(100)
    dataloader = DataLoader(data, batch_size=10) # type: ignore
    classifier = Classifier()
    #display(classifier, dataloader)
    train(classifier, dataloader, 1000)
