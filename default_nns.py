from custom_nns import *

def default_cnn1(number_actions, image_dim):
    convolution1 = [
        ["Conv2d-1",image_dim[0],32,5],
        ["ReLU-1"],
        ["MaxPool2d-1",3,2]
    ]
    convolution2 = [
        ["Conv2d-2",32,32,3],
        ["ReLU-2"],
        ["MaxPool2d-2",3,2]
    ]
    convolution3 = [
        ["Conv2d-3",32,64,2],
        ["ReLU-3"],
        ["MaxPool2d-3",3,2]
    ]
    fc1 = [
        ["Flatten-1"],
        ["Linear-1",-1,40],
        ["ReLU-1"]
    ]
    fc2 = [
        ["Linear-2",40,number_actions],
    ]
    return makeNN([convolution1, convolution2, convolution3], [fc1,fc2], image_dim)


def default_cnn2(number_actions, image_dim):
    convolution1 = [
        ["Conv2d-1",image_dim[0],16,4,4],
        ["Tanh-1"],
        ["MaxPool2d-1",3,2]
    ]
    convolution2 = [
        ["Conv2d-2",16,32,2,2],
        ["Tanh-2"],
        ["MaxPool2d-2",3,2]
    ]
    fc1 = [
        ["Flatten-1"],
        ["Linear-1",-1,256],
        ["ReLU-1"]
    ]
    fc2 = [
        ["Linear-2",256,number_actions],
    ]
    return makeNN([convolution1, convolution2], [fc1,fc2], image_dim)

def default_cnn3(number_actions, image_dim): #preferred by Mario
    convolution1 = [
        ["Conv2d-1",image_dim[0],16,4,4],
        ["Tanh-1"]
    ]
    convolution2 = [
        ["Conv2d-2",16,32,2,2],
        ["Tanh-2"],
        ["MaxPool2d-2",3,2]
    ]
    fc1 = [
        ["Flatten-1"],
        ["Linear-1",-1,256],
        ["ReLU-1"]
    ]
    fc2 = [
        ["Linear-2",256,number_actions],
    ]
    return makeNN([convolution1, convolution2], [fc1,fc2], image_dim)

#DeepMinds Q-Network Architecture
def default_cnn4(number_actions, image_dim):
    convolution1 = [
        ["Conv2d-1",image_dim[0],32,8,4],
        ["ReLU-1"],
        ["MaxPool2d- 1",4,2]
    ]
    convolution2 = [
        ["Conv2d-2",32,64,4,2],
        ["ReLU-2"],
        ["MaxPool2d-2",2,2]
    ]
    convolution3 = [
        ["Conv2d-3",64,64,1],
        ["ReLU-3"],
    ]
    fc1 = [
        ["Flatten-1"],
        ["Linear-1",-1,512],
        ["ReLU-1"]
    ]
    fc2 = [
        ["Linear-2",512,number_actions],
    ]
    return makeNN([convolution1, convolution2,convolution3], [fc1,fc2], image_dim)