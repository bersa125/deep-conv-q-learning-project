import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from configuration_paramaters import RESULTS_DIR

#Dynamic chart generation class
class PlotDynamicUpdate():
    active=True

    def define_labels(self,labels, secondline=False):
        self.x_label=labels[0]
        self.y_label=labels[1]
        self.title=labels[2]
        self.second_y_label=labels[3]
        self.secondLine=secondline
    def on_launch(self):
        #Set up plot
        plt.ion()
        self.previous_backend=plt.get_backend()
        plt.switch_backend("TkAgg")
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], 'r-')
        #Autoscale
        self.ax.set_autoscaley_on(True)
        self.ax.set_autoscalex_on(True)
        #Other stuff
        self.ax.grid()
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title)
        if(self.secondLine):
            self.ax.set_ylabel(self.y_label,color="tab:red")
            self.ax.tick_params(axis='y', labelcolor="tab:red")
            self.ax2=self.ax.twinx()
            self.ax2.set_ylabel(self.second_y_label,color="tab:blue")
            self.lines2=self.ax2.plot([],[], 'b-')
            self.ax2.tick_params(axis='y', labelcolor="tab:blue")


    def on_running(self, xdata, ydata, y2data):
        if(self.active):
            #Update data (with the new _and_ the old points)
            self.lines.set_xdata(xdata)
            self.lines.set_ydata(ydata)
            #Need both of these in order to rescale
            self.ax.relim()
            self.ax.autoscale_view()
            if(self.secondLine):
                self.lines2[0].set_xdata(xdata)
                self.lines2[0].set_ydata(y2data)
                self.ax2.relim()
                self.ax2.autoscale_view()
            #We need to draw *and* flush
            try:
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()
            except:
                self.active=False

    def on_close(self,rewards,losses=[]):
        plt.ioff()
        plt.close()
        plt.switch_backend(self.previous_backend)
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot(range(1, len(rewards)+1),rewards, 'r-')
        #Autoscale
        self.ax.set_autoscaley_on(True)
        self.ax.set_autoscalex_on(True)
        #Other stuff
        self.ax.grid()
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title)
        if(self.secondLine):
            self.ax.set_ylabel(self.y_label,color="tab:red")
            self.ax.tick_params(axis='y', labelcolor="tab:red")
            self.ax2=self.ax.twinx()
            self.ax2.set_ylabel(self.second_y_label,color="tab:blue")
            self.lines2=self.ax2.plot(range(1, len(losses)+1),losses, 'b-')
            self.ax2.tick_params(axis='y', labelcolor="tab:blue")
        plt.show()

    def on_close_draw(self,rewards,losses=[]):
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot(range(1, len(rewards)+1),rewards, 'r-')
        #Autoscale
        self.ax.set_autoscaley_on(True)
        self.ax.set_autoscalex_on(True)
        #Other stuff
        self.ax.grid()
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title)
        if(self.secondLine):
            self.ax.set_ylabel(self.y_label,color="tab:red")
            self.ax.tick_params(axis='y', labelcolor="tab:red")
            self.ax2=self.ax.twinx()
            self.ax2.set_ylabel(self.second_y_label,color="tab:blue")
            self.lines2=self.ax2.plot(range(1, len(losses)+1),losses, 'b-')
            self.ax2.tick_params(axis='y', labelcolor="tab:blue")
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        plt.savefig(RESULTS_DIR+"/"+self.title+'_results.png')
        plt.close()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Average_movements:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average(self):
        return np.mean(self.list_of_rewards)

class CNN(nn.Module):

    def __init__(self, cnn_layers, dnn_layers):
        super(CNN, self).__init__()
        self.convolution = nn.ModuleList(cnn_layers)
        self.fullyconnected=nn.ModuleList(dnn_layers)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        for layer in self.convolution:
            x = layer(x)
        size =  x.data.view(1, -1).size(1)
        return size

    def forward(self, x):
        for layer in self.convolution:
            x = layer(x)
        for layer in self.fullyconnected:
            x = layer(x)
        return x


# Selector class used to pick an output from a tensor

class SoftmaxSelector(nn.Module):

    def __init__(self, T):
        super(SoftmaxSelector, self).__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T,dim=1)
        actions = probs.multinomial(1)
        return actions


# Puts together the Network and the Final action selector

class AI:

    def __init__(self, nn, selector):
        self.nn = nn
        self.selector = selector

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        output = self.nn(input)
        actions = self.selector(output)
        return actions.data.numpy()
