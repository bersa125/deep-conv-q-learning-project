
from ai_utils import *

def createSequentialLayer(input): #input = lista di liste dove il primo attributo di ogni sottolista e' una stringa che identifica il tipo di operazione richiesta (Ex: per una Conv2d abbiamo Conv2d-1)
    output=nn.Sequential()
    for module in input:
        toAdd=None

        """ Espandibile a piacimento"""
        if module[0].startswith(("Conv2d","conv2d")):
            if(len(module)==6):#with padding
                toAdd=nn.Conv2d(in_channels=module[1],out_channels=module[2],kernel_size=module[3],stride=module[4],padding=module[5])
            if(len(module)==5):#with stride
                toAdd=nn.Conv2d(in_channels=module[1],out_channels=module[2],kernel_size=module[3],stride=module[4])
            if(len(module)==4):#minimal
                toAdd=nn.Conv2d(in_channels=module[1],out_channels=module[2],kernel_size=module[3])
        elif module[0].startswith(("ReLU","relu")):
            toAdd=nn.ReLU()
        elif module[0].startswith(("Tanh","tanh")):
            toAdd=nn.Tanh()
        elif module[0].startswith(("MaxPool2d","maxpool2d")):
            if(len(module)==4):#with padding
                toAdd=nn.MaxPool2d(kernel_size=module[1], stride=module[2], padding=module[3])
            if(len(module)==3):#with stride
                toAdd=nn.MaxPool2d(kernel_size=module[1], stride=module[2])
            if(len(module)==2):#minimal
                toAdd=nn.MaxPool2d(kernel_size=module[1])
        elif module[0].startswith(("Linear","linear")):
            if(module[1]==-1):#Caso in cui va richiamato count_neurons => in module[3] va immessa la serie di Sequential della convolution e in module[4] va immessa img_dim
                dummyCnn= CNN(module[3],[])
                toAdd=nn.Linear(in_features=dummyCnn.count_neurons(module[4]), out_features=module[2])
            else:
                toAdd=nn.Linear(in_features=module[1], out_features=module[2])
        elif module[0].startswith( ("Flatten","flatten") ):#classe custom per il Flattening
            toAdd=Flatten()

        if(toAdd!=None):
            output.add_module(name=module[0],module=toAdd)
    return output

def makeNN(conv_inputs, fuc_inputs, img_dim):#fuc_inputs inizialmente contiene liste composte da solo due componenti che vengono arricchite successivamente dove necessario
    conv_sequentials=[]
    fuc_sequentials=[]
    for conv in conv_inputs:
        conv_sequentials.append(createSequentialLayer(conv))
    for fuc in fuc_inputs:
        for module in fuc:
            if module[0].startswith(("Linear","linear")):
                if(module[1]==-1):#Caso di ricalcolo delle in_features trovato e corretto automaticamente
                    module.append(conv_sequentials)
                    module.append(img_dim)
        fuc_sequentials.append(createSequentialLayer(fuc))
    return CNN(conv_sequentials,fuc_sequentials)
