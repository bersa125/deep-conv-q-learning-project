import torch

#Constants used inside the application:
#-Color dimension
GRAYSCALE = 1
COLORED = 3
#-Environment variables:
DOOM_CORRIDOR = 0
DOOM_PREDICT_POSITION = 1
DOOM_HEALTH_GATHERING = 2
ATARI_CENTIPEDE = 3
ATARI_CHOPPER = 4
ATARI_GRAVITAR = 5
ATARI_MSPACMAN = 6
ATARI_JAMESBOND = 7
ATARI_POOYAN = 8
ATARI_FISHING = 9
SUPER_MARIO = 10
#-Environmnents' lists
ENVIRONMENTS = ['ppaquette/DoomCorridor-v0','ppaquette/DoomPredictPosition-v0','ppaquette/DoomHealthGathering-v0', 'CentipedeNoFrameskip-v0', 'ChopperCommandNoFrameskip-v0', 'GravitarNoFrameskip-v0', 'MsPacmanNoFrameskip-v0', 'JamesbondNoFrameskip-v0', 'PooyanNoFrameskip-v0', 'FishingDerbyNoFrameskip-v0' ,'SuperMarioBros-v0']
ENV_NAMES=['DoomCorridor','DoomPredictPosition','DoomHealthGathering','Centipede', 'ChopperCommand','Gravitar','MsPacman','Jamesbond','Pooyan','FishingDerby','SuperMarioBros']
#-Working directories
RESUME_DIR="./resume"
PARAMETERS_DIR="./parameters"
RESULTS_DIR="./results"
#-Automatic CUDA selection
USE_CUDA=torch.cuda.is_available() # change to False only if CUDA is available but isn't needed

#Enviroment configuration:
#-To be always set
ENV_SELECTED = SUPER_MARIO
EPOCHS = 100
MOVE_SET=2  # mario = 0-2 ; doom = 0-3 ; otherwise isn't important
GOAL_SCORE = 32000
#-To be set only if PARAMETER_SELECTION is True
BATCH_SIZE = 256
NUMBER_OF_PHOTOGRAMS= 200
LEARNING_RATE = 0.0001
FRAME_SKIP=4
NSTEP_EVALUATION=10
MEMORY_CAPACITY=10000
image_dim = (GRAYSCALE, 84, 84)
#-To be set only if PARAMETER_SELECTION is False
#   Autotuning Options:
EPOCHS_POOL=[30] # Fixed on low number of epochs
LEARNING_POOL=[0.001]# 0.0001 , 0.001, and also 0.01 and 0.1 but with mostly bad results
BATCH_POOL=[128,256]# 256,128 or even 64
NUMBER_OF_PHOTOGRAMS_POOL=[200,50]# 100, 200 or possibly 50
IMAGE_PASSED_POOL=[(COLORED, 84, 84), (GRAYSCALE, 84, 84)]# (GRAYSCALE, 80, 80),(GRAYSCALE, 84, 84),(COLORED, 80, 80),(COLORED, 84, 84) can be tried
FRAME_SKIP_POOL=[0,2,4]# 2, 4 or 6
DEFAULT_CNN_POOL=[3]# 0,1,2,3 corresponding to cnns 1,2,3,4 in default_nns.py
NSTEP_EVALUATION_POOL=[20,30,40]# 10, 20 or more
MEMORY_CAPACITY_POOL=[10000]# 10000 or more
EPOCH_NULL_STRICTNESS=15 #Number of epochs in which the average must rise from null (5 is a good value for most of the enviroments in order to detect the best combinations in a faster way)
#-Used during the learning phase
TEMPERATURE=1.0

#Execution Controls:
RESUME_WORK=True #Do you want to save the memory state?
GRAPH_LOSS=True #Do you want to draw the average loss on Graph?
PLOT_CREATION=False #Do you want to create the plot in real time?
RENDER=False #Do you want to render the game?
PARAMETER_SELECTION=False #Do you want to manually choose some Enviroment configs?








