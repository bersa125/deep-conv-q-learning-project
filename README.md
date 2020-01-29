# Deep-conv-q-learning-project
Deep convolutional Q-Learning project powered by gym.

The project provides a customizable AI that applies n-step Q-Learning to Open-ai gym environments. 

This work is an extension and an improvement of the now deprecated files provided by [Achronus for the Doom gym environment](https://github.com/Achronus/Machine-Learning-101/tree/master/coding_templates_and_data_files/artificial_intelligence/1.%20deep_convolutional_q_learning).

In its actual state the software its runs on Python 2.7 and it is designed to works with some of old Atari games, [gym-doom](https://github.com/ppaquette/gym-doom) and [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros). In particular the last two environment packages are not required in order to start the program.

# Required packages

Launch these commands in your Python environment (2.7) in order to install the required packages:

```shell
pip install gym
pip install Pillow
pip install scipy
pip install numpy
pip install pytorch
pip install superjson
```
The following software have also to be installed on your environment:
* boost, 
* boost-python, 
* torch, 
* torchvision, 
* cudatoolkit, 
* cudann. 

## Atari games:
```shell
pip install gym[atari]
```
## Doom:
```shell
pip install ppaquette-gym-doom
```

## Super Mario:
```shell
pip install gym-super-mario-bros
#After 7.0.1 nes-py lost some core functions used by the algorithms
pip install nes-py==7.0.1
```

# Configuration
Just modify configuration_parameters.py in order to change the Enviroment, the Learning parameters and some additional Running functions.

If you want to change the CNNs consider the default-nns.py file and follow the translations as described in the custom-nns.py file.

# Launch
Simply launch the main.py file using python in a terminal opened in the project directory:
```shell
python main.py
```
# Google Colab execution 

The program can work also on Google Colab enviroments.

Remember to set in configuration_parameters.py :
```python
PLOT_CREATION=False
RENDER=False
```
to avoid problems as rendering isn't admitted (creation of the plot takes place silently and in an autonomous way).

# Authors

* **Federico Alfano**  - [federicoalfano](https://github.com/federicoalfano/)
* **Luca Vargiu** - [bersa125](https://github.com/bersa125)
