# deep-conv-q-learning-project
Deep convolutional Q-Learning project powered by gym

Just modify configuration_parameters.py in order to change the Enviroment, the Learning parameters and some additional Running functions.

## Additional packages to install if you use Google colab
In case you want to use the Super Mario Environment:
```shell
pip install gym-super-mario-bros
```
In case you want to use Dump/Restore function in memory:
```shell
pip install superjson
```
### Remember also
to set in configuration_parameters.py :
```shell
PLOT_CREATION=False
RENDER=False
```
to avoid problems since rendering on colab isn't admitted (creation of the plot takes place silently and in an autonomous way).
