
# Importing the libraries
import gym
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import progressbar
import zlib
from torch.autograd import Variable
from frame_skipping import SkipWrapper
import experience_replay, image_preprocessing
from ai_utils import SoftmaxSelector, AI, PlotDynamicUpdate, Average_movements
from configuration_paramaters import *
from experience_replay import Step

def exit_func(plot, memory_buffer, rewards, losses, selected_env, count):
    try:
        if PLOT_CREATION:
            plot.on_close(rewards,losses)
        else:
            print("Silently drawing the plot...")
            plot.on_close_draw(rewards,losses)
            print(" Done!")
        if RESUME_WORK:
            from superjson import json
            if not os.path.exists(RESUME_DIR+"/"+ENV_NAMES[selected_env]):
                os.makedirs(RESUME_DIR+"/"+ENV_NAMES[selected_env])
            print("Dumping the state of memory on files...")
            bar = progressbar.ProgressBar(maxval=len(memory_buffer), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for i in range(len(memory_buffer)):
                file = open(RESUME_DIR+"/"+ENV_NAMES[selected_env]+"/"+ENV_NAMES[selected_env]+"_back_"+str(i)+".dat", 'w')
                buffer_tuple=[]
                for step in memory_buffer[i]:
                    if step.done:
                        buffer_tuple.append(Step(state=step.state, action=step.action, reward=step.reward ,done=1))
                    else:
                        buffer_tuple.append(Step(state=step.state, action=step.action, reward=step.reward ,done=0))
                file.write(zlib.compress(json.dumps(tuple(buffer_tuple))))
                file.close()
                bar.update(i+1)
            bar.finish()
            print(" Done!")
    except KeyboardInterrupt:
        if count>0:
            exit_func(plot, memory_buffer, rewards, losses, selected_env, count - 1)

def env_selector(env_selected, frame_skip, image_dim, moveset=0):
    gs = image_dim[0]==1
    if(env_selected==DOOM_CORRIDOR or env_selected==DOOM_PREDICT_POSITION or env_selected==DOOM_HEALTH_GATHERING):
        from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
        """
        - minimal - Will only use the levels' allowed actions (+ NOOP)
        - constant-7 - Will use the 7 minimum actions (+NOOP) to complete all levels
        - constant-17 - Will use the 17 most common actions (+NOOP) to complete all levels
        - full - Will use all available actions (+ NOOP)
        """
        movesets=["minimal","constant-7","constant-17","full"]
        return image_preprocessing.ImageProcessing(SkipWrapper(frame_skip)(ToDiscrete(movesets[moveset])(gym.make(ENVIRONMENTS[env_selected]))), width = image_dim[1], height = image_dim[2], grayscale = gs)
    elif(env_selected==SUPER_MARIO):
        from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
        """
        - RIGHT_ONLY - actions for the simple run right environment
        - SIMPLE_MOVEMENT - actions for very simple movement
        - COMPLEX_MOVEMENT - actions for more complex movement
        """
        movesets=[RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT]
        return image_preprocessing.ImageProcessing(SkipWrapper(frame_skip)(BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(ENVIRONMENTS[SUPER_MARIO]), movesets[moveset])), width = image_dim[1], height = image_dim[2], grayscale = gs)
    else:
        return image_preprocessing.ImageProcessing(SkipWrapper(frame_skip)(gym.make(ENVIRONMENTS[env_selected])), width = image_dim[1], height = image_dim[2], grayscale = gs)


def asynchronous_n_step_eligibility_batch(batch, cnn):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


def execute(game_env, selected_env, network, epochs, goal_score, learning_rate, batch_size, number_of_photograms,n_steps_ev,memory_capacity):
    print("Using CUDA: "+str(USE_CUDA))
    # Building an AI
    cnn = network
    softmax_body = SoftmaxSelector(T=TEMPERATURE)
    ai = AI(nn=cnn, selector=softmax_body)
    am = Average_movements(epochs)
    # Setting up Experience Replay
    n_steps = experience_replay.NStepProgress(env=game_env, ai=ai, n_step=n_steps_ev)
    if not RENDER:
        n_steps.disable_rendering()
    memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=memory_capacity)
    if RESUME_WORK:
        if os.path.exists(RESUME_DIR+"/"+ENV_NAMES[selected_env]):
            from superjson import json
            print("Retrieving previous state...")
            #Reconstruction of the memory buffer
            bar = progressbar.ProgressBar(maxval=len(os.listdir(RESUME_DIR+"/"+ENV_NAMES[selected_env])), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for i in range(len(os.listdir(RESUME_DIR+"/"+ENV_NAMES[selected_env]))):
                f=ENV_NAMES[selected_env]+"_back_"+str(i)+".dat"
                if os.path.exists(os.path.join(RESUME_DIR+"/"+ENV_NAMES[selected_env], f)):
                    file = open(os.path.join(RESUME_DIR+"/"+ENV_NAMES[selected_env], f), 'r')
                    lists=json.loads(zlib.decompress(file.read()))
                    file.close()
                    save=[]
                    for step in lists:
                        if(step[3]==1):
                            save.append(Step(state =  step[0], action =  step[1], reward =  step[2], done =  True))
                        else:
                            save.append(Step(state =  step[0], action =  step[1], reward =  step[2], done =  False))
                    memory.buffer.append(tuple(save))
                    if len(memory.buffer) > memory.capacity: #Just in case the capacity is changed in the configurations
                        memory.buffer.popleft()
                    bar.update(i+1)
            bar.finish()
            print(" Done!")
    # Training the AI
    loss = nn.MSELoss() #nn.CrossEntropyLoss()
    if(USE_CUDA):
        loss = loss.cuda(torch.cuda.current_device()) #nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    rewards = []
    losses = []

    # Configuring pyplot for realtime drawing of average reward
    plot=PlotDynamicUpdate()
    plot.define_labels(['Epochs','Average Reward',ENVIRONMENTS[selected_env],'Average Loss'],GRAPH_LOSS)
    if PLOT_CREATION:
        plot.on_launch()

    for epoch in range(1, epochs + 1):
        try:
            memory.run_steps(number_of_photograms)
            #average loss values
            batches=0
            avg_loss_value=0
            for batch in memory.sample_batch(batch_size):
                if(USE_CUDA):
                    inputs, targets = asynchronous_n_step_eligibility_batch(batch, cnn)
                    inputs, targets = Variable(inputs), Variable(targets).cuda(torch.cuda.current_device())
                    predictions = cnn(inputs).cuda(torch.cuda.current_device())
                    loss_error = loss(predictions, targets).cuda(torch.cuda.current_device())
                    optimizer.zero_grad()
                    loss_error.backward()
                    optimizer.step()
                    avg_loss_value+=loss_error.item()
                else:
                    inputs, targets = asynchronous_n_step_eligibility_batch(batch, cnn)
                    inputs, targets = Variable(inputs), Variable(targets)
                    predictions = cnn(inputs)
                    loss_error = loss(predictions, targets)
                    optimizer.zero_grad()
                    loss_error.backward()
                    optimizer.step()
                    avg_loss_value+=loss_error.item()
                batches+=1
            if(batches>0):
                avg_loss_value=avg_loss_value/batches
            if (avg_loss_value==0.0):
                avg_loss_value=float('NaN')
            rewards_steps = n_steps.rewards_steps()
            am.add(rewards_steps)
            avg_reward = am.average()
            rewards.append(avg_reward)
            losses.append(avg_loss_value)
            print(" Epoch: %s, Average Reward: %s, Environment: %s, Learning rate: %s, Batch size: %s, Photograms: %s" % (str(epoch), str(avg_reward), str(ENVIRONMENTS[selected_env]), str(learning_rate), str(batch_size), str(number_of_photograms)))
            #Update the plot
            if PLOT_CREATION:
                plot.on_running(range(1, epoch + 1),rewards,losses)
            if avg_reward >= goal_score:
                print("Congratulations, your AI wins")
                break
        except KeyboardInterrupt:
            print("Exit signal captured... Starting exit procedure")
            break
    exit_func(plot, memory.buffer, rewards, losses, selected_env, 1)