
from ai import *
from default_nns import *
from ai_utils import Average_movements
import os

def readFromFile(file):
    ret=[]
    #reads if file exists and puts each line in the list
    if os.path.exists(file):
        filer = open(file, 'r')
        for line in filer:
            ret.append(line)
        filer.close()
    return ret
def writeOnFile(list ,file):
    #controls if the path exists and creates it
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    #overwrites the file with new content
    thefile = open(file, 'w')
    for item in list:
        thefile.write("%s\n" % item)
    thefile.close()
    return True

#Used to collect the final reward with selected variables
def diagnostic_execute(game_env, selected_env, network, epochs, learning_rate, batch_size, number_of_photograms,n_steps_ev,memory_capacity):
    cnn = network
    softmax_body = SoftmaxSelector(T=1.0)
    ai = AI(nn=cnn, selector=softmax_body)
    am = Average_movements(100)
    n_steps = experience_replay.NStepProgress(env=game_env, ai=ai, n_step=n_steps_ev)
    n_steps.disable_rendering()
    memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=memory_capacity)
    loss = nn.MSELoss() #nn.CrossEntropyLoss()
    if(USE_CUDA):
        loss = loss.cuda(torch.cuda.current_device()) #nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    ret = -100
    for epoch in range(1, epochs + 1):
        memory.run_steps(number_of_photograms)
        for batch in memory.sample_batch(batch_size):
            if(USE_CUDA):
                inputs, targets = asynchronous_n_step_eligibility_batch(batch, cnn)
                inputs, targets = Variable(inputs), Variable(targets).cuda(torch.cuda.current_device())
                predictions = cnn(inputs).cuda(torch.cuda.current_device())
                loss_error = loss(predictions, targets).cuda(torch.cuda.current_device())#torch.max(targets, 1)[1]
                optimizer.zero_grad()
                loss_error.backward()
                optimizer.step()
            else:
                inputs, targets = asynchronous_n_step_eligibility_batch(batch, cnn)
                inputs, targets = Variable(inputs), Variable(targets)
                predictions = cnn(inputs)
                loss_error = loss(predictions, targets)#torch.max(targets, 1)[1]
                optimizer.zero_grad()
                loss_error.backward()
                optimizer.step()
        rewards_steps = n_steps.rewards_steps()
        am.add(rewards_steps)
        avg_reward = am.average()
        ret=avg_reward
        if(epoch>=EPOCH_NULL_STRICTNESS and str(ret)=='nan'):
            print(" Blocked, no advance...")
            break
        print(" Epoch: %s, Average Reward: %s, Environment: %s, Learning rate: %s, Batch size: %s, Photograms: %s" % (str(epoch), str(avg_reward), str(ENV_NAMES[selected_env]), str(learning_rate), str(batch_size), str(number_of_photograms)))
    game_env.close()
    return ret

def test_parameters(selected_env, using_default_nns=1, custom_nns=[], moves=0):
    print("Entering testing phase: "+ENV_NAMES[selected_env])
    print("Number of tests to forward: "+str(len(EPOCHS_POOL)*len(LEARNING_POOL)*len(BATCH_POOL)*len(IMAGE_PASSED_POOL)*len(NUMBER_OF_PHOTOGRAMS_POOL)*len(FRAME_SKIP_POOL)*len(DEFAULT_CNN_POOL)*len(NSTEP_EVALUATION_POOL)*len(MEMORY_CAPACITY_POOL))+" with Average NAN Reward tolerance: "+str(EPOCH_NULL_STRICTNESS))
    reward=[-100]
    if(using_default_nns==1):
        for i in DEFAULT_CNN_POOL:
            for skip in FRAME_SKIP_POOL:
                for image in IMAGE_PASSED_POOL:
                    for learn in LEARNING_POOL:
                        for batch_size in BATCH_POOL:
                            for photogs in NUMBER_OF_PHOTOGRAMS_POOL:
                                for nsteps in NSTEP_EVALUATION_POOL:
                                    for memory in MEMORY_CAPACITY_POOL:
                                        cumulative=0
                                        for epoch in EPOCHS_POOL:
                                            game_env = env_selector(selected_env, skip, image, moveset=moves)
                                            #game_env = gym.wrappers.Monitor(game_env, "videos", force=True, video_callable=lambda episode_id: True)
                                            number_actions = game_env.action_space.n
                                            game_env.reset()
                                            if(i==0):
                                                print("Epochs to Test: %s Frameskip: %s, Image Dimension: %s, Cnn: cnn1, NSteps: %s, Memory dimension: %s" % (str(epoch), str(skip), str(image), str(nsteps), str(memory)))
                                                cumulative+=diagnostic_execute(game_env, selected_env, default_cnn1(number_actions, image), epoch, learn, batch_size, photogs,nsteps,memory)
                                            elif(i==1):
                                                print("Epochs to Test: %s Frameskip: %s, Image Dimension: %s, Cnn: cnn2, NSteps: %s, Memory dimension: %s" % (str(epoch), str(skip), str(image), str(nsteps), str(memory)))
                                                cumulative+=diagnostic_execute(game_env, selected_env, default_cnn2(number_actions, image), epoch, learn, batch_size, photogs,nsteps,memory)
                                            elif(i==2):
                                                print("Epochs to Test: %s Frameskip: %s, Image Dimension: %s, Cnn: cnn3, NSteps: %s, Memory dimension: %s" % (str(epoch), str(skip), str(image), str(nsteps), str(memory)))
                                                cumulative+=diagnostic_execute(game_env, selected_env, default_cnn3(number_actions, image), epoch, learn, batch_size, photogs,nsteps,memory)
                                            elif(i==3):
                                                print("Epochs to Test: %s Frameskip: %s, Image Dimension: %s, Cnn: cnn4, NSteps: %s, Memory dimension: %s" % (str(epoch), str(skip), str(image), str(nsteps), str(memory)))
                                                cumulative+=diagnostic_execute(game_env, selected_env, default_cnn4(number_actions, image), epoch, learn, batch_size, photogs,nsteps,memory)
                                        cumulative=cumulative/len(EPOCHS_POOL)
                                        if(cumulative>reward[0]):
                                            reward=[cumulative, skip, image[0],image[1],image[2], learn, batch_size, photogs, i,nsteps,memory]
    else:
        #to implement if necessary
        return  None

    writeOnFile(reward, str(PARAMETERS_DIR + "/" + ENV_NAMES[selected_env] + "_" + str(moves)))
    print("Exiting testing phase: "+ENV_NAMES[selected_env])
    return reward

def executeBestParameters(selected_env, epochs, goal, using_default_nns=1, custom_nns=[], moves=0):
    params=[]
    image=(GRAYSCALE,80,80)
    if(using_default_nns==1):
        if (os.path.exists(str(PARAMETERS_DIR + "/" + ENV_NAMES[selected_env] + "_" + str(moves)))):#file exists
            params=readFromFile(str(PARAMETERS_DIR + "/" + ENV_NAMES[selected_env] + "_" + str(moves)))
            #conversione di params[2] in image
            image=(int(params[2]),int(params[3]),int(params[4]))
        else:
            params=test_parameters(selected_env, using_default_nns=using_default_nns, custom_nns=custom_nns, moves=moves)
            image=(int(params[2]),int(params[3]),int(params[4]))
        print("Now starting the real test...")
        game_env = env_selector(selected_env, int(params[1]), image, moveset=moves)
        game_env = gym.wrappers.Monitor(game_env, "videos", force=True, video_callable=lambda episode_id: True)
        number_actions = game_env.action_space.n
        game_env.reset()
        if(int(params[8])==0):
            execute(game_env=game_env, network=default_cnn1(number_actions, image), selected_env=selected_env, learning_rate=float(params[5]),  epochs=epochs, batch_size=int(params[6]), goal_score=goal, number_of_photograms=int(params[7]),n_steps_ev=int(params[9]),memory_capacity=int(params[10]))
        if(int(params[8])==1):
            execute(game_env=game_env, network=default_cnn2(number_actions, image), selected_env=selected_env, learning_rate=float(params[5]),  epochs=epochs, batch_size=int(params[6]), goal_score=goal, number_of_photograms=int(params[7]),n_steps_ev=int(params[9]),memory_capacity=int(params[10]))
        if(int(params[8])==2):
            execute(game_env=game_env, network=default_cnn3(number_actions, image), selected_env=selected_env, learning_rate=float(params[5]),  epochs=epochs, batch_size=int(params[6]), goal_score=goal, number_of_photograms=int(params[7]),n_steps_ev=int(params[9]),memory_capacity=int(params[10]))
        if(int(params[8])==3):
            execute(game_env=game_env, network=default_cnn4(number_actions, image), selected_env=selected_env, learning_rate=float(params[5]),  epochs=epochs, batch_size=int(params[6]), goal_score=goal, number_of_photograms=int(params[7]),n_steps_ev=int(params[9]),memory_capacity=int(params[10]))
        game_env.close()
    else:
        #To implement if necessary
        return None
