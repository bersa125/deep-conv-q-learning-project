from parameters_auto_tuning import *
from configuration_paramaters import *

if __name__ == "__main__":

    if(PARAMETER_SELECTION):
        """ Manual Parameter Selection """
        game_env = env_selector(ENV_SELECTED, FRAME_SKIP, image_dim, moveset=MOVE_SET)
        game_env = gym.wrappers.Monitor(game_env, "videos", force=True, video_callable=lambda episode_id: True)
        number_actions = game_env.action_space.n
        game_env.reset()
        execute(game_env=game_env, network=default_cnn3(number_actions, image_dim), selected_env=ENV_SELECTED, learning_rate=LEARNING_RATE,  epochs=EPOCHS, batch_size=BATCH_SIZE, goal_score=GOAL_SCORE, number_of_photograms=NUMBER_OF_PHOTOGRAMS,n_steps_ev=NSTEP_EVALUATION,memory_capacity=MEMORY_CAPACITY)
        game_env.close()
    else:
        executeBestParameters(ENV_SELECTED,EPOCHS,GOAL_SCORE,moves=MOVE_SET)
