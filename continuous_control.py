"""
Train and run a DDPG agent in a unity environment simulating the continous
control of an actuated robotic arm.

# Algorithm 1 from DDPG paper
# Initialise critic network Q(s, a|θ^Q) and actor μ(s|θ^μ)
# Initialise target network Q' μ' with weights
# Initialise replay buffer
# for episode i =1, M do
#  Initialise a random process 𝒩 for aciton exploration
#  Receive initial observation state s1
#  for t = 1, T do
#    Select action a_t = μ(s|θ^μ) + 𝒩_t
#    Execute action, observe reward r_t, observe new state s_{t+1}
#    Store transition (s_t, a_t, r_t, s_t+1) in replay buffer
"""

from unityagents import UnityEnvironment
from drlnd.common.agents import DDPGAgent
from drlnd.common.agents.utils import ReplayBuffer, ActionType
from drlnd.common.utils import get_next_results_directory, path_from_project_home
import numpy as np
import torch
import click
from collections import deque
import numpy as np
import os
import yaml

def get_unity_env(path: str):
    env = UnityEnvironment(file_name=path)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]   
    num_agents = len(env_info.agents)

    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    print(f"Num agents: {num_agents}\nAction dim: f{action_size}\nState dim: {state_size}")

    return env, brain_name, num_agents, action_size, state_size



@click.group()
def cli():
    pass


@cli.command()
@click.option("--n-episodes", type=int, default = 1, 
    help="""
    Number of episodes to train for
    """)
@click.option("--note", type=str, default=None, 
    help="""
    Note to record to .txt file when results are saved.
    """)
def train(n_episodes, note):
    """Train an agent in the 'one_agent' environment using DDPG.
    """
    env, brain_name, num_agents, action_size, state_size = get_unity_env('./unity_environments/one_agent/Reacher_Linux/Reacher.x86_64')

    buffer = ReplayBuffer(action_size, int(1e6), 64, 1234, action_dtype = ActionType.CONTINUOUS)
    agent = DDPGAgent(state_size, action_size, buffer)

    episode_scores = deque(maxlen=100)
    average_scores = []
    noise_fn_taper = 250
    scale = 1.0
    for i in range(n_episodes):
        print(f"Episode: {i}")
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations                   # get the current state (for each agent)
        #print(f"Initial state: {state}")
        score = 0.0                                            # initialize the score (for each agent)
        
        # Include some noise in the action selection, which we linearly scale
        scale = max([1e-4, scale*(1.0-(float(i)/noise_fn_taper))])
        noise_fn = lambda : torch.from_numpy(np.random.normal(loc=0.0, scale=scale, size=(1, action_size))).float()
        while True:
            action = (
                agent.act(
                    torch.from_numpy(state).float(), 
                    noise_func = noise_fn)
                    .numpy()
                    )
            env_info = env.step([action])[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished
            agent.replay_buffer.add(*state, *action, *reward, *next_state, *done)
            agent.learn(0.99)

            score += env_info.rewards[0]                         # update the score (for each agent)
            state = next_state                               # roll over states to next time step
            if np.any(done):                                  # exit loop if episode finished
                break
        episode_scores.append(score)
        average_scores.append(np.mean(episode_scores))
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(score)))

    results_directory = get_next_results_directory()
    agent.save_weights(results_directory)
    np.savetxt(os.path.join(results_directory, 'scores.txt'), average_scores)
    if note is not None:
        with open(os.path.join(results_directory, 'note.txt'), 'w') as f:
            f.write(note)
    params = {
        'noise_fn': repr(noise_fn),
        'noise_fn_taper': noise_fn_taper, 
        'taper': 'linear'}
    with open(os.path.join(results_directory, 'params.yml'), 'w') as f:
        yaml.dump(params, f)


@cli.command()
@click.option("--weights-path", type=str, default=None, 
    help="""
    Path to the directory containing the trained weights of the agent's network.
    Can be none, in which case the pre-trained weights in resources are used.
    """)
@click.option('--n-episodes', type=int, default=1, 
    help = """
    Number of episodes to train an agent for.
    """)
def run(weights_path: str, n_episodes: int):
    """Initialise an agent using pre-trained network weights and observe the 
    agent's interaction with the environment.
    """
    env, brain_name, num_agents, action_size, state_size = get_unity_env('./unity_environments/one_agent/Reacher_Linux/Reacher.x86_64')

    buffer = ReplayBuffer(action_size, int(1e6), 64, 1234, action_dtype = ActionType.CONTINUOUS)
    agent = DDPGAgent(state_size, action_size, buffer)
    if weights_path is None:
        weights_path = path_from_project_home('./resources/solved_weights/')
    agent.load_weights(weights_path)

    for i in range(n_episodes):
        print(f"Episode: {i}")
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations
        score = 0.0                                            # initialize the score (for each agent)
        for i in range(300):
            action = agent.act(torch.from_numpy(state).float()).numpy() # select an action (for each agent)
            env_info = env.step([action])[brain_name]           # send all actions to tne environment
            state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished

            score += env_info.rewards[0]                         # update the score (for each agent)
            if np.any(done):                                  # exit loop if episode finished
                break
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(score)))


# Collect trajectories using agents
if __name__ == "__main__":
    project_home = (os.getcwd() if 'PROJECT_HOME' not in os.environ.keys() else os.environ['PROJECT_HOME'])
    os.environ['PROJECT_HOME'] = project_home
    cli()