from unityagents import UnityEnvironment
from drlnd.common.agents import DDPGAgent
from drlnd.common.agents.utils import ReplayBuffer, ActionType
import numpy as np
import torch

def run():
    env = UnityEnvironment(file_name='./unity_environments/one_agent/Reacher_Linux/Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    num_agents = len(env_info.agents)
    print(f"Number of agents: {num_agents}")

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]

    # At all points in time the agent acts according to the same policy, so we only
    # need one agent, and feed each state to the agent sequentially.
    buffer = ReplayBuffer(action_size, int(1e6), 64, 1234, action_dtype = ActionType.CONTINUOUS)
    agent = DDPGAgent(state_size, action_size, buffer)

    n_episodes = 2
    for i in range(n_episodes):
        print(f"Episode: {i}")
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations                   # get the current state (for each agent)
        #print(f"Initial state: {state}")
        score = 0.0                                            # initialize the score (for each agent)
        while True:
            action = agent.act(torch.from_numpy(state).float()).numpy() # select an action (for each agent)
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
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(score)))

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

# Collect trajectories using agents
if __name__ == "__main__":
    run()