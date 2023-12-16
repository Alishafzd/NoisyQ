# NoisyQ: A Noise-Based Exploration Method for DDQN
This project explores the idea of adding noise to the Q-values of a deep reinforcement learning (D-RL) agent to enhance its exploration in a complex grid world maze environment. We propose three different methods of adding noise to the loss function of a double deep Q-network (DDQN) algorithm and compare them with two baselines: epsilon greedy and noisy net. We use three metrics to evaluate the performance of each method: coverage matrix, entropy, and episodes per step.

Our results show that adding noise to the Q-values improves the coverage and entropy of the agent, but also reduces its efficiency and convergence. We analyze the possible reasons for this trade-off and suggest some directions for future research, such as using a noise network architecture, and applying the idea to more complex environments. We hope that our work can inspire more research on noisy-based exploration methods for D-RL.

# Code
The code for this project is written in Python and uses PyTorch as the deep learning framework. There are three ipython notebooks available to run each method: epsilon greedy, noisy net, and noisyq. The grid environment is the same for all methods and is defined in the grid_env.py file. The only difference is the network and the agent, which are defined in the network.py and agent.py files respectively, and are overrided in eahc method's ipython code. The notebooks can be run interactively and show the results immediately.

# Results
The results of this project are presented in two figures. The first figure shows the heat map of the coverage matrix for each method, along with the entropy values. The coverage matrix measures how well the agent explores the state space, and the entropy quantifies the diversity of the agent’s behavior. The figure demonstrates that noisyq models outperform the baselines in terms of exploration, especially at the first two checkpoints.

![Heat maps comparisons](/results/heat_maps.png | width=100)

The second figure shows the episodes per step for each method. The episodes per step reflects the efficiency of the agent’s learning. The figure reveals that noisyq models have not converged due to the local minimum problem!

<img src=“https://github.com/Alishafzd/NoisyQ/results/episodes.png” width=“100” height=“100” align="center"> Episodes per step for different methods. The constant tangent of the line indicates convergence.

