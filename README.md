# Reinforcement Learning (double deep Q-learning) in Rust using Burn API
## Preliminary Notes
As you can see from my Github, I usually work in applying Reinforcement Learning to physics design problems (see [paper](https://arxiv.org/abs/2405.07907) ).. The codebase I usually use is owned by the JILA team, implements double deep Q-Networks and is in C++. I usually couple that to my C++ / Rust physics simulations.

This is an ongoing repo where, partly for my own edification, I implement a double deep Q-Network learning RL in Rust. Goal was to gain solidify my foundation in RL + gain familiarity with the Burn API in Rust for doing ML. Eventually I'd like to make a pure Rust repo where both my RL and my simulations are in Rust. It's very much a work in progress, especially as I learn more about Burn, so let me know if you see any issues. There's a lot of code fixing and testing to be done.
Eventually after that, I'd like to implement and play around with physics informed variations of double deep Q- Networks.

P.S. I recently learned the Burn project, in their github examples folder, actually offers a DQN example. Mine is a lot more coarse, but I hope to experiment around with mine to customize it for as much policy interpretability as possible as well as designing physics informed RL. Especially for usecases like my github repo [lattice evolution](https://github.com/rootware/lattice_evolution).

## How this project is organized + Things to Know
You have a choice of RL agents that utilize Q-learning. The currently implemented one is DQN, with DDQN coming soon. These can be found under the `src/agent/` folder.
Environments can be found under `src/environment/`. The only one currently implemented is a simple 2D grid walker that can move up, down, left, right. This simple agent is useful for visualizing the Q-values for each point on the cartesian grid that forms the state space.

Model contains the neural network being used for learning the Q-values  and action selection via the target and policy networks. Note that the copying functions for models that are useful for updating in DQN/DDQN are implemented as part of the Model. Reason for this is that I haven't found a clean way to iterate over all the layers in a given model in Burn, so have to manually code the copying over layer by layer using the `map` function for updating.

## Recent edits:
- Uses Libtorch backend successfully
- Changed to Xavier initialization
- Organized files
- Implemented the model weighted copy function to set up for implementing DDQN.
- On some tested case, ReLU was running into the "dying ReLU neurons" issue. CIrcumvented by adding a Leaky ReLU.
- The DDQN (implemented in `ddqn.rs`) was benchmarked. Need to rewrite DQN (implemented in `dqn.rs`) to be similar in notation to how it's done in literature.
- Improve batching.
- Improve how and when constants such as `STATE_SIZE, HIDDEN_SIZE, NUM_ACTIONS` are defined.

## Specific To-Dos:
- Implement epsilon greedy : right now, have epsilon decreasing but it doesn't use a function and is done manually
- ~Move towards epochs and proper batching of samples from replay buffer~
- ~Use Libtorch backend if possible~ Using Libtorch backend
- Test on more involved examples to see if it's successfully learning e.g. shaken lattice repo
