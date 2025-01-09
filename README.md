# Reinforcement Learning (double deep Q-learning) in Rust using Burn API
## Introduction
As you can see from my Github, I usually work in applying Reinforcement Learning to physics design problems. The codebase I usually use is owned by the JILA team, implements double deep Q-Networks and is in C++. I usually couple that to my C++ / Rust physics simulations.

This is an ongoing repo where, mostly for my own edification, I implement a deep Q-Network RL in Rust. Goal was to gain solidify my foundation in RL + gain familiarity with the Burn API in Rust for doing ML. Eventually I'd like to make a pure Rust repo where both my RL and my simulations are in Rust. It's very much a work in progress, especially as I learn more about Burn, so let me know if you see any issues. There's a lot of code fixing and testing to be done.
Eventually after that, I'd like to implement and play around with physics informed variations of double deep Q- Networks.

P.S. I recently learned the Burn project, in their github examples folder, actually offers a DQN example. Mine is a lot more coarse, but I hope to experiment around with mine to customize it for as much policy interpretability as possible as well as designing physics informed RL. Especially for usecases like my github repo `lattice_evolution`.

## Tests done and Recent edits:
- Uses Libtorch backend successfully
- Changed to Xavier initialization
- For some reason, Sigmoid gives better results for Q-Table than ReLu
- While it can often find the best policy, if an entry in the Q-Table becomes zero it stays zero. Unsure why this is happening.
- Disparity between best policy and the zero epsilon policy. Am testing by printing out Q-Tables for examples.

## Specific To-Dos:
- Implement epsilon greedy : right now, have epsilon decreasing but it doesn't use a function and is done manually
- ~Move towards epochs and proper batching of samples from replay buffer~
- ~Use Libtorch backend if possible~ Using Libtorch backend
- Test on more involved examples to see if it's successfully learning e.g. shaken lattice repo
- Implement double deep by adding a target and policy network with soft and hard updates
