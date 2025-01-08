## Introduction
As you can see from my Github, I usually work in applying Reinforcement Learning to physics design problems. The codebase I usually use is owned by the JILA team, implements double deep Q-Networks and is in C++. I usually couple that to my C++ / Rust physics simulations.

This is an ongoing repo where, mostly for my own edification, I implement a deep Q-Network RL in Rust. Goal was to gain solidify my foundation in RL + gain familiarity with the Burn API in Rust for doing ML. Eventually I'd like to make a pure Rust repo where both my RL and my simulations are in Rust. 

It's very much a work in progress, especially as I learn more about Burn and become more comfortable in Rust. There's a lot of code fixing and testing to be done.
Eventually after that, I'd like to implement and play around with physics informed variations of double deep Q- Networks.

P.S. I recently learned the Burn project, in their github examples folder, actually offers a DQN example. Mine is a lot more coarse haha, but I hope to experiment around with mine and see if I can play around with interpretability and designing physics informed networks. Especially for usecases like my github repo `lattice_evolution`.