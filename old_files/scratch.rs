


fn main() {
    let input_size = 4; // Example input size (e.g., state size)
    let hidden_size = 64;
    let output_size = 2; // Number of actions (Q-values)

    // Initialize DQN model and optimizer
    let model = DQN::new(input_size, hidden_size, output_size);
    let mut optimizer = Adam::default().build(&model);

    // Experience replay buffer
    let mut replay_buffer = ReplayBuffer::new(10000);

    // Training parameters
    let batch_size = 32;
    let gamma = 0.99; // Discount factor
    let epsilon = 0.1; // Epsilon for exploration-exploitation trade-off
    let episodes = 1000;
    let mut total_timesteps = 0;

    // Simulate training loop
    for episode in 0..episodes {
        let mut state = Tensor::from_data(vec![0.0, 0.0, 0.0, 0.0], (4, 1)); // Initialize state
        let mut done = false;
        let mut total_reward = 0.0;

        while !done {
            total_timesteps += 1;
            
            // Select action using epsilon-greedy policy
            let action = if rand::random::<f32>() < epsilon {
                // Exploration: random action
                Tensor::from_data(vec![rand::random::<u8>() as f32], (1, 1))
            } else {
                // Exploitation: choose action with max Q-value
                let q_values = model.forward(state.clone());
                let action_idx = q_values.argmax(0).unwrap();
                action_idx
            };

            // Simulate environment interaction (replace with actual environment logic)
            let reward = 1.0; // Example reward
            let next_state = Tensor::from_data(vec![0.1, 0.1, 0.1, 0.1], (4, 1)); // Example next state
            done = true; // Example terminal condition

            // Store experience in replay buffer
            replay_buffer.add(state.clone(), action.clone(), reward, next_state.clone(), done);





            // Update state
            state = next_state;
            total_reward += reward;
        }

        println!("Episode {}: Total Reward: {}", episode, total_reward);
    }
}
