# Marl Co-op
Multi-file version of the "Emergent Cooperation vs Competition" MARL baseline.

In this file: 
• Predator–Prey gridworld (discrete, synchronous multi-agent)
• Independent DQN agents (baseline) with epsilon-greedy
• Reward mixing: r_i' = alpha * r_i + (1 - alpha) * R_team (sweep alpha in [0,1])
• Metrics: team reward, reward variance (equality), action entropy, simple coordination index

To run: 
python MARL_Emergent_Cooperation_Baseline.py --episodes <number, ex: 2000> --alpha <0-1.0> --seed <number, ex: 42>
