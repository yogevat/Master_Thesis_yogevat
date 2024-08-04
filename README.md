# 🐶 RoboDog Simulation and Reinforcement Learning 🐾

Welcome to the RoboDog Simulation project, where we use motion planning algorithms to simulate and train a robotic dog using CoppeliaSim and Python. This project combines Rapidly-exploring Random Trees (RRT) and Reinforcement Learning (RL) to achieve sophisticated path planning and task execution.

## 🌟 Overview

This project demonstrates how a robotic dog can navigate and interact within a simulated environment using advanced algorithms. The environment is crafted in CoppeliaSim, with learning facilitated by the Stable Baselines3 library in Python.

## 📂 Files

### 🐍 Python Scripts

- **Run_RRT_In_Coppeliasim.py**: Connects to CoppeliaSim using the ZeroMQ Remote API to perform RRT path planning. It reads paths from an Excel file and simulates the dog's movement.

- **Envronment_RL_Push.py**: Implements the RL environment using OpenAI's Gym, defining the observation and action spaces, and incorporating the reward function for pushing a cube.

- **Learning_RL_Push.py**: Trains a model using Proximal Policy Optimization (PPO) with the `RobotModelEnv`. It also supports loading pre-trained models.

### 🧮 MATLAB Scripts

- **RoboDog_RRT_Planner.m**: Uses the RRT algorithm to plan a path in the simulation and saves the path data for execution. It leverages the `plannerRRT` function for planning and visualization.

- **DrawRectangle.m**: A utility function for visualizing the robot's position and orientation as rectangles.

### 🗺️ CoppeliaSim Scene Files

- **RoboDog_Learning_Push.ttt**: Sets up the simulation environment for RL tasks, including the robotic dog and a cube for interaction.

- **RoboDog_RRT.ttt**: Provides the simulation environment for executing RRT-based path planning.

## 🚀 Setup

1. **CoppeliaSim**: Install CoppeliaSim and enable the ZeroMQ Remote API plugin. Ensure scene files (`.ttt`) are in the correct directory.

2. **Python Environment**: Install the required packages:

   ```bash
   pip install stable-baselines3 gym numpy zmq opencv-python-headless
