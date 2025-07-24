


# Reinforcement Learning Pong Game (Course Project)
## Objective

This project is all about teaching an AI to play Pong using reinforcement learning. The goal is to get the agent to learn and get better at the game by itself, using Q-learning. It’s a fun way to see how machine learning can actually play games and improve over time!

This project was completed as part of my course requirements during my master's program.

## Description
A reinforcement learning implementation of Pong, using Python and Pygame. The project includes training and evaluation scripts, and uses Q-learning with a saved Q-table.

## How to Run
1. Create and activate a Python virtual environment:
	```zsh
	python3 -m venv .venv
	source .venv/bin/activate
	```
2. Install dependencies:
	```zsh
	pip install --upgrade pip
	pip install -r requirements.txt
	```
3. Run the main script:
	```zsh
	python pong.py
	```

## Files
- `pong.py`: Main script to run Pong RL.
- `player_A_250_qtable.pkl`: Pre-trained Q-table for the agent.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Output

![Game Output](play.gif)
