# Alpha234
## Alexander Wang (aswang96) and Trevor Tsue (ttsue)
AlphaZero Project for Stanford's CS 234: Reinforcement Learning

Based on https://github.com/suragnair/alpha-zero-general


## New features
- Multiprocessing
  - Self-Play Monte Carlo Tree Search
  - Opponent Play
- TensorboardX
  - Optional tensorboard for model training statistics
- Elo Score
  - Elo score based on self-play
- Resnet34
  - Resnet model adapted to games


## How to Train
First, install requirements

`pip install -r requirements.txt`


*Optional: change settings in config.py*

Run main method to train

`python main.py`
