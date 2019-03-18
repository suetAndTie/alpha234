from random_agent import RandomAgent
from mcts_agent import MCTSAgent
from game import Game

def main():
    # agents = [RandomAgent(), MCTSAgent(timeLimit=1000)]
    agents = [RandomAgent(), MCTSAgent(timeLimit=1000), RandomAgent()] #curr player is either 1 or -1, so index 0 is ignored. Player at index 1 plays first
    game = Game()

    done = False
    while not done:
        next_action = agents[game.currentPlayer].choose_action(game)
        next_state, value, done, info = game.step(next_action)
        game.gameState.get_visual_state()
    print('Reward: {}'.format(game.currentPlayer * -1)) # whoevers turn it is when game loop terminates has lost

if __name__ == '__main__':
    main()
