from random_agent import RandomAgent
from game import Game

def main():
    agents = [RandomAgent(), RandomAgent()]
    game = Game()

    done = False
    while not done:
        game.gameState.get_visual_state()
        next_action = agents[game.currentPlayer].choose_action(game)
        # next_action)
        next_state, value, done, info = game.step(next_action)
    game.gameState.get_visual_state()

if __name__ == '__main__':
    main()
