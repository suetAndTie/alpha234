from agents.random_agent import RandomAgent
from games.connect4.game import Game
from agents.user_agent import User

def main():
    agents = {1:User(), -1:RandomAgent()}
    game = Game()

    done = False
    while not done:
        game.gameState.get_visual_state()
        next_action = agents[game.currentPlayer].choose_action(game)
        # next_action
        next_state, value, done, info = game.step(next_action)
    game.gameState.get_visual_state()
    print(game.currentPlayer)

if __name__ == '__main__':
    main()
