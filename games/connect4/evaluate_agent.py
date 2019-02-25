from random_agent import RandomAgent
from mcts_agent import MCTSAgent
from game import Game

def main():
    user_agent = RandomAgent()
    adversarial_agent = RandomAgent()
    agents = [RandomAgent(), user_agent, adversarial_agent] #curr player is either 1 or -1, so index 0 is ignored
    game = Game()

    num_simulations = 200
    victories = 0

    for i in range(num_simulations):
        game.reset()
        if i%10 == 0:
            print('{} simulations run'.format(i))
        done = False
        while not done:
            next_action = agents[game.currentPlayer].choose_action(game)
            next_state, value, done, info = game.step(next_action)
        if game.currentPlayer == -1:
            victories += 1

    print('{}/{} games won by user agent (Random v Random)'.format(victories, num_simulations))

    user_agent = MCTSAgent(timeLimit=1000)
    # user_agent = RandomAgent()
    adversarial_agent = RandomAgent()
    # agents = [user_agent, adversarial_agent]
    agents = [RandomAgent(), user_agent, adversarial_agent] #curr player is either 1 or -1, so index 0 is ignored
    game = Game()

    num_simulations = 200
    victories = 0

    for i in range(num_simulations):
        game.reset()
        if i%10 == 0:
            print('{} simulations run'.format(i))
        done = False
        while not done:
            next_action = agents[game.currentPlayer].choose_action(game)
            next_state, value, done, info = game.step(next_action)
        if game.currentPlayer == -1:
            victories += 1

    print('{}/{} games won by user agent (MCTS v Random)'.format(victories, num_simulations))

    user_agent = MCTSAgent(timeLimit=1000)
    # user_agent = RandomAgent()
    adversarial_agent = RandomAgent()
    # agents = [user_agent, adversarial_agent]
    agents = [RandomAgent(), adversarial_agent, user_agent] #curr player is either 1 or -1, so index 0 is ignored
    game = Game()

    num_simulations = 200
    victories = 0

    for i in range(num_simulations):
        game.reset()
        if i%10 == 0:
            print('{} simulations run'.format(i))
        done = False
        while not done:
            next_action = agents[game.currentPlayer].choose_action(game)
            next_state, value, done, info = game.step(next_action)
        if game.currentPlayer == 1:
            victories += 1

    print('{}/{} games won by user agent (Random v MCTS)'.format(victories, num_simulations))

    user_agent = MCTSAgent(timeLimit=1000)
    # user_agent = RandomAgent()
    adversarial_agent = MCTSAgent(timeLimit=1000)
    # agents = [user_agent, adversarial_agent]
    agents = [RandomAgent(), user_agent, adversarial_agent] #curr player is either 1 or -1, so index 0 is ignored
    game = Game()

    num_simulations = 200
    victories = 0

    for i in range(num_simulations):
        game.reset()
        if i%10 == 0:
            print('{} simulations run'.format(i))
        done = False
        while not done:
            next_action = agents[game.currentPlayer].choose_action(game)
            next_state, value, done, info = game.step(next_action)
        if game.currentPlayer == -1:
            victories += 1

    print('{}/{} games won by user agent (MCTS v MCTS)'.format(victories, num_simulations))

if __name__ == '__main__':
    main()
