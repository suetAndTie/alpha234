from random_agent import RandomAgent
from game import Game



def main():
    user_agent = RandomAgent()
    adversarial_agent = RandomAgent()
    agents = [user_agent, adversarial_agent]
    game = Game()

    num_simulations = 500
    victories = 0

    for i in range(num_simulations):
        game.reset()
        if i%100 == 0:
            print('{} simulations run'.format(i))
        done = False
        while not done:
            next_action = agents[game.currentPlayer].choose_action(game)
            next_state, value, done, info = game.step(next_action)
        if game.currentPlayer == -1:
            victories += 1

    print('{}/{} games won by user agent'.format(victories, num_simulations))

if __name__ == '__main__':
    main()
