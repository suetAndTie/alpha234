class User():
    # def __init__(self, name, state_size, action_size):
    #     pass

    def choose_action(state, game):
        print(game.gameState.allowedActions)
        action = input('Enter your action: ')
        return int(action)
