"""
Based on
https://github.com/suragnair/alpha-zero-general
"""


import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
from utils.multiprocessing import executor_init

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            assert valids[action] > 0, "action {} is not valid".format(action)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.
        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        bar = tqdm(desc='Arena.playGames', total=num)
        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            bar.update()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==-1:
                oneWon+=1
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            bar.update()

        bar.close()

        return oneWon, twoWon, draws


class ArenaMP(Arena):
    """
    Arena class that utilizes multiprocessing.
    Note: Use non-human players only
    """
    def __init__(self, player1, player2, game, display=None, lock=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
            lock: optional lock for multiprocessing
        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        super().__init__(player1, player2, game, display=None)
        self.lock = lock

    def playGames(self, num, num_workers=cpu_count(), verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.
        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        bar = tqdm(desc='Arena.playGames', total=num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0

        with ProcessPoolExecutor(max_workers=nworkers, initializer=executor_init, initargs=(self.lock,)) as executor:
            futures = []
            for _ in range(num):
                # gameResult = self.playGame(verbose=verbose)
                futures.append(executor.submit(self.playGame, verbose))

            for future in as_completed(futures):
                gameResult = future.result()
                if gameResult==1:
                    oneWon+=1
                elif gameResult==-1:
                    twoWon+=1
                else:
                    draws+=1
                # bookkeeping + plot progress
                bar.update()

            self.player1, self.player2 = self.player2, self.player1

            futures = []
            for _ in range(num):
                # gameResult = self.playGame(verbose=verbose)
                futures.append(executor.submit(self.playGame, verbose))

            for future in as_completed(futures):
                gameResult = future.result()
                if gameResult==-1:
                    oneWon+=1
                elif gameResult==1:
                    twoWon+=1
                else:
                    draws+=1
                # bookkeeping + plot progress
                bar.update()

        bar.close()

        return oneWon, twoWon, draws
