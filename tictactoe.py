import numpy as np
import random


class Tictactoe:
    win_reward = 1
    lose_reward = -2
    draw_reward = 0.5

    empty_board = np.zeros([3, 3])

    def __init__(self):
        self.board = self.empty_board.copy()

        self.agent_symbol = self.win_reward
        self.computer_symbol = self.lose_reward

    def checkwin(self):
        # Checks rows
        for row in range(3):
            if np.all(self.board[row] == self.board[row][0]) and self.board[row][0] != 0:
                return self.board[row][0], True
        # Checks columns
        for col in range(3):
            if np.all(self.board.T[col] == self.board.T[col][0]) and self.board.T[col][0] != 0:
                return self.board.T[col][0], True
        # Checks diagonal
        if np.all(self.board.diagonal() == self.board[1, 1]) and self.board[1, 1] != 0:
            return self.board[1, 1], True
        # Checks reverse diagonal
        if np.all(np.fliplr(self.board).diagonal() == self.board[1, 1]) and self.board[1, 1] != 0:
            return self.board[1, 1], True
        # Checks if draw
        if not np.isin(0, self.board):
            return self.draw_reward, True

        return 0, False

    def play(self, position):

        coordinate = (position//3, position % 3)
        if not self.valid_move(coordinate):
            return -10, False

        self.board[coordinate] = self.agent_symbol

        reward, done = self.checkwin()

        if not done:
            position = self.computer_move()
            coordinate = (position//3, position % 3)
            self.board[coordinate] = self.computer_symbol
            reward, done = self.checkwin()

        return reward, done

    def valid_move(self, coordinate):
        return self.board[coordinate] == 0

    def get_board(self):
        return self.board

    def display_board(self):
        print(self.board)

    def reset(self):

        self.board = self.empty_board.copy()

        if random.random() < 0.5:
            move = self.computer_move()
            self.board[move//3, move % 3] = self.computer_symbol

        return self.board

    def minimax(self, board, alpha, beta, maximizing):

        result, _ = self.checkwin()

        if result == self.computer_symbol:
            return 1

        if result == self.agent_symbol:
            return -1

        if result == self.draw_reward:
            return 0

        if maximizing:
            best_score = -10
            valid_moves = np.where(board.flatten() == 0)[0]
            np.random.shuffle(valid_moves)

            for i in valid_moves:
                self.board[i//3, i % 3] = self.computer_symbol
                score = self.minimax(board, alpha, beta, False)
                self.board[i // 3, i % 3] = 0

                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break

            return best_score
        else:

            best_score = 10
            valid_moves = np.where(board.flatten() == 0)[0]
            np.random.shuffle(valid_moves)

            for i in valid_moves:
                self.board[i // 3, i % 3] = self.agent_symbol
                score = self.minimax(board, alpha, beta, True)
                self.board[i // 3, i % 3] = 0

                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score

    def computer_move(self):

        best_move = None
        best_score = -10
        alpha = -np.inf
        beta = np.inf
        valid_moves = np.where(self.board.flatten() == 0)[0]
        np.random.shuffle(valid_moves)  # minimax chooses randomly amongst moves with tied score

        for i in valid_moves:
            self.board[i // 3, i % 3] = self.computer_symbol
            score = self.minimax(self.board, alpha, beta, False)
            self.board[i // 3, i % 3] = 0

            if score > best_score:
                best_score = score
                best_move = i
        return best_move

    def human_play(self):
        done = False
        result = None
        self.display_board()
        while not done:
            i = input()
            self.board[int(i) // 3, int(i) % 3] = self.agent_symbol
            self.display_board()
            result, done = self.checkwin()
            if done:
                break

            i = self.computer_move()
            self.board[i // 3, i % 3] = self.computer_symbol
            self.display_board()
            result, done = self.checkwin()
        print(result)
