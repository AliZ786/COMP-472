# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time

class Game:
	MINIMAX = 0
	ALPHABETA = 1
	HUMAN = 2
	AI = 3
	
	# Number of white and black pieces on the board
	num_X = 0
	num_O = 0

	def __init__(self, recommend = True):
		self.initialize_game()
		self.recommend = recommend

	# Parametrized constructor
	def __init__(self, n, b, pb, s, d1, d2, t, recommend = True):
		self.n = n
		self.b = b
		self.pb = pb
		self.s = s
		self.d1 = d1
		self.d2 = d2
		self.t = t
		self.current_state = []
		self.initialize_game()
		self.recommend = recommend
		self.f = open(F'gameTrace-{self.n}{self.b}{self.s}{self.t}', "w")
		self.f.write(F'n={self.n} b={self.b} s={self.s} t={self.t}\n')
		self.f.write(F'blocs={self.pb}\n\n')
		
	def initialize_game(self):
		for y in range(self.n):
			arr = []
			for x in range(self.n):
				# put blocs
				for bloc in self.pb:
					if (bloc) == (y, x):
						arr.append('#')
					else:
						arr.append(".")
			self.current_state.append(arr)

		# Player X always plays first
		self.player_turn = 'X'

	def draw_board(self):
		print()
		self.f.write("\n")
		for y in range(0, self.n):
			for x in range(0, self.n):
				print(F'{self.current_state[x][y]}', end="")
				self.f.write(F'{self.current_state[x][y]}')
			print()
			self.f.write("\n")
		print()
		self.f.write("\n")
		
	def is_valid(self, px, py):
		if px < 0 or px > self.n or py < 0 or py > self.n:
			return False
		elif self.current_state[px][py] != '.':
			return False
		else:
			return True

	def is_end(self):
		# Vertical win
		for i in range(self.n):
			lineCount = 0
			for j in range(self.n-1):
				if(self.current_state[j][i] == "#" or self.current_state[j][i] == "."
					or self.current_state[j][i] != self.current_state[j+1][i]):
					lineCount = 0
				else:
					lineCount += 1

				if(lineCount == self.s-1):
					return self.current_state[j][i]
		
		# Horizontal win
		for i in range(self.n):
			lineCount = 0
			for j in range(self.n-1):
				if(self.current_state[j][i] == "#" or self.current_state[j][i] == "."
					or self.current_state[j][i] != self.current_state[j][i+1]):
					lineCount = 0
				else:
					lineCount += 1

				if(lineCount == self.s-1):
					return self.current_state[j][i]
		
		# Main diagonal win
		for i in range((self.n + 1) - self.s):
			lineCount = 0
			for j in range(self.n - 1 - j):
				if(self.current_state[i][i+j] == "#" or self.current_state[i][i+j] == "."
					or self.current_state[i][i+j] != self.current_state[i][i+j+1]):
					lineCount = 0
				else:
					lineCount += 1

				if(lineCount == self.s-1):
					return self.current_state[i][i + j]
		
		# Second diagonal win
		for i in range((self.n + 1) - self.s):
			lineCount = 0
			for j in range(self.n - 1 - j):
				if(self.current_state[i][self.n - 1 - i - j] == "#" or self.current_state[i][self.n - 1 - i - j] == "."
					or self.current_state[i][self.n - 1 - i - j] != self.current_state[i][self.n - 1 - (i + 1) - j]):
					lineCount = 0
				else:
					lineCount += 1

				if(lineCount == self.s-1):
					return self.current_state[i][self.n - 1 - i - j]

		# Something (random diagonals)
		
		# Is whole board full?
		for i in range(0, self.n):
			for j in range(0, self.n):
				# There's an empty field, we continue the game
				if (self.current_state[i][j] == '.'):
					return None
		# It's a tie!
		return '.'

	def check_end(self):
		self.result = self.is_end()
		# Printing the appropriate message if the game has ended
		if self.result != None:
			if self.result == 'X':
				print('The winner is X!')
				self.f.write('The winner is X!')
			elif self.result == 'O':
				print('The winner is O!')
				self.f.write('The winner is O!')
			elif self.result == '.':
				print("It's a tie!")
				self.f.write("It's a tie!")
			self.initialize_game()
		return self.result

	def input_move(self):
		while True:
			print(F'Player {self.player_turn}, enter your move:')
			px = int(input('enter the x coordinate: '))
			py = int(input('enter the y coordinate: '))
			if self.is_valid(px, py):
				return (px,py)
			else:
				print('The move is not valid! Try again.')

	def switch_player(self):
		if self.player_turn == 'X':
			self.player_turn = 'O'
		elif self.player_turn == 'O':
			self.player_turn = 'X'
		return self.player_turn

	def minimax(self, max=False):
		# Minimizing for 'X' and maximizing for 'O'
		# Possible values are:
		# -1 - win for 'X'
		# 0  - a tie
		# 1  - loss for 'X'
		# We're initially setting it to 2 or -2 as worse than the worst case:
		value = 2
		if max:
			value = -2
		x = None
		y = None
		result = self.is_end()
		if result == 'X':
			return (-1, x, y)
		elif result == 'O':
			return (1, x, y)
		elif result == '.':
			return (0, x, y)
		for i in range(0, 3):
			for j in range(0, 3):
				if self.current_state[i][j] == '.':
					if max:
						self.current_state[i][j] = 'O'
						(v, _, _) = self.minimax(max=False)
						if v > value:
							value = v
							x = i
							y = j
					else:
						self.current_state[i][j] = 'X'
						(v, _, _) = self.minimax(max=True)
						if v < value:
							value = v
							x = i
							y = j
					self.current_state[i][j] = '.'
		return (value, x, y)

	def alphabeta(self, alpha=-2, beta=2, max=False):
		# Minimizing for 'X' and maximizing for 'O'
		# Possible values are:
		# -1 - win for 'X'
		# 0  - a tie
		# 1  - loss for 'X'
		# We're initially setting it to 2 or -2 as worse than the worst case:
		value = 2
		if max:
			value = -2
		x = None
		y = None
		result = self.is_end()
		if result == 'X':
			return (-1, x, y)
		elif result == 'O':
			return (1, x, y)
		elif result == '.':
			return (0, x, y)
		for i in range(0, 3):
			for j in range(0, 3):
				if self.current_state[i][j] == '.':
					if max:
						self.current_state[i][j] = 'O'
						(v, _, _) = self.alphabeta(alpha, beta, max=False)
						if v > value:
							value = v
							x = i
							y = j
					else:
						self.current_state[i][j] = 'X'
						(v, _, _) = self.alphabeta(alpha, beta, max=True)
						if v < value:
							value = v
							x = i
							y = j
					self.current_state[i][j] = '.'
					if max: 
						if value >= beta:
							return (value, x, y)
						if value > alpha:
							alpha = value
					else:
						if value <= alpha:
							return (value, x, y)
						if value < beta:
							beta = value
		return (value, x, y)

	def play(self, algo=None,player_x=None,player_o=None):
		# Write parameters of each player to the file
		if player_x == self.AI:
			self.f.write(F'Player 1: AI d={self.d1} ')
		else:
			self.f.write(F'Player 1: HUMAN d={self.d1} ')
		if algo == self.ALPHABETA:
			self.f.write(F'a=True \n')
		else:
			self.f.write(F'a=False \n')

		if player_o == self.AI:
			self.f.write(F'Player 2: AI d={self.d2} ')
		else:
			self.f.write(F'Player 2: HUMAN d={self.d2} ')
		if algo == self.ALPHABETA:
			self.f.write(F'a=True ')
		else:
			self.f.write(F'a=False ')

		self.f.write('\n')

		if algo == None:
			algo = self.ALPHABETA
		if player_x == None:
			player_x = self.HUMAN
		if player_o == None:
			player_o = self.HUMAN
		while True:
			self.draw_board()
			if self.check_end():
				return
			start = time.time()
			if algo == self.MINIMAX:
				if self.player_turn == 'X':
					(_, x, y) = self.minimax(max=False)
				else:
					(_, x, y) = self.minimax(max=True)
			else: # algo == self.ALPHABETA
				if self.player_turn == 'X':
					(m, x, y) = self.alphabeta(max=False)
				else:
					(m, x, y) = self.alphabeta(max=True)
			end = time.time()
			elapsed_t = end - start
			if (elapsed_t > self.t):
				print(F'Player {self.player_turn} is eliminated for taking too much time to return a move.')
			if (self.player_turn == 'X' and player_x == self.HUMAN) or (self.player_turn == 'O' and player_o == self.HUMAN):
					if self.recommend:
						print(F'Evaluation time: {round(end - start, 7)}s')
						print(F'Recommended move: x = {x}, y = {y}')
					(x,y) = self.input_move()
			if (self.player_turn == 'X' and player_x == self.AI) or (self.player_turn == 'O' and player_o == self.AI):
						print(F'Evaluation time: {round(end - start, 7)}s')
						print(F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')
			self.current_state[x][y] = self.player_turn
			self.switch_player()
	
	# Determining number of white pieces
	def num_white(self):
		for i in range(0, self.n):
			for j in range(0, self.n):
				if (self.current_state[i][j] == 'X'):
					num_X = num_X + 1
				else:
					return None
		return num_X				

	# Determining number of black pieces
	def num_black(self):
		for i in range(0, self.n):
			for j in range(0, self.n):
				if (self.current_state[i][j] == 'O'):
					num_O = num_O + 1
				else:
					return None
		return num_O

	# Developing e1 (simple heuristic)
	def e1(self):
		e = self.num_white - self.num_black
		return e

def main():
	# g = Game(recommend=True)

	n = 3
	b = 2
	pb = [[0, 0], [1, 2]]
	s = 4
	d1 = 2
	d2 = 2
	t = 5

	g = Game(3, 3, [(0,0), (1,1), (1,2)], 3, 2, 2, 5, recommend=True)
	#g.play(algo=Game.ALPHABETA,player_x=Game.AI,player_o=Game.AI)
	g.play(algo=Game.MINIMAX,player_x=Game.AI,player_o=Game.HUMAN)


if __name__ == "__main__":
	main()
