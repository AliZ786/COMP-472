# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time

class Game:
	MINIMAX = 0
	ALPHABETA = 1
	HUMAN = 2
	AI = 3

	# Required attributes
	size = 0
	num_bloc = 0
	pos_bloc = 0
	s = 0
	d1 = 0
	d2 = 0
	t = 0
	a = False
	play_mode = 'AI-AI'
	
	# Number of white and black pieces on the board
	num_X = 0
	num_O = 0

	def __init__(self, recommend = True):
		self.initialize_game()
		self.recommend = recommend

	# Parametrized constructor
	def __init__(self, n, b, pb, s, d1, d2, t, a, p_mode, recommend = True):
		self.size = n
		self.num_bloc = b
		self.pos_bloc = pb
		self.s = s
		self.d1 = d1
		self.d2 = d2
		self.t = t
		self.a = a
		self.play_mode = p_mode
		self.current_state = []
		self.initialize_game()
		self.recommend = recommend
		
	def initialize_game(self):
		for y in range(self.size):
			arr = []
			for x in range(self.size):
				# put blocs
				for bloc in self.pos_bloc:
					if bloc == (y, x):
						arr.append('#')
					else:
						arr.append(".")
			self.current_state.append(arr)

		# Player X always plays first
		self.player_turn = 'X'

	def draw_board(self):
		print()
		for y in range(0, self.size):
			for x in range(0, self.size):
				print(F'{self.current_state[x][y]}', end="")
			print()
		print()
		
	def is_valid(self, px, py):
		if px < 0 or px > self.size or py < 0 or py > self.size:
			return False
		elif self.current_state[px][py] != '.':
			return False
		else:
			return True

	def is_end(self):
		# Vertical win
		for i in range(self.size):
			lineCount = 0
			for j in range(self.size-1):
				if(self.current_state[j][i] == "#" or self.current_state[j][i] == "."
					or self.current_state[j][i] != self.current_state[j+1][i]):
					lineCount = 0
				else:
					lineCount += 1

				if(lineCount == self.s-1):
					return self.current_state[j][i]
		
		# Horizontal win
		for i in range(self.size):
			lineCount = 0
			for j in range(self.size-1):
				if(self.current_state[j][i] == "#" or self.current_state[j][i] == "."
					or self.current_state[j][i] != self.current_state[j][i+1]):
					lineCount = 0
				else:
					lineCount += 1

				if(lineCount == self.s-1):
					return self.current_state[j][i]
		
		# Main diagonal win
		for i in range((self.size + 1) - self.s):
			lineCount = 0
			for j in range(self.size - 1 - j):
				if(self.current_state[i][i+j] == "#" or self.current_state[i][i+j] == "."
					or self.current_state[i][i+j] != self.current_state[i][i+j+1]):
					lineCount = 0
				else:
					lineCount += 1

				if(lineCount == self.s-1):
					return self.current_state[i][i + j]
		
		# Second diagonal win
		for i in range((self.size + 1) - self.s):
			lineCount = 0
			for j in range(self.size - 1 - j):
				if(self.current_state[i][self.size - 1 - i - j] == "#" or self.current_state[i][self.size - 1 - i - j] == "."
					or self.current_state[i][self.size - 1 - i - j] != self.current_state[i][self.size - 1 - (i + 1) - j]):
					lineCount = 0
				else:
					lineCount += 1

				if(lineCount == self.s-1):
					return self.current_state[i][self.size - 1 - i - j]

		# Something (random diagonals)
		
		# Is whole board full?
		for i in range(0, self.size):
			for j in range(0, self.size):
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
			elif self.result == 'O':
				print('The winner is O!')
			elif self.result == '.':
				print("It's a tie!")
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
		for i in range(0, self.size):
			for j in range(0, self.size):
				if (self.current_state[i][j] == 'X'):
					num_X = num_X + 1
				else:
					return None
		return num_X				

	# Determining number of black pieces
	def num_black(self):
		for i in range(0, self.size):
			for j in range(0, self.size):
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
	b = 4
	pb = 0
	s = 3
	d1 = 2
	d2 = 2
	t = 5
	a = True
	play_mode = 'AI-AI'

	g = Game(3, 3, [1,2], 4, 2, 2, 5, True, 'AI-AI', recommend=True)
	g.play(algo=Game.ALPHABETA,player_x=Game.AI,player_o=Game.AI)
	g.play(algo=Game.MINIMAX,player_x=Game.AI,player_o=Game.HUMAN)

	gametrace_filename = "gameTrace-" + str(n)  + str(b) + str(s) + str(t) + ".txt"

	# Open Game Trace File
	f = open(gametrace_filename, "w")
	f.write("n=" + str(n) + " b=" + str(b) + " s=" + str(s) + " t=" + str(t))
	f.write("\n\nPlayer 1: " + " d=" + str(d1))
	f.write("\nPlayer 2: " + " d=" + str(d2))
	
	# f.write(str(g.draw_board()))

	f.close

if __name__ == "__main__":
	main()
