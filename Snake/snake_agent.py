import numpy as np
import helper
from helper import *
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr) 
    #   gamma which is another parameter helpful in calculating next move, in other words  
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        # self.board = np.zeros((DISPLAY_SIZE // GRID_SIZE, DISPLAY_SIZE // GRID_SIZE))
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()

        self.previous_move = 0


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None


    #   This is a function you should write. 
    #   Function Helper:IT gets the current state, and based on the 
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the 
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on. 
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        print("IN helper_func")
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE

        head_x, head_y, body, food_x, food_y = state

        # Calculate the number of adjacent walls
        num_adjacent_walls_x = 1 if head_x == 40 or head_x == 520 else 2 if [head_x, head_y] in [[40, 40], [40, 520], [520, 40], [520, 520]] else 0
        num_adjacent_walls_y = 1 if head_y == 40 or head_y == 520 else 2 if [head_x, head_y] in [[40, 40], [40, 520], [520, 40], [520, 520]] else 0

        # Calculate the direction of the food
        food_dir_x = 0 if food_x < head_x else 1 if food_x > head_x else 2
        food_dir_y = 0 if food_y < head_y else 1 if food_y > head_y else 2

        # Calculate the number of adjacent body parts
        num_adjacent_body_top = 1 if [head_x, head_y - 40] in body else 0
        num_adjacent_body_bottom = 1 if [head_x, head_y + 40] in body else 0
        num_adjacent_body_left = 1 if [head_x - 40, head_y] in body else 0
        num_adjacent_body_right = 1 if [head_x + 40, head_y] in body else 0


        board_width = DISPLAY_SIZE // GRID_SIZE
        board_height = DISPLAY_SIZE // GRID_SIZE

        board = np.zeros((board_width, board_height))
        head_x, head_y, body, food_x, food_y = state
        head = (head_x // GRID_SIZE, head_y // GRID_SIZE)

        # Update board with the snake and food locations
        for pos in body:
            board[pos[0] // GRID_SIZE, pos[1] // GRID_SIZE] = 1
        board[food_x // GRID_SIZE, food_y // GRID_SIZE] = 2
        board[head_x // GRID_SIZE, head_y // GRID_SIZE] = 3

        # print(board)

        # Get possible moves
        possible_moves = []
    
        # Check if the snake can move up
        if self.previous_move != 1 and head[1] > 1 and board[head[0]][head[1] - 1] != 1 and (head[0], head[1] - 40) not in body:
            possible_moves.append(0)

        # Check if the snake can move down
        if self.previous_move != 0 and head[1] < 12 and board[head[0]][head[1] + 1] != 1 and (head[0], head[1] + 40) not in body:
            possible_moves.append(1)

        # Check if the snake can move left
        if self.previous_move != 3 and head[0] > 1 and board[head[0] - 1][head[1]] != 1 and (head[0] - 40, head[1]) not in body:
            possible_moves.append(2)

        # Check if the snake can move right
        if self.previous_move != 2 and head[0] < 12 and board[head[0] + 1][head[1]] != 1 and (head[0] + 40, head[1]) not in body:
            possible_moves.append(3)

        if(possible_moves):
            # print("previous move:   --->   ", self.previous_move)
            self.previous_move = possible_moves[0]
        else:
            possible_moves.append(0)

        return (num_adjacent_walls_x, num_adjacent_walls_y, food_dir_x, food_dir_y,
                num_adjacent_body_top, num_adjacent_body_bottom, num_adjacent_body_left, num_adjacent_body_right, possible_moves)



    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -40
        elif points > self.points:
            return 1
        else:
            return -0.1

    #   This is the code you need to write. 
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make 
    #   using the compute reward function defined above. 
    #   This function also keeps track of the fact that we are in 
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make. 
    #   the LPC variable can be used to determine the learning rate (lr), but if 
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively. 
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    def agent_action(self, state, points, dead):
        print("IN AGENT_ACTION")
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        


        # Get the current state and compute the reward
        s = self.helper_func(state)
        r = self.compute_reward(points, dead)
        maxQ = 0  # Initialize maxQ to a default value
        self.lr = 0.7
        head_x, head_y, body, food_x, food_y = state


        # Update points and reset actions if the snake died
        if dead:
            self.reset()

        # Explore with probability Ne
        if self._train and np.random.uniform() < self.Ne:
            a = np.random.choice([0,1,2,3])
        else:
            a = np.random.choice([0,1,2,3])
            minD = float("inf")
            for next_move in (s[8]):
                head_x, head_y, body, food_x, food_y = state
                if next_move == 0: head_y -= 40
                if next_move == 1: head_y += 40
                if next_move == 2: head_x -= 40
                if next_move == 3: head_x += 40
                d = ((head_y - food_y)**2 + (head_x - food_x)**2)**0.5
                # print(d)
                # print("action ---->  ", next_move)
                if d < minD:
                    minD = d
                    a = next_move
                    # print("a   ->   ", a)


            self.previous_move = a

            # Store the current state and action as the previous ones
            self.s = s
            self.a = a

            # Store the current points as the previous one
            self.points = points

            #self.save_model()
            
            return a
    

        # Check if the agent is in training mode
        if self._train:

            
            # Get the current state and compute the reward
            s = self.helper_func(state)
            r = self.compute_reward(points, dead)
            maxQ = 0  # Initialize maxQ to a default value
            
            # Explore with probability Ne
            if self._train and np.random.uniform() < self.Ne:
                a = np.random.choice([0,1,2,3])
            else:
                # Get the action with the highest Q value
                q = [self.Q[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], a] for a in self.actions]
                maxQ = max(q)
                count = q.count(maxQ)
                # In case there're several state-action max values 
                # we select a random one among them
                if count > 1:
                    best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                    i = np.random.choice(best)
                else:
                    i = q.index(maxQ)

                # Select the action with the highest Q value
                a = self.actions[i]
                

            # Update the Q table
            if self.a is not None:
                # Update the Q-value for the current state and action based on the reward
                self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], a] += self.lr * (r + self.gamma * self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], self.previous_move] - self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], self.a])


            # Update the N table
            self.N[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], a] += 1

            # Store the current state and action as the previous ones
            self.s = s
            self.a = a

            # Store the current points as the previous one
            self.points = points

            #self.save_model()

            self.previous_move = a
            
            return a
        
        else:
            # Get the possible moves
            possible_moves = self.helper_func(state)

            # print("possible moves: ", possible_moves)
            if possible_moves[8]: return random.choice(possible_moves[8]) 
            else: return 0
            