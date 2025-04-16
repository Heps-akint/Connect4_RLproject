import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from flask import Flask, render_template, jsonify, request

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Out-of-place addition to prevent in-place errors
        out = F.relu(out)
        return out

# Define the ConnectFour Neural Network
class ConnectNet(nn.Module):
    def __init__(self, num_residual_blocks=6):
        super(ConnectNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # Stack multiple Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        self.conv_final = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 6 * 7, 512)
        self.dropout = nn.Dropout(p=0.6)  # Increased dropout for regularization
        self.fc_policy = nn.Linear(512, 7)
        self.fc_value = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 1, 6, 7)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.residual_blocks(x)
        x = F.relu(self.bn_final(self.conv_final(x)))
        x = x.view(-1, 64 * 6 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        policy_logits = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy_logits, value

# Define the Connect Four game environment
class ConnectFour:
    ROWS = 6
    COLS = 7

    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1  # Player 1 starts

    def make_move(self, col):
        for row in reversed(range(self.ROWS)):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.current_player *= -1  # Switch player
                return True
        return False  # Column is full

    def valid_moves(self):
        return [col for col in range(self.COLS) if self.board[0, col] == 0]

    def is_full(self):
        return np.all(self.board != 0)

    def check_winner(self):
        # Check horizontal locations for win
        for c in range(self.COLS - 3):
            for r in range(self.ROWS):
                piece = self.board[r][c]
                if piece != 0 and all(self.board[r][c + i] == piece for i in range(4)):
                    return piece

        # Check vertical locations for win
        for c in range(self.COLS):
            for r in range(self.ROWS - 3):
                piece = self.board[r][c]
                if piece != 0 and all(self.board[r + i][c] == piece for i in range(4)):
                    return piece

        # Check positively sloped diagonals
        for c in range(self.COLS - 3):
            for r in range(self.ROWS - 3):
                piece = self.board[r][c]
                if piece != 0 and all(self.board[r + i][c + i] == piece for i in range(4)):
                    return piece

        # Check negatively sloped diagonals
        for c in range(self.COLS - 3):
            for r in range(3, self.ROWS):
                piece = self.board[r][c]
                if piece != 0 and all(self.board[r - i][c + i] == piece for i in range(4)):
                    return piece

        return 0  # No winner

    def reset(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1

# MCTS Node
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = copy.deepcopy(state)
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

# UCB Score Calculation
def ucb_score(parent, child, c_puct=2.0):
    prior_score = c_puct * child.prior * math.sqrt(parent.visits) / (1 + child.visits)
    value_score = child.value / (1 + child.visits)
    return prior_score - value_score

# Backpropagate the value up the path
def backpropagate(path, value):
    for node in reversed(path):
        node.visits += 1
        node.value += value
        value = -value  # Flip the value for the opponent

# Monte Carlo Tree Search
def mcts_search(root, net, num_simulations=800):
    for _ in range(num_simulations):
        node = root
        path = []

        # Selection
        while node.children:
            # Select the child with the highest UCB score
            child = max(node.children.values(), key=lambda child: ucb_score(node, child))
            node = child
            path.append(node)

        # Expansion
        winner = node.state.check_winner()
        if winner == 0 and not node.state.is_full():
            # Prepare the state tensor
            state_tensor = torch.tensor(node.state.board, dtype=torch.float32).unsqueeze(0).to(device)
            net.eval()
            with torch.no_grad():
                policy_logits, value = net(state_tensor)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]

            valid_moves = node.state.valid_moves()
            policy = {idx: policy[idx] for idx in valid_moves}
            policy_sum = sum(policy.values())
            for idx in policy:
                policy[idx] /= policy_sum  # Normalize the probabilities

            # Add Dirichlet noise at the root node for exploration
            if node == root:
                dirichlet_alpha = 0.3
                epsilon = 0.25
                dirichlet_noise = np.random.dirichlet([dirichlet_alpha] * len(valid_moves))
                for i, idx in enumerate(valid_moves):
                    policy[idx] = (1 - epsilon) * policy[idx] + epsilon * dirichlet_noise[i]

            # Expand children
            for idx in valid_moves:
                child_state = copy.deepcopy(node.state)
                child_state.make_move(idx)
                child_node = MCTSNode(child_state, node)
                child_node.prior = policy[idx]
                node.children[idx] = child_node
            # Use the value estimate from the neural network
            backpropagate(path + [node], value.item())
        else:
            # Terminal node
            value = winner if winner != 0 else 0
            backpropagate(path + [node], value)
    # Choose the move with the most visits
    best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_move

# Load model
def load_model(model_path='connect4_best_model.pth'):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        net = ConnectNet(num_residual_blocks=6).to(device)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        print(f"Model loaded successfully from {model_path}")
        return net
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize Flask app
app = Flask(__name__)

# Global game state
game = ConnectFour()
net = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset():
    global game
    game.reset()
    return jsonify({'board': game.board.tolist(), 'current_player': game.current_player})

@app.route('/make_move', methods=['POST'])
def make_move():
    global game
    data = request.json
    col = data.get('col')
    sims = int(data.get('sims', 800))  # AI difficulty: number of MCTS simulations
    
    # Check if the column is valid
    if col not in game.valid_moves():
        return jsonify({'error': 'Invalid move'}), 400
    
    # Make the player's move
    game.make_move(col)
    
    # Check if the game is over after player's move
    winner = game.check_winner()
    if winner != 0 or game.is_full():
        return jsonify({
            'board': game.board.tolist(),
            'current_player': int(game.current_player),
            'game_over': True,
            'winner': int(winner)
        })
    
    # Make the AI's move
    root = MCTSNode(game)
    ai_move = mcts_search(root, net, num_simulations=sims)
    game.make_move(ai_move)
    
    # Check if the game is over after AI's move
    winner = game.check_winner()
    game_over = bool(winner != 0 or game.is_full())
    
    # Convert numpy int64 to Python int
    board_list = game.board.tolist()
    player = int(game.current_player)
    ai_move_int = int(ai_move)
    winner_int = int(winner)
    
    return jsonify({
        'board': board_list,
        'current_player': player,
        'ai_move': ai_move_int,
        'game_over': game_over,
        'winner': winner_int
    })

if __name__ == '__main__':
    app.run(debug=True)
