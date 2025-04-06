# Connect4 AI Web Frontend Guide

## Overview

This is a web-based frontend for playing against the trained Connect4 AI model. The implementation includes:

- A Flask backend that loads the trained PyTorch model and handles game logic
- An interactive HTML/CSS/JavaScript frontend for gameplay
- Full integration with the trained model to ensure consistent AI performance

## Installation

1. Make sure you have Python installed (Python 3.6 or higher recommended)

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the trained model file `connect4_best_model.pth` is in the root directory of the project

## Running the Application

1. Start the Flask server:

```bash
python app.py
```

2. Open a web browser and navigate to:

```
http://127.0.0.1:5000
```

## How to Play

1. The game board will appear with empty slots
2. You play as the red pieces (Player 1) and go first
3. Click on any column to drop your piece
4. The AI (yellow pieces) will automatically make its move
5. The first player to connect four pieces in a row (horizontally, vertically, or diagonally) wins
6. Click "New Game" to reset the board and play again

## Technical Details

- The backend uses the same neural network architecture and MCTS algorithm as in the Jupyter notebook
- The AI's thinking process is handled by the Flask server, which uses the trained model to predict optimal moves
- The game state is synchronized between the frontend and backend to ensure accuracy

## Troubleshooting

If you encounter any issues:

1. Ensure the model file exists and is named correctly
2. Check that you have all required dependencies installed
3. Look for error messages in the Flask server console
4. Make sure your browser has JavaScript enabled
