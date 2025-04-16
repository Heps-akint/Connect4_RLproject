document.addEventListener('DOMContentLoaded', () => {
    // Initialize game board
    const boardBg = document.querySelector('.board-bg');
    const piecesContainer = document.querySelector('.pieces');
    const columnSelector = document.querySelector('.column-selector');
    const statusElement = document.getElementById('status');
    const resetButton = document.getElementById('reset-btn');
    const spinner = document.getElementById('spinner');
    const difficultySelect = document.getElementById('difficulty');

    const ROWS = 6;
    const COLS = 7;
    let gameBoard = Array(ROWS).fill().map(() => Array(COLS).fill(0));
    let gameOver = false;
    let isPlayerTurn = true; // Player always goes first

    // Create the game board
    function createBoard() {
        // Clear existing elements
        boardBg.innerHTML = '';
        piecesContainer.innerHTML = '';
        columnSelector.innerHTML = '';

        // Create the board cells (empty slots)
        for (let row = 0; row < ROWS; row++) {
            for (let col = 0; col < COLS; col++) {
                const cell = document.createElement('div');
                cell.classList.add('board-cell');
                cell.dataset.row = row;
                cell.dataset.col = col;
                boardBg.appendChild(cell);
            }
        }

        // Create column selectors for user input
        for (let col = 0; col < COLS; col++) {
            const colBtn = document.createElement('div');
            colBtn.classList.add('column-btn');
            colBtn.dataset.col = col;
            colBtn.addEventListener('click', () => handleColumnClick(col));
            columnSelector.appendChild(colBtn);
        }
    }

    // Handle column click
    function handleColumnClick(col) {
        if (gameOver || !isPlayerTurn) return;

        // Ensure the column isn't full
        if (gameBoard[0][col] !== 0) {
            statusElement.textContent = "That column is full. Try another one.";
            return;
        }

        statusElement.textContent = "AI is thinking...";
        isPlayerTurn = false;

        // Show spinner and disable input with chosen difficulty
        const sims = parseInt(difficultySelect.value, 10);
        spinner.classList.remove('hidden');
        columnSelector.style.pointerEvents = 'none';

        // Send the move to the server
        fetch('/make_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ col: col, sims: sims })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error(data.error);
                statusElement.textContent = data.error;
                isPlayerTurn = true;
                return;
            }

            // Update the game board
            gameBoard = data.board;
            renderPieces();

            // Check if game is over
            if (data.game_over) {
                gameOver = true;
                if (data.winner === 1) {
                    statusElement.textContent = "You win! Congratulations!";
                    highlightWinningPieces();
                } else if (data.winner === -1) {
                    statusElement.textContent = "AI wins! Better luck next time.";
                    highlightWinningPieces();
                } else {
                    statusElement.textContent = "It's a draw!";
                }
            } else {
                isPlayerTurn = true;
                statusElement.textContent = "Your turn! Click a column to drop your piece.";
            }
        })
        .catch(error => {
            console.error('Error:', error);
            statusElement.textContent = "An error occurred. Please try again.";
            isPlayerTurn = true;
        })
        .finally(() => {
            spinner.classList.add('hidden');
            columnSelector.style.pointerEvents = 'auto';
        });
    }

    // Render pieces based on game state
    function renderPieces() {
        piecesContainer.innerHTML = '';
        
        for (let row = 0; row < ROWS; row++) {
            for (let col = 0; col < COLS; col++) {
                if (gameBoard[row][col] !== 0) {
                    const piece = document.createElement('div');
                    piece.classList.add('piece');
                    piece.dataset.row = row;
                    piece.dataset.col = col;
                    
                    if (gameBoard[row][col] === 1) {
                        piece.classList.add('player-piece');
                    } else {
                        piece.classList.add('ai-piece');
                    }
                    
                    // Position the piece
                    const boardCell = boardBg.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                    const rect = boardCell.getBoundingClientRect();
                    const boardRect = boardBg.getBoundingClientRect();
                    
                    piece.style.left = `${rect.left - boardRect.left + (rect.width - 60) / 2}px`;
                    piece.style.top = `${rect.top - boardRect.top + (rect.height - 60) / 2}px`;
                    
                    piecesContainer.appendChild(piece);
                    // Animate piece drop
                    setTimeout(() => piece.classList.add('drop'), 50);
                }
            }
        }
    }

    // Highlight winning pieces
    function highlightWinningPieces() {
        // Check horizontal wins
        for (let row = 0; row < ROWS; row++) {
            for (let col = 0; col <= COLS - 4; col++) {
                const piece = gameBoard[row][col];
                if (piece !== 0) {
                    let win = true;
                    for (let i = 1; i < 4; i++) {
                        if (gameBoard[row][col + i] !== piece) {
                            win = false;
                            break;
                        }
                    }
                    if (win) {
                        for (let i = 0; i < 4; i++) {
                            const winPiece = piecesContainer.querySelector(`[data-row="${row}"][data-col="${col + i}"]`);
                            if (winPiece) winPiece.classList.add('winning-piece');
                        }
                        return;
                    }
                }
            }
        }

        // Check vertical wins
        for (let col = 0; col < COLS; col++) {
            for (let row = 0; row <= ROWS - 4; row++) {
                const piece = gameBoard[row][col];
                if (piece !== 0) {
                    let win = true;
                    for (let i = 1; i < 4; i++) {
                        if (gameBoard[row + i][col] !== piece) {
                            win = false;
                            break;
                        }
                    }
                    if (win) {
                        for (let i = 0; i < 4; i++) {
                            const winPiece = piecesContainer.querySelector(`[data-row="${row + i}"][data-col="${col}"]`);
                            if (winPiece) winPiece.classList.add('winning-piece');
                        }
                        return;
                    }
                }
            }
        }

        // Check diagonal wins (positive slope)
        for (let row = 3; row < ROWS; row++) {
            for (let col = 0; col <= COLS - 4; col++) {
                const piece = gameBoard[row][col];
                if (piece !== 0) {
                    let win = true;
                    for (let i = 1; i < 4; i++) {
                        if (gameBoard[row - i][col + i] !== piece) {
                            win = false;
                            break;
                        }
                    }
                    if (win) {
                        for (let i = 0; i < 4; i++) {
                            const winPiece = piecesContainer.querySelector(`[data-row="${row - i}"][data-col="${col + i}"]`);
                            if (winPiece) winPiece.classList.add('winning-piece');
                        }
                        return;
                    }
                }
            }
        }

        // Check diagonal wins (negative slope)
        for (let row = 0; row <= ROWS - 4; row++) {
            for (let col = 0; col <= COLS - 4; col++) {
                const piece = gameBoard[row][col];
                if (piece !== 0) {
                    let win = true;
                    for (let i = 1; i < 4; i++) {
                        if (gameBoard[row + i][col + i] !== piece) {
                            win = false;
                            break;
                        }
                    }
                    if (win) {
                        for (let i = 0; i < 4; i++) {
                            const winPiece = piecesContainer.querySelector(`[data-row="${row + i}"][data-col="${col + i}"]`);
                            if (winPiece) winPiece.classList.add('winning-piece');
                        }
                        return;
                    }
                }
            }
        }
    }

    // Reset game
    function resetGame() {
        fetch('/reset', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            gameBoard = data.board;
            gameOver = false;
            isPlayerTurn = true;
            statusElement.textContent = "Your turn! Click a column to drop your piece.";
            renderPieces();
        })
        .catch(error => {
            console.error('Error:', error);
            statusElement.textContent = "Error resetting game.";
        });
    }

    // Add event listener to reset button
    resetButton.addEventListener('click', resetGame);

    // Initialize the game
    createBoard();
    resetGame();

    // Responsive design adjustments
    window.addEventListener('resize', () => {
        renderPieces();
    });
});
