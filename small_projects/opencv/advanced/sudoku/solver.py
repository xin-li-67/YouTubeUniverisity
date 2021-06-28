def solve(board):
    find = find_empty(board)
    if not find:
        return True
    else:
        row, col = find
    
    for i in range(1, 10):
        if isValid(board, i, (row, col)):
            board[row][col] = i
            if solve(board):
                return True
            
            board[row][col] = 0
    
    return False

def isValid(board, num, pos):
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False
    
    curr_x = pos[1] // 3
    curr_y = pos[0] // 3

    for i in range(curr_y * 3, curr_y * 3 + 3):
        for j in range(curr_x * 3, curr_x * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True

def print_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")

def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)
                
    return None