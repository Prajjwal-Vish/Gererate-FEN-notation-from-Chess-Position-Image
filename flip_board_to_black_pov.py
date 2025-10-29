import chess

def reverse_rank(rank_str):
    """Reverse files in a single rank string (mirrors horizontally)."""
    tokens = []
    i = 0
    while i < len(rank_str):
        if rank_str[i].isdigit():
            tokens.append(rank_str[i])
            i += 1
        else:
            tokens.append(rank_str[i])
            i += 1
    return ''.join(tokens[::-1])

def black_perspective_fen(fen_str):
    """Full 180° rotation: reverse ranks + reverse files per rank."""
    board = chess.Board(fen_str)
    position = board.fen().split(' ')[0]
    ranks = position.split('/')
    # Vertical flip: reverse rank order
    ranks = ranks[::-1]
    # Horizontal flip: reverse files in each
    reversed_ranks = [reverse_rank(r) for r in ranks]
    new_position = '/'.join(reversed_ranks)
    # Keep rest of FEN unchanged
    rest = ' '.join(board.fen().split(' ')[1:])
    return new_position + ' ' + rest

# Example
original = "6R1/rPP3bP/3K2N1/6q1/1Q1Pp3/1p1p4/2p3pp/3k1b1r w - - 0 1"
fixed = black_perspective_fen(original)
print(fixed)  # r1b1k3/pp3p2/4p1p1/3pP1Q1/1q6/1N2K3/Pb3PPr/1R6 w - - 0 1