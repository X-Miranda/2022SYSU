import copy
from ChessBoard import *


class Evaluate(object):
    # 棋子棋力得分
    single_chess_point = {
        'c': 989,  # 车
        'm': 439,  # 马
        'p': 442,  # 炮
        's': 226,  # 士
        'x': 210,  # 象
        'z': 55,  # 卒
        'j': 65536  # 将
    }
    # 红兵（卒）位置得分
    red_bin_pos_point = [
        [1, 3, 9, 10, 12, 10, 9, 3, 1],
        [18, 36, 56, 95, 118, 95, 56, 36, 18],
        [15, 28, 42, 73, 80, 73, 42, 28, 15],
        [13, 22, 30, 42, 52, 42, 30, 22, 13],
        [8, 17, 18, 21, 26, 21, 18, 17, 8],
        [3, 0, 7, 0, 8, 0, 7, 0, 3],
        [-1, 0, -3, 0, 3, 0, -3, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # 红车位置得分
    red_che_pos_point = [
        [185, 195, 190, 210, 220, 210, 190, 195, 185],
        [185, 203, 198, 230, 245, 230, 198, 203, 185],
        [180, 198, 190, 215, 225, 215, 190, 198, 180],
        [180, 200, 195, 220, 230, 220, 195, 200, 180],
        [180, 190, 180, 205, 225, 205, 180, 190, 180],
        [155, 185, 172, 215, 215, 215, 172, 185, 155],
        [110, 148, 135, 185, 190, 185, 135, 148, 110],
        [100, 115, 105, 140, 135, 140, 105, 115, 110],
        [115, 95, 100, 155, 115, 155, 100, 95, 115],
        [20, 120, 105, 140, 115, 150, 105, 120, 20]
    ]
    # 红马位置得分
    red_ma_pos_point = [
        [80, 105, 135, 120, 80, 120, 135, 105, 80],
        [80, 115, 200, 135, 105, 135, 200, 115, 80],
        [120, 125, 135, 150, 145, 150, 135, 125, 120],
        [105, 175, 145, 175, 150, 175, 145, 175, 105],
        [90, 135, 125, 145, 135, 145, 125, 135, 90],
        [80, 120, 135, 125, 120, 125, 135, 120, 80],
        [45, 90, 105, 190, 110, 90, 105, 90, 45],
        [80, 45, 105, 105, 80, 105, 105, 45, 80],
        [20, 45, 80, 80, -10, 80, 80, 45, 20],
        [20, -20, 20, 20, 20, 20, 20, -20, 20]
    ]
    # 红炮位置得分
    red_pao_pos_point = [
        [190, 180, 190, 70, 10, 70, 190, 180, 190],
        [70, 120, 100, 90, 150, 90, 100, 120, 70],
        [70, 90, 80, 90, 200, 90, 80, 90, 70],
        [60, 80, 60, 50, 210, 50, 60, 80, 60],
        [90, 50, 90, 70, 220, 70, 90, 50, 90],
        [120, 70, 100, 60, 230, 60, 100, 70, 120],
        [10, 30, 10, 30, 120, 30, 10, 30, 10],
        [30, -20, 30, 20, 200, 20, 30, -20, 30],
        [30, 10, 30, 30, -10, 30, 30, 10, 30],
        [20, 20, 20, 20, -10, 20, 20, 20, 20]
    ]
    # 红将位置得分
    red_jiang_pos_point = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 9750, 9800, 9750, 0, 0, 0],
        [0, 0, 0, 9900, 9900, 9900, 0, 0, 0],
        [0, 0, 0, 10000, 10000, 10000, 0, 0, 0],
    ]
    # 红相或士位置得分
    red_xiang_shi_pos_point = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 60, 0, 0, 0, 60, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [80, 0, 0, 80, 90, 80, 0, 0, 80],
        [0, 0, 0, 0, 0, 120, 0, 0, 0],
        [0, 0, 70, 100, 0, 100, 70, 0, 0],
    ]

    red_pos_point = {
        'z': red_bin_pos_point,
        'm': red_ma_pos_point,
        'c': red_che_pos_point,
        'j': red_jiang_pos_point,
        'p': red_pao_pos_point,
        'x': red_xiang_shi_pos_point,
        's': red_xiang_shi_pos_point
    }

    def __init__(self, team):
        self.team = team

    def get_single_chess_point(self, chess: Chess):
        if chess.team == self.team:
            return self.single_chess_point[chess.name]
        else:
            return -1 * self.single_chess_point[chess.name]

    def get_chess_pos_point(self, chess: Chess):
        red_pos_point_table = self.red_pos_point[chess.name]
        if chess.team == 'r':
            pos_point = red_pos_point_table[chess.row][chess.col]
        else:
            pos_point = red_pos_point_table[9 - chess.row][chess.col]
        if chess.team != self.team:
            pos_point *= -1
        return pos_point

    def evaluate(self, chessboard: ChessBoard):
        point = 0
        for chess in chessboard.get_chess():
            point += self.get_single_chess_point(chess)
            point += self.get_chess_pos_point(chess)
        return point


class ChessMap(object):
    def __init__(self, chessboard: ChessBoard):
        self.chess_map = copy.deepcopy(chessboard.chessboard_map)


class MyAI(object):
    def __init__(self, team, deepest=3):
        self.team = team
        self.deepest = deepest   #最深度
        self.evaluate_class = Evaluate(self.team) #评估分数
        self.best_move = None  #最佳步
        self.last_move = []   #记前面几步，以防陷入循环

#获取下一步最佳移动。它调用了alpha_beta方法进行Alpha-Beta搜索，找到最佳的移动。
    def get_next_step(self, chessboard: ChessBoard):
        self.alpha_beta(chessboard, 0, -float('inf'), float('inf'), True)

        if self.best_move:#如果存在最佳移动，将最佳移动解包为当前行、列和下一个行、列。
            cur_row, cur_col, nxt_row, nxt_col = self.best_move

            # 去重
            self.last_move.append((nxt_row, nxt_col, cur_row, cur_col))
            if len(self.last_move) > 3:
                self.last_move.pop(0)

            # 加深
            if len(chessboard.get_chess()) <= 18:
                self.deepest = 4
            if len(chessboard.get_chess()) <= 15:
                self.deepest = 5

            return cur_row, cur_col, nxt_row, nxt_col
        return None

    @staticmethod
    def make_move(chessboard, chess, new_row, new_col):
        # 记录旧位置和棋子
        old_row, old_col = chess.row, chess.col
        taken_chess = chessboard.chessboard_map[new_row][new_col] #被吃掉的棋子
        # 执行移动
        chessboard.chessboard_map[old_row][old_col] = None #移除旧位置
        chessboard.chessboard_map[new_row][new_col] = chess #在新位置放置棋子
        chess.update_position(new_row, new_col)  #更新棋子的位置信息
        return old_row, old_col, taken_chess

    @staticmethod
    def undo_move(chessboard, chess, old_row, old_col, taken_chess):
        # 撤销移动
        chessboard.chessboard_map[chess.row][chess.col] = taken_chess
        chessboard.chessboard_map[old_row][old_col] = chess
        chess.update_position(old_row, old_col)

#其中包含Minimax算法和剪枝，在每一层递归中，根据当前轮到的玩家是最大化玩家还是最小化玩家，选择最大化或最小化评估值。
    def alpha_beta(self, chessboard, depth, alpha, beta, max_player):
        if depth == self.deepest:  #限制搜索深度
            return self.evaluate_class.evaluate(chessboard)

        if max_player: #最大化玩家
            max_eval = -float('inf') #初始化最大评估值为负无穷
            for chess in chessboard.get_chess(): 
                if chess.team == self.team: #检查当前棋子是否属于当前玩家的队伍。
                    for nxt_row, nxt_col in chessboard.get_put_down_position(chess): #遍历当前棋子可以移动到的下一个可能位置。

                        if (chess.row, chess.col, nxt_row, nxt_col) in self.last_move:
                            continue #检查这个移动是否已经在上一次移动中使用过，如果是则跳过。

                        old_row, old_col, taken_chess = self.make_move(chessboard, chess, nxt_row, nxt_col)
                        eval = self.alpha_beta(chessboard, depth + 1, alpha, beta, False) #递归调用alpha_beta函数，以当前棋盘状态的变化为基础，继续搜索下一层。
                        self.undo_move(chessboard, chess, old_row, old_col, taken_chess) #恢复之前的移动，以便进一步尝试其他移动。
                        
                        if eval > max_eval:
                            max_eval = eval
                            if depth == 0:
                                self.best_move = (old_row, old_col, nxt_row, nxt_col)
                        alpha = max(alpha, eval)
                        if beta <= alpha: #如果beta值小于或等于alpha值，表示当前节点不再具有最佳解，可以剪枝并退出循环。
                            break
            return max_eval
        else:
            min_eval = float('inf')
            for chess in chessboard.get_chess():
                if chess.team != self.team:
                    for nxt_row, nxt_col in chessboard.get_put_down_position(chess):
                        if (chess.row, chess.col, nxt_row, nxt_col) in self.last_move:
                            continue
                        old_row, old_col, taken_chess = self.make_move(chessboard, chess, nxt_row, nxt_col)
                        eval = self.alpha_beta(chessboard, depth + 1, alpha, beta, True)
                        self.undo_move(chessboard, chess, old_row, old_col, taken_chess)
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval