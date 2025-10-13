import sys
import argparse

from Game import *
from Dot import *
from ChessBoard import *
from MyAI import *
from ChessAI import *


def ai_move(ai, game, chessboard, screen):
    cur_row, cur_col, nxt_row, nxt_col = ai.get_next_step(chessboard)
    ClickBox(screen, cur_row, cur_col)
    chessboard.move_chess(nxt_row, nxt_col)
    ClickBox.clean()
    # 产生将军局面
    if chessboard.judge_attack_general(game.get_player()):
        print('--- 黑方被将军 ---\n') if game.get_player() == 'r' else print('--- 红方被将军 ---\n')
        if chessboard.judge_win(game.get_player()):
            print('--- 红方获胜 ---\n') if game.get_player() == 'r' else print('--- 黑方获胜 ---\n')
            game.set_win(game.get_player())
            return
        else:
            game.set_attack(True)
    # 产生必胜局面
    else:
        if chessboard.judge_win(game.get_player()):
            print('--- 红方获胜 ---\n') if game.get_player() == 'r' else print('--- 黑方获胜 ---\n')
            game.set_win(game.get_player())
            return
        game.set_attack(False)
    # 产生和棋局面
    if chessboard.judge_draw():
        print('--- 和棋 ---\n')
        game.set_draw()

    game.back_button.add_history(chessboard.get_chessboard_str_map())
    game.exchange()
    return



def main():
    # 初始化pygame
    pygame.init()
    # 创建用来显示画面的对象（理解为相框）
    screen = pygame.display.set_mode((750, 667))
    # 游戏背景图片
    background_img = pygame.image.load("images/bg.jpg")
    # 游戏棋盘
    # chessboard_img = pygame.image.load("images/bg.png")
    # 创建棋盘对象
    chessboard = ChessBoard(screen)
    # 创建计时器
    clock = pygame.time.Clock()
    # 创建游戏对象（像当前走棋方、游戏是否结束等都封装到这个对象中）
    game = Game(screen, chessboard)
    game.back_button.add_history(chessboard.get_chessboard_str_map())

    # 是否调换先后手（False MyAI红方先手    True chessAI红方先手）
    reverse = False

    if reverse:
        player1 = ChessAI(game.red)
        player2 = MyAI(game.black)
        print(f'红色方：ChessAI\n'
              f'黑色方：MyAI\n')
    else:
        player1 = MyAI(game.red)
        player2 = ChessAI(game.black)
        print(f'红色方：MyAI\n'
              f'黑色方：ChessAI\n')

    counter = 0

    while not game.show_win and not game.show_draw:
        currentAI = player1 if counter % 2 == 0 else player2
        counter += 1

        ai_move(currentAI, game, chessboard, screen)

        #update_display(game.screen, background_img, game.chessboard, game)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # 显示游戏背景
        screen.blit(background_img, (0, 0))
        screen.blit(background_img, (0, 270))
        screen.blit(background_img, (0, 540))

        # # 显示棋盘
        # # screen.blit(chessboard_img, (50, 50))
        # chessboard.show()
        #
        # # 显示棋盘上的所有棋子
        # # for line_chess in chessboard_map:
        # for line_chess in chessboard.chessboard_map:
        #     for chess in line_chess:
        #         if chess:
        #             # screen.blit(chess[0], chess[1])
        #             chess.show()

        # 显示棋盘以及棋子
        chessboard.show_chessboard_and_chess()

        # 标记点击的棋子
        ClickBox.show()

        # 显示可以落子的位置图片
        Dot.show_all()

        # 显示游戏相关信息
        game.show()

        # 显示screen这个相框的内容（此时在这个相框中的内容像照片、文字等会显示出来）
        pygame.display.update()

        # FPS（每秒钟显示画面的次数）
        clock.tick(120)  # 通过一定的延时，实现1秒钟能够循环60次

if __name__ == '__main__':
    main()
