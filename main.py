import sys
import numpy as np
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QMessageBox, QTextEdit)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThread
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QTextCursor


class GomokuBoard(QWidget):
    """五子棋棋盘组件"""
    move_made = Signal(int, int, int)  # 发送信号：行, 列, 玩家(1=人类, 2=AI)
    game_over = Signal(int)  # 发送信号：获胜者(0=平局, 1=人类, 2=AI)

    def __init__(self, size=15, parent=None):
        super().__init__(parent)
        self.size = size  # 棋盘大小，默认15x15
        self.cell_size = 30  # 每个格子的大小
        self.board = np.zeros((size, size), dtype=int)  # 0=空, 1=人类, 2=AI
        self.current_player = 1  # 1=人类先行
        self.game_over_flag = False

        # 设置窗口大小
        self.setMinimumSize(size * self.cell_size, size * self.cell_size)
        self.setMaximumSize(size * self.cell_size, size * self.cell_size)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制棋盘背景
        painter.fillRect(self.rect(), QColor(240, 190, 100))

        # 绘制网格线
        pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(pen)

        for i in range(self.size):
            # 横线
            painter.drawLine(
                self.cell_size // 2,
                self.cell_size // 2 + i * self.cell_size,
                self.cell_size // 2 + (self.size - 1) * self.cell_size,
                self.cell_size // 2 + i * self.cell_size
            )
            # 竖线
            painter.drawLine(
                self.cell_size // 2 + i * self.cell_size,
                self.cell_size // 2,
                self.cell_size // 2 + i * self.cell_size,
                self.cell_size // 2 + (self.size - 1) * self.cell_size
            )

        # 绘制天元和星位
        star_positions = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        for x, y in star_positions:
            if 0 <= x < self.size and 0 <= y < self.size:
                painter.setBrush(QBrush(QColor(0, 0, 0)))
                painter.drawEllipse(
                    self.cell_size // 2 + x * self.cell_size - 3,
                    self.cell_size // 2 + y * self.cell_size - 3,
                    6, 6
                )

        # 绘制棋子
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:  # 人类（黑棋）
                    painter.setBrush(QBrush(QColor(0, 0, 0)))
                    painter.drawEllipse(
                        self.cell_size // 2 + j * self.cell_size - self.cell_size // 2 + 2,
                        self.cell_size // 2 + i * self.cell_size - self.cell_size // 2 + 2,
                        self.cell_size - 4,
                        self.cell_size - 4
                    )
                elif self.board[i][j] == 2:  # AI（白棋）
                    painter.setBrush(QBrush(QColor(255, 255, 255)))
                    painter.drawEllipse(
                        self.cell_size // 2 + j * self.cell_size - self.cell_size // 2 + 2,
                        self.cell_size // 2 + i * self.cell_size - self.cell_size // 2 + 2,
                        self.cell_size - 4,
                        self.cell_size - 4
                    )
                    # 白棋画边框
                    painter.setPen(QPen(QColor(0, 0, 0), 1))
                    painter.drawEllipse(
                        self.cell_size // 2 + j * self.cell_size - self.cell_size // 2 + 2,
                        self.cell_size // 2 + i * self.cell_size - self.cell_size // 2 + 2,
                        self.cell_size - 4,
                        self.cell_size - 4
                    )

    def mousePressEvent(self, event):
        if self.game_over_flag or self.current_player != 1:
            return

        # 计算落子位置
        x = event.position().x()
        y = event.position().y()

        col = int(round((x - self.cell_size // 2) / self.cell_size))
        row = int(round((y - self.cell_size // 2) / self.cell_size))

        # 检查位置是否有效
        if 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0:
            self.board[row][col] = 1
            self.move_made.emit(row, col, 1)
            self.update()

            # 检查游戏是否结束
            if self.check_win(row, col, 1):
                self.game_over_flag = True
                self.game_over.emit(1)
                return

            # 检查是否平局
            if np.all(self.board != 0):
                self.game_over_flag = True
                self.game_over.emit(0)
                return

            # 切换到AI回合
            self.current_player = 2

    def make_ai_move(self, row, col):
        """AI落子"""
        if self.game_over_flag or self.current_player != 2:
            return

        if 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0:
            self.board[row][col] = 2
            self.move_made.emit(row, col, 2)
            self.update()

            # 检查游戏是否结束
            if self.check_win(row, col, 2):
                self.game_over_flag = True
                self.game_over.emit(2)
                return

            # 检查是否平局
            if np.all(self.board != 0):
                self.game_over_flag = True
                self.game_over.emit(0)
                return

            # 切换到人类回合
            self.current_player = 1

    def check_win(self, row, col, player):
        """检查是否获胜"""
        directions = [
            [(0, 1), (0, -1)],  # 水平
            [(1, 0), (-1, 0)],  # 垂直
            [(1, 1), (-1, -1)],  # 对角线
            [(1, -1), (-1, 1)]  # 反对角线
        ]

        for dir_pair in directions:
            count = 1  # 已落子的位置

            # 检查两个方向
            for dr, dc in dir_pair:
                r, c = row + dr, col + dc
                while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                    count += 1
                    r += dr
                    c += dc

            if count >= 5:
                return True

        return False

    def reset_board(self):
        """重置棋盘"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.game_over_flag = False
        self.update()


class ImprovedQLearningAI:
    """具备基础棋理的Q-learning AI"""

    def __init__(self, board_size=15, learning_rate=0.1, discount_factor=0.9, epsilon=0.3):
        self.board_size = board_size
        self.learning_rate = learning_rate  # 学习率
        self.discount_factor = discount_factor  # 折扣因子
        self.epsilon = epsilon  # 探索率

        # Q表：存储每个状态-动作对的价值
        self.q_table = {}

        # 记录上一步的状态和动作，用于学习
        self.prev_state = None
        self.prev_action = None

        # 棋型评分表（基础常识）
        self.pattern_scores = {
            "win": 100000,  # 能形成五连，必胜
            "four_open": 10000,  # 活四（两端都可形成五连）
            "four_semi": 5000,  # 冲四（一端可形成五连）
            "three_open": 1000,  # 活三
            "three_semi": 500,  # 眠三
            "two_open": 100,  # 活二
            "two_semi": 50  # 眠二
        }

    def state_to_key(self, board):
        """将棋盘状态转换为可哈希的键"""
        return tuple(tuple(row) for row in board)

    def get_q_value(self, state_key, action):
        """获取Q值，如果不存在则初始化为0"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            # 初始Q值基于棋型评分，而不是0，赋予基础常识
            row, col = action
            temp_board = np.array(state_key)
            temp_board[row, col] = 2  # 假设AI落子
            score = self.evaluate_position(temp_board, row, col, 2)
            self.q_table[state_key][action] = score * 0.1  # 基础分数作为初始Q值
        return self.q_table[state_key][action]

    def count_consecutive(self, board, row, col, player, dr, dc):
        """计算在某个方向上连续的棋子数"""
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r][c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def evaluate_position(self, board, row, col, player):
        """评估落子位置的价值，基于基础棋理"""
        if board[row][col] != 0:
            return 0  # 非空位置价值为0

        # 临时落子，评估该位置
        temp_board = board.copy()
        temp_board[row][col] = player

        total_score = 0
        directions = [
            [(0, 1), (0, -1)],  # 水平
            [(1, 0), (-1, 0)],  # 垂直
            [(1, 1), (-1, -1)],  # 对角线
            [(1, -1), (-1, 1)]  # 反对角线
        ]

        for dir_pair in directions:
            # 计算当前方向连续的棋子数
            count = 1  # 当前落子
            blocked_ends = 0

            # 检查两个方向
            for dr, dc in dir_pair:
                consecutive = self.count_consecutive(temp_board, row, col, player, dr, dc)
                count += consecutive

                # 检查该方向是否被阻挡
                r = row + dr * (consecutive + 1)
                c = col + dc * (consecutive + 1)
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if temp_board[r][c] == 3 - player:  # 对方棋子阻挡
                        blocked_ends += 1
                else:
                    blocked_ends += 1  # 边界阻挡

            # 根据连子数和阻挡情况评分
            if count >= 5:
                total_score += self.pattern_scores["win"]
            elif count == 4:
                if blocked_ends == 0:  # 活四
                    total_score += self.pattern_scores["four_open"]
                elif blocked_ends == 1:  # 冲四
                    total_score += self.pattern_scores["four_semi"]
            elif count == 3:
                if blocked_ends == 0:  # 活三
                    total_score += self.pattern_scores["three_open"]
                elif blocked_ends == 1:  # 眠三
                    total_score += self.pattern_scores["three_semi"]
            elif count == 2:
                if blocked_ends == 0:  # 活二
                    total_score += self.pattern_scores["two_open"]
                elif blocked_ends == 1:  # 眠二
                    total_score += self.pattern_scores["two_semi"]

        # 同时评估对方（人类）在该位置的潜在威胁
        opponent = 3 - player
        temp_board[row][col] = opponent
        opponent_score = 0

        for dir_pair in directions:
            count = 1
            blocked_ends = 0

            for dr, dc in dir_pair:
                consecutive = self.count_consecutive(temp_board, row, col, opponent, dr, dc)
                count += consecutive

                r = row + dr * (consecutive + 1)
                c = col + dc * (consecutive + 1)
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if temp_board[r][c] == player:
                        blocked_ends += 1
                else:
                    blocked_ends += 1

            # 对方的威胁需要优先考虑（尤其是活四、冲四）
            if count >= 5:
                opponent_score += self.pattern_scores["win"] * 0.8  # 稍低于自己获胜的价值
            elif count == 4:
                if blocked_ends == 0:
                    opponent_score += self.pattern_scores["four_open"] * 0.9  # 几乎优先于自己的冲四
                elif blocked_ends == 1:
                    opponent_score += self.pattern_scores["four_semi"] * 0.8

        return total_score + opponent_score  # 综合考虑自己的机会和对方的威胁

    def stable_softmax(self, values):
        """数值稳定的softmax实现，防止溢出"""
        # 处理空值或单值情况
        if len(values) == 0:
            return np.array([])
        if len(values) == 1:
            return np.array([1.0])

        # 减去最大值以提高数值稳定性
        values = np.array(values, dtype=np.float64)
        max_val = np.max(values)
        exp_values = np.exp(values - max_val)  # 关键：减去最大值防止指数溢出

        # 处理可能的零和情况
        sum_exp = np.sum(exp_values)
        if sum_exp == 0:
            return np.ones_like(values) / len(values)  # 均匀分布

        return exp_values / sum_exp

    def choose_action(self, board):
        """选择落子位置，结合基础棋理和Q-learning"""
        current_state = board.copy()
        state_key = self.state_to_key(current_state)
        self.prev_state = state_key

        # 找到所有合法落子位置
        valid_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if current_state[i][j] == 0:
                    valid_actions.append((i, j))

        if not valid_actions:
            return None

        # 评估所有合法位置的基础价值
        action_values = {}
        for action in valid_actions:
            # 基础棋理评分 + Q学习评分
            base_score = self.evaluate_position(current_state, action[0], action[1], 2)
            q_score = self.get_q_value(state_key, action)
            action_values[action] = base_score + q_score * 10  # Q值权重调整

        # 检查是否有必须落子的位置（如阻止对方获胜）
        max_value = max(action_values.values()) if action_values else 0

        # 检查获胜落子
        win_threshold = self.pattern_scores["win"] * 0.9  # 设置一个略低于满分的阈值
        winning_moves = [a for a, v in action_values.items() if v >= win_threshold]
        if winning_moves:
            self.prev_action = random.choice(winning_moves)
            return self.prev_action

        # 检查需要阻止对方获胜的位置
        block_threshold = self.pattern_scores["win"] * 0.7  # 阻挡阈值
        blocking_moves = [a for a, v in action_values.items() if v >= block_threshold and v < win_threshold]
        if blocking_moves:
            self.prev_action = random.choice(blocking_moves)
            return self.prev_action

        # epsilon-贪婪策略：高价值动作有更高概率被选中
        if random.random() < self.epsilon:
            # 使用数值稳定的softmax计算概率
            values = list(action_values.values())
            probs = self.stable_softmax(values)

            # 确保概率有效
            if np.isnan(probs).any() or np.sum(probs) <= 0:
                # 如果概率计算失败，使用均匀分布
                probs = np.ones_like(probs) / len(probs)

            # 选择动作
            action = random.choices(valid_actions, weights=probs, k=1)[0]
            self.prev_action = action
            return action
        else:
            # 选择价值最高的动作
            best_actions = [a for a, v in action_values.items() if v == max_value]
            action = random.choice(best_actions)
            self.prev_action = action
            return action

    def learn(self, new_board, reward):
        """从结果中学习，更新Q表"""
        if self.prev_state is None or self.prev_action is None:
            return

        # 新状态
        new_state = new_board.copy()
        new_state_key = self.state_to_key(new_state)

        # 计算新状态的最大Q值
        valid_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if new_state[i][j] == 0:
                    valid_actions.append((i, j))

        if valid_actions:
            max_future_q = max(self.get_q_value(new_state_key, a) for a in valid_actions)
        else:
            max_future_q = 0

        # 更新Q值：结合基础棋理评分
        old_q = self.get_q_value(self.prev_state, self.prev_action)
        # 计算该动作的基础评分作为参考
        row, col = self.prev_action
        base_score = self.evaluate_position(np.array(self.prev_state), row, col, 2) * 0.01
        # 更新公式加入基础评分作为正则项
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q)
        new_q = new_q * 0.9 + base_score * 0.1  # 保留部分基础评分特性
        self.q_table[self.prev_state][self.prev_action] = new_q

    def decay_epsilon(self, rate=0.995):
        """降低探索率"""
        self.epsilon = max(0.05, self.epsilon * rate)

    def save_model(self, filename="gomoku_ai_model.pkl"):
        """保存模型"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'board_size': self.board_size
            }, f)

    def load_model(self, filename="gomoku_ai_model.pkl"):
        """加载模型"""
        import pickle
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data.get('epsilon', 0.3)
                self.board_size = data.get('board_size', 15)
            return True
        except:
            return False


class EvaluationThread(QThread):
    update_signal = Signal(str)
    result_signal = Signal(dict)

    def __init__(self, ai, parent=None):
        super().__init__(parent)
        self.ai = ai

    def run(self):
        self.update_signal.emit("准备进行AI性能评估...\n")
        # 保存当前AI的探索率并临时禁用探索
        original_epsilon = self.ai.epsilon
        self.ai.epsilon = 0.0  # 评估时不进行探索

        # 执行评估
        results = self.evaluate_ai_performance(self.ai)

        # 恢复原始探索率
        self.ai.epsilon = original_epsilon

        self.result_signal.emit(results)

    def evaluate_ai_performance(self, ai, num_games=50, board_size=15):
        """评估AI性能，让AI与随机落子的对手进行多轮对战"""
        results = {
            "total_games": num_games,
            "ai_wins": 0,
            "random_wins": 0,
            "draws": 0,
            "avg_steps": 0,
            "ai_win_steps": [],
            "random_win_steps": [],
            "draw_steps": []
        }

        self.update_signal.emit(f"开始评估，共进行{num_games}场对战...\n")

        for game in range(num_games):
            # 每局交替先手，确保公平性
            ai_first = game % 2 == 0

            # 初始化棋盘
            board = np.zeros((board_size, board_size), dtype=int)
            game_over = False
            winner = 0
            steps = 0

            while not game_over:
                current_player = 1 if (steps % 2 == 0) == ai_first else 2

                # 找出所有合法落子位置
                valid_moves = []
                for i in range(board_size):
                    for j in range(board_size):
                        if board[i][j] == 0:
                            valid_moves.append((i, j))

                if not valid_moves:  # 平局
                    game_over = True
                    winner = 0
                    break

                if current_player == 1:  # AI落子
                    move = self.ai.choose_action(board)
                    if not move:
                        move = valid_moves[0]
                    row, col = move
                    board[row][col] = 1

                    # 检查AI是否获胜
                    if self.check_win(board, row, col, 1, board_size):
                        game_over = True
                        winner = 1
                        break
                else:  # 随机对手落子
                    move_idx = random.choice(range(len(valid_moves)))
                    row, col = valid_moves[move_idx]
                    board[row][col] = 2

                    # 检查随机对手是否获胜
                    if self.check_win(board, row, col, 2, board_size):
                        game_over = True
                        winner = 2
                        break

                steps += 1

            # 更新结果统计
            results["avg_steps"] += steps

            if winner == 1:
                results["ai_wins"] += 1
                results["ai_win_steps"].append(steps)
            elif winner == 2:
                results["random_wins"] += 1
                results["random_win_steps"].append(steps)
            else:
                results["draws"] += 1
                results["draw_steps"].append(steps)

            # 每完成10%进度更新一次
            if (game + 1) % (num_games // 10) == 0:
                progress = (game + 1) / num_games * 100
                self.update_signal.emit(f"评估进度: {progress:.0f}% ({game + 1}/{num_games}场)\n")

        # 计算平均步数
        results["avg_steps"] /= num_games

        # 计算各类结果的平均步数
        results["avg_ai_win_steps"] = np.mean(results["ai_win_steps"]) if results["ai_win_steps"] else 0
        results["avg_random_win_steps"] = np.mean(results["random_win_steps"]) if results["random_win_steps"] else 0
        results["avg_draw_steps"] = np.mean(results["draw_steps"]) if results["draw_steps"] else 0

        # 计算胜率
        results["ai_win_rate"] = results["ai_wins"] / num_games
        results["random_win_rate"] = results["random_wins"] / num_games
        results["draw_rate"] = results["draws"] / num_games

        return results

    def check_win(self, board, row, col, player, board_size):
        """检查是否获胜"""
        directions = [
            [(0, 1), (0, -1)],  # 水平
            [(1, 0), (-1, 0)],  # 垂直
            [(1, 1), (-1, -1)],  # 对角线
            [(1, -1), (-1, 1)]  # 反对角线
        ]

        for dir_pair in directions:
            count = 1  # 已落子的位置

            # 检查两个方向
            for dr, dc in dir_pair:
                r, c = row + dr, col + dc
                while 0 <= r < board_size and 0 <= c < board_size and board[r][c] == player:
                    count += 1
                    r += dr
                    c += dc

            if count >= 5:
                return True

        return False


class GomokuGame(QMainWindow):
    """五子棋游戏主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("自学习五子棋（带基础棋理）")
        self.setGeometry(100, 100, 600, 700)

        # 初始化AI（带基础棋理）
        self.ai = ImprovedQLearningAI()
        self.ai.load_model()  # 尝试加载已保存的模型

        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 标题
        title_label = QLabel("自学习五子棋（带基础棋理）")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # 棋盘
        self.board = GomokuBoard()
        self.board.move_made.connect(self.on_move_made)
        self.board.game_over.connect(self.on_game_over)
        main_layout.addWidget(self.board)

        # 评估结果显示
        self.evaluation_result = QTextEdit()
        self.evaluation_result.setReadOnly(True)
        self.evaluation_result.setMinimumHeight(150)
        main_layout.addWidget(self.evaluation_result)

        # 状态信息
        self.status_label = QLabel("游戏开始，你执黑棋，请落子")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # 按钮布局
        button_layout = QHBoxLayout()

        # 重新开始按钮
        self.restart_button = QPushButton("重新开始")
        self.restart_button.clicked.connect(self.restart_game)
        button_layout.addWidget(self.restart_button)

        # 保存模型按钮
        self.save_button = QPushButton("保存AI模型")
        self.save_button.clicked.connect(self.save_ai_model)
        button_layout.addWidget(self.save_button)

        # 评估模型按钮
        self.evaluate_button = QPushButton("评估AI模型")
        self.evaluate_button.clicked.connect(self.evaluate_ai)
        button_layout.addWidget(self.evaluate_button)

        # AI学习进度
        self.ai_progress = QLabel(f"AI学习进度: 已记录 {len(self.ai.q_table)} 种局面")
        button_layout.addWidget(self.ai_progress)

        main_layout.addLayout(button_layout)

        # AI思考计时器
        self.ai_timer = QTimer(self)
        self.ai_timer.setInterval(800)  # AI延迟0.8秒落子
        self.ai_timer.timeout.connect(self.ai_make_move)
        self.ai_thinking = False  # 标记AI是否正在思考

    def evaluate_ai(self):
        """评估AI模型性能"""
        self.evaluation_result.clear()
        self.evaluation_result.append("开始评估AI性能，请稍候...\n")
        self.evaluation_result.repaint()  # 立即刷新显示

        # 禁用按钮防止重复评估
        self.evaluate_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        self.save_button.setEnabled(False)

        # 创建并启动评估线程
        self.eval_thread = EvaluationThread(self.ai)
        self.eval_thread.update_signal.connect(self.append_evaluation_text)
        self.eval_thread.result_signal.connect(self.show_evaluation_results)
        self.eval_thread.finished.connect(self.evaluation_finished)
        self.eval_thread.start()

    def append_evaluation_text(self, text):
        """追加评估过程文本"""
        self.evaluation_result.append(text)
        # 滚动到底部 - 修复了此处的错误
        self.evaluation_result.moveCursor(QTextCursor.End)

    def show_evaluation_results(self, results):
        """显示评估结果"""
        self.evaluation_result.append("\n===== AI性能评估结果 =====")
        self.evaluation_result.append(f"总对战次数: {results['total_games']}")
        self.evaluation_result.append(f"AI获胜次数: {results['ai_wins']} ({results['ai_win_rate']:.2%})")
        self.evaluation_result.append(f"随机对手获胜次数: {results['random_wins']} ({results['random_win_rate']:.2%})")
        self.evaluation_result.append(f"平局次数: {results['draws']} ({results['draw_rate']:.2%})")
        self.evaluation_result.append(f"平均步数: {results['avg_steps']:.1f}")
        self.evaluation_result.append(f"AI获胜平均步数: {results['avg_ai_win_steps']:.1f}")
        self.evaluation_result.append(f"随机对手获胜平均步数: {results['avg_random_win_steps']:.1f}")
        self.evaluation_result.append(f"平局平均步数: {results['avg_draw_steps']:.1f}")
        self.evaluation_result.append("===========================")
        # 滚动到底部 - 修复了此处的错误
        self.evaluation_result.moveCursor(QTextCursor.End)

    def evaluation_finished(self):
        """评估完成后启用按钮"""
        self.evaluate_button.setEnabled(True)
        self.restart_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.evaluation_result.append("\n评估完成！")
        # 滚动到底部 - 修复了此处的错误
        self.evaluation_result.moveCursor(QTextCursor.End)

    @Slot(int, int, int)
    def on_move_made(self, row, col, player):
        """处理落子事件"""
        if player == 1 and not self.board.game_over_flag:  # 人类落子后，让AI学习并准备落子
            self.status_label.setText("AI思考中...")
            self.ai_thinking = True
            # 给AI一个小的负奖励
            self.ai.learn(self.board.board, -0.1)
            self.ai_timer.start()
            self.update_ai_progress()
        else:  # AI落子后
            self.ai_thinking = False
            self.status_label.setText("轮到你落子")

    def ai_make_move(self):
        """AI落子"""
        # 确保只在AI回合且游戏未结束时执行
        if not self.ai_thinking or self.board.current_player != 2 or self.board.game_over_flag:
            self.ai_timer.stop()
            self.ai_thinking = False
            return

        self.ai_timer.stop()
        move = self.ai.choose_action(self.board.board)
        if move:
            self.board.make_ai_move(move[0], move[1])
        self.ai_thinking = False

    @Slot(int)
    def on_game_over(self, winner):
        """处理游戏结束事件"""
        # 停止AI计时器，避免游戏结束后继续思考
        self.ai_timer.stop()
        self.ai_thinking = False

        if winner == 1:
            QMessageBox.information(self, "游戏结束", "恭喜你获胜！")
            # 人类获胜，给AI一个大的负奖励
            self.ai.learn(self.board.board, -10)
        elif winner == 2:
            QMessageBox.information(self, "游戏结束", "AI获胜！")
            # AI获胜，给一个大的正奖励
            self.ai.learn(self.board.board, 10)
        else:
            QMessageBox.information(self, "游戏结束", "平局！")
            # 平局，给一个小的奖励
            self.ai.learn(self.board.board, 1)

        # 降低探索率
        self.ai.decay_epsilon()
        self.update_ai_progress()
        self.status_label.setText("游戏结束，点击重新开始")

    def restart_game(self):
        """重新开始游戏"""
        self.ai_timer.stop()
        self.ai_thinking = False
        self.board.reset_board()
        self.status_label.setText("游戏开始，你执黑棋，请落子")

    def save_ai_model(self):
        """保存AI模型"""
        self.ai.save_model()
        QMessageBox.information(self, "保存成功", f"AI模型已保存，当前已学习 {len(self.ai.q_table)} 种局面")

    def update_ai_progress(self):
        """更新AI学习进度显示"""
        self.ai_progress.setText(f"AI学习进度: 已记录 {len(self.ai.q_table)} 种局面, 探索率: {self.ai.epsilon:.3f}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    game = GomokuGame()
    game.show()
    sys.exit(app.exec())
