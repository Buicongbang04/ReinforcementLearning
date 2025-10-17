import tkinter as tk
import random, pickle, os

MODE = "train"
BOARD_SIZE, WIN_LENGTH = 12, 5
CELL_SIZE, MARGIN = 40, 16
SAVE_PATH = "policy_value.pkl"

class Game:
    def __init__(self): self.reset()
    def reset(self):
        self.board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.turn = 1
        self.winner = 0
        self.history = []

    def place(self,row,col):
        if self.winner or self.board[row][col]: 
            return False
        self.board[row][col] = self.turn
        self.history.append((row,col))

        if check_win(self.board,row,col,self.turn): 
            self.winner = self.turn
        elif all(self.board[r][c] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)): 
            self.winner = 3

        self.turn = 3-self.turn
        return True

def check_win(board,row,col,player):
    for dr,dc in [(1,0),(0,1),(1,1),(1,-1)]:
        count = 1
        r, c = row + dr,col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player: 
            count += 1
            r += dr
            c += dc

        r,c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player: 
            count += 1
            r -= dr
            c -= dc

        if count >= WIN_LENGTH: 
            return True
        
    return False

random.seed(0)
ZOBRIST=[[[random.getrandbits(64) for _ in range(3)] for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def rotate_board(board):
    n = len(board)
    return [[board[n-1-c][r] for c in range(n)] for r in range(n)]

def reflect_board(board):
    return [list(reversed(row)) for row in board]

def hash_board(board):
    h = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v=board[r][c]
            if v: 
                h ^= ZOBRIST[r][c][v]

    return h

def canonical_hash(board):
    variants = []
    cur = board

    for _ in range(4): 
        variants += [cur,reflect_board(cur)]
        cur = rotate_board(cur)

    return min(hash_board(b) for b in variants)

def active_bounds(board):
    rows = [r for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c]]
    cols = [c for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c]]

    if not rows: 
        return 0,BOARD_SIZE-1,0,BOARD_SIZE-1
    
    return max(0,min(rows)-2), min(BOARD_SIZE-1,max(rows)+2), max(0,min(cols)-2), min(BOARD_SIZE-1,max(cols)+2)

def candidate_moves(board):
    moves = set(); r0,r1,c0,c1=active_bounds(board)
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    if not any(board[r][c] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)):
        m = BOARD_SIZE//2
        return [(m,m)]
    
    for r in range(r0,r1+1):
        for c in range(c0,c1+1):
            if board[r][c]:
                for dr,dc in directions:
                    for d in (1,2):
                        nr,nc = r+dr*d,c+dc*d
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == 0: 
                            moves.add((nr,nc))
    if not moves:
        for r in range(r0,r1+1):
            for c in range(c0,c1+1):
                if board[r][c] == 0: 
                    moves.add((r,c))
    center = (BOARD_SIZE-1)/2
    return sorted(moves,key=lambda x:abs(x[0]-center)+abs(x[1]-center))

PATTERNS=[(5,1e0),(4,1e-1),(3,5e-3),(2,1e-3)]

def segment_score(board,row,col,dr,dc,player):
    er = row+(WIN_LENGTH-1)*dr; ec=col+(WIN_LENGTH-1)*dc
    if not(0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE): 
        return 0
    
    count = 0
    for k in range(WIN_LENGTH):
        v = board[row+k*dr][col+k*dc]
        
        if v == player: 
            count+=1
        elif v != 0: 
            return 0
   
    for L,val in PATTERNS:
        if count >= L:
            return val
   
    return 0

def heuristic(board,player):
    s = 0
    for dr,dc in [(1,0),(0,1),(1,1),(1,-1)]:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                s += segment_score(board,r,c,dr,dc,player) - segment_score(board,r,c,dr,dc,3-player)

    return 1e-3*s

class DPAgent:
    def __init__(self, epsilon=0.3, learning_rate=0.2):
        self.epsilon, self.learning_rate = epsilon, learning_rate
        self.value_table = {}
        self.episode_count=0
        self.load()

    def load(self):
        if os.path.exists(SAVE_PATH): 
            print("Bắt đầu tải")
            self.value_table = pickle.load(open(SAVE_PATH,'rb'))

    def save(self): 
        print("Bắt đầu lưu")
        pickle.dump(self.value_table, open(SAVE_PATH,'wb'))

    def select_move(self, game, allow_explore):
        print("Bắt đầu lượt AI")

        if allow_explore and random.random()<self.epsilon:
            candidates=candidate_moves(game.board)
            return random.choice(candidates) if candidates else None
        
        best_move=None; best_value=-1e18; player=game.turn
        for r,c in candidate_moves(game.board):
            game.board[r][c] = player
            if check_win(game.board,r,c,player): value=1.0
            else:
                bh = canonical_hash(game.board)
                value = self.value_table.get((bh,3-player),0.0) + heuristic(game.board,3-player)

            if value > best_value: 
                best_value, best_move = value,(r,c)

            game.board[r][c]=0
        return best_move
    def update_td0(self, prev_hash, prev_turn, terminal, curr_hash, curr_turn):
        if terminal == 1: 
            target=1.0
        elif terminal == 2: 
            target=-1.0
        elif terminal == 3:
            target=-0.1
        else: 
            target = self.value_table.get((curr_hash,curr_turn),0.0)

        reward_for_prev = target if prev_turn == 1 else -target
        old = self.value_table.get((prev_hash,prev_turn),0.0)
        self.value_table[(prev_hash,prev_turn)] = old + self.learning_rate*(reward_for_prev-old)

class UI:
    def __init__(self):
        print("Bắt đầu khởi tạo UI")
        self.game, self.agent = Game(), DPAgent()
        self.root = tk.Tk(); self.root.title(f"12x12 DP ({MODE})")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        w,h = MARGIN * 2 + CELL_SIZE * BOARD_SIZE, MARGIN * 2 + CELL_SIZE * BOARD_SIZE
        self.canvas = tk.Canvas(self.root,width=w,height=h,bg="#fff")
        self.canvas.grid(row=0,column=0,columnspan=3)
        self.canvas.bind("<Button-1>",self.on_click)
        tk.Button(self.root,text="Ván mới",command=self.new_game).grid(row=1,column=0,sticky="ew")
        tk.Button(self.root,text="Dừng",command=self.stop).grid(row=1,column=1,sticky="ew")
        tk.Label(self.root,text=f"Chế độ: {MODE}").grid(row=1,column=2,sticky="ew")
        self.running=True; self.prev_state=None; self.draw()

        if MODE=="train": 
            print("Bắt đầu TRAIN")
            self.loop_train()
        else: 
            print("Bắt đầu TEST")

    def new_game(self):
         print("Bắt đầu ván mới")
         self.game.reset()
         self.prev_state=None
         self.draw()

    def stop(self): 
        print("Bắt đầu dừng")
        self.running=False

    def after(self,ms,fn): 
        self.root.after(ms,fn)

    def loop_train(self):
        if not self.running or MODE!="train": 
            return
        
        if self.prev_state is None: 
            self.prev_state = (canonical_hash(self.game.board), self.game.turn)

        move = self.agent.select_move(self.game,allow_explore=True)
        if move: 
            self.game.place(*move)
            self.draw()
        terminal = self.game.winner
        curr_hash,curr_turn = (canonical_hash(self.game.board), self.game.turn)
        self.agent.update_td0(self.prev_state[0],self.prev_state[1],terminal,curr_hash,curr_turn)
        if terminal:
            self.agent.save()
            self.agent.episode_count += 1
            if self.agent.episode_count % 100 == 0:
                self.agent.epsilon = max(0.05,self.agent.epsilon*0.97)
                self.agent.learning_rate = max(0.05,self.agent.learning_rate*0.99)
            self.new_game()
        else:
            self.prev_state=(curr_hash,curr_turn)
        self.after(1,self.loop_train)

    def on_click(self,e):
        if MODE!="test" or not self.running or self.game.winner: 
            return
        row,col = (e.y-MARGIN)//CELL_SIZE,(e.x-MARGIN)//CELL_SIZE
        if 0 <= row < BOARD_SIZE and 0 <= col <BOARD_SIZE and self.game.board[row][col] == 0:
            print("Bắt đầu lượt người")
            self.game.place(row,col)
            self.draw()

            if self.game.winner: 
                self.end_game()
                return
            self.after(10,self.ai_test)

    def ai_test(self):
        if MODE!="test" or not self.running or self.game.winner: 
            return
        move = self.agent.select_move(self.game,allow_explore=False)
        if move: 
            self.game.place(*move)
            self.draw()
        if self.game.winner:
            self.end_game()

    def end_game(self): 
        self.agent.update_td0(canonical_hash([[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]),1,self.game.winner,0,1)
        self.agent.save()

    def on_close(self): 
        self.agent.save()
        self.root.destroy()
    
    def draw(self):
        self.canvas.delete("all")
        for i in range(BOARD_SIZE+1):
            self.canvas.create_line(MARGIN, MARGIN + i * CELL_SIZE, MARGIN + BOARD_SIZE * CELL_SIZE, MARGIN + i * CELL_SIZE)
            self.canvas.create_line(MARGIN + i * CELL_SIZE, MARGIN, MARGIN + i * CELL_SIZE, MARGIN + BOARD_SIZE * CELL_SIZE)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                v = self.game.board[r][c]
                if v == 1:
                    x1 = MARGIN + c * CELL_SIZE + 6
                    y1 = MARGIN + r * CELL_SIZE + 6
                    x2 = x1 + CELL_SIZE - 12
                    y2 = y1 + CELL_SIZE - 12
                    self.canvas.create_line(x1, y1, x2, y2, width=2)
                    self.canvas.create_line(x1, y2, x2, y1, width=2)
                elif v == 2:
                    x1 = MARGIN + c * CELL_SIZE + 6
                    y1 = MARGIN + r * CELL_SIZE + 6
                    x2 = x1 + CELL_SIZE - 12
                    y2 = y1 + CELL_SIZE - 12
                    self.canvas.create_oval(x1, y1, x2, y2, width=2)

        if self.game.winner:
            msg="X thắng" if self.game.winner==1 else ("O thắng" if self.game.winner==2 else "Hòa")
            self.canvas.create_text(MARGIN+CELL_SIZE*BOARD_SIZE//2,10,anchor="n",text=msg,font=("Arial",14,"bold"))

if __name__ == "__main__":
    ui=UI(); print("Bắt đầu chạy ứng dụng")
    ui.root.mainloop()
