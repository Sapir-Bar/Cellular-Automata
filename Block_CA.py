import numpy as np
import matplotlib.pyplot as plt
import pygame
import pygame_gui
import time
import os

# ========== Cell and Board Classes ==========
class Cell:
    def __init__(self, p):
        # Initializes the cell state randomly: 1 (alive) with probability p
        self.state = np.random.choice([0, 1], p=[1 - p, p])

    def flip(self):
        # Inverts the cell's state (0 -> 1, 1 -> 0)
        self.state ^= 1


class Board:
    def __init__(self, N, p, wraparound=True, glider=None, init_mode="Random"):
        self.N = N
        self.wraparound = wraparound
        self.blocks_even = self._create_blocks(start=1)
        self.blocks_odd = self._create_blocks(start=0) 
        self.grid = np.array([[Cell(p) for _ in range(N)] for _ in range(N)])

        # Choose initialization mode
        if glider is not None:
            self._place_pattern(glider)
            print("Glider placed")
            print(f"{p}")
        elif init_mode == "Odd Columns Alive":
            self._initialize_odd_columns()
        elif init_mode == "Odd Diagonals Alive":
            self._initialize_diagonal_pattern()

    def _initialize_odd_columns(self):
        # Activates all cells in odd-indexed columns
        for i in range(self.N):
            for j in range(self.N):
                if j % 2 == 1:
                    self.grid[i][j].state = 1
                else:
                    self.grid[i][j].state = 0

    def _initialize_diagonal_pattern(self):
        # Activates cells where (i + j) is odd — creates a checkerboard-like pattern
        for i in range(self.N):
            for j in range(self.N):
                if (i + j) % 2 == 1:
                    self.grid[i][j].state = 1
                else:
                    self.grid[i][j].state = 0

    def _place_pattern(self, pattern):
        # Places a glider at a random odd-aligned position
        height = len(pattern)
        width = len(pattern[0])
        max_top = self.N - height
        max_left = self.N - width
        top = np.random.randint(0, (max_top + 1) // 2) * 2 + 1
        left = np.random.randint(0, (max_left + 1) // 2) * 2 + 1
        for i, row in enumerate(pattern):
            for j, val in enumerate(row):
                if 0 <= top + i < self.N and 0 <= left + j < self.N:
                    self.grid[top + i][left + j].state = val

    def _create_blocks(self, start):
        # Returns the top-left coordinates of all 2x2 blocks for even/odd generations
        coords = []
        for i in range(start, self.N, 2):
            for j in range(start, self.N, 2):
                if self.wraparound or (i + 1 < self.N and j + 1 < self.N):
                    coords.append((i, j))
        return coords

    def apply_block_rules(self, i, j):
        # Applies rules to a single 2x2 block at position (i, j)
        N = self.N
        if not self.wraparound and (i + 1 >= N or j + 1 >= N):
            return

        indices = [
            (i, j), (i, (j + 1) % N),
            ((i + 1) % N, j), ((i + 1) % N, (j + 1) % N)
        ]
        cells = [self.grid[x][y] for x, y in indices]
        density = sum(cell.state for cell in cells)

        if density == 2:
            return # Stable block — no changes
        elif density in [0, 1, 4]:
            for cell in cells:
                cell.flip() # Full flip
        elif density == 3: 
            # Flip and rotate 180 degrees
            self.grid[i][j], self.grid[(i + 1) % N][(j + 1) % N] = self.grid[(i + 1) % N][(j + 1) % N], self.grid[i][j]
            self.grid[i][(j + 1) % N], self.grid[(i + 1) % N][j] = self.grid[(i + 1) % N][j], self.grid[i][(j + 1) % N]
            for cell in cells:
                cell.flip()

    def step(self, even):
        # Applies rules for the current generation
        blocks = self.blocks_even if even else self.blocks_odd
        for i, j in blocks:
            self.apply_block_rules(i, j)

    def get_state_array(self):
        # Returns the board as a 2D numpy array of states
        return np.array([[cell.state for cell in row] for row in self.grid])


# ========== Game Management ==========
class GameManager:
    def __init__(self, N=100, p=0.5, max_steps=250, wraparound=True, glider=None, init_mode="Random"):
        self.N = N
        self.p = p
        self.max_steps = max_steps
        self.wraparound = wraparound
        self.board = Board(N, p, wraparound=wraparound, glider=glider, init_mode=init_mode)
        self.current_step = 0

    def step(self):
        # Advances the game one generation
        even = (self.current_step % 2 == 0)
        self.board.step(even)
        self.current_step += 1

    def draw(self, surface, offset_x):
        # Draws the grid onto the GUI surface
        board_area_size = 560
        cell_size = int(board_area_size // self.N)
        for i in range(self.N):
            for j in range(self.N):
                color = (0, 0, 0) if self.board.grid[i][j].state == 1 else (255, 255, 255)
                x = int(offset_x + j * cell_size)
                y = int(i * cell_size)
                rect = pygame.Rect(x, y, cell_size, cell_size)
                pygame.draw.rect(surface, color, rect)


# ========== Simulator GUI Controller ==========
class Simulator:
    def __init__(self):
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        pygame.init()

        # Create a window and a GUI manager
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Cellular Automata Simulation")
        self.control_rect = pygame.Rect(0, 0, 240, 600)
        self.manager = pygame_gui.UIManager((800, 600), "theme.json")
        self.controls = self._build_gui()
        self.clock = pygame.time.Clock()

        self.game_manager = None
        self.wraparound = True
        self.init_mode = "Random"
        self.glider = [[0,1], [1,0], [1,0], [0,1]] 

    def _build_gui(self):
        # Creates all GUI elements: sliders, buttons, dropdowns
        CONTROL_X = 10
        CONTROL_WIDTH = 220
        ROW_HEIGHT = 45
        row_y = 10

        # Creates a slider with a label above it
        def slider_label_pair(text, value, min_val, max_val):
            nonlocal row_y
            label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((CONTROL_X, row_y), (CONTROL_WIDTH, 20)),
                text=f"{text}: {value}",
                manager=self.manager,
                object_id="#label"
                )
            row_y += 20
            slider = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect((CONTROL_X, row_y), (CONTROL_WIDTH, 25)),
                start_value=value, value_range=(min_val, max_val), manager=self.manager)
            row_y += ROW_HEIGHT
            return label, slider
        
        # Initialize controls
        gen_label, gen_slider = slider_label_pair("Generations", 250, 1, 1000)
        prob_label, prob_slider = slider_label_pair("Initial State Probability", 50, 0, 100)
        size_label, size_slider = slider_label_pair("Board Size", 100, 10, 200)

        wrap_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((CONTROL_X, row_y), (CONTROL_WIDTH, 30)),
            text="Wraparound: Yes", manager=self.manager)
        row_y += ROW_HEIGHT

        play_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((CONTROL_X, row_y), (CONTROL_WIDTH, 30)),
            text='Play', manager=self.manager)
        row_y += ROW_HEIGHT

        reset_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((CONTROL_X, row_y), (CONTROL_WIDTH, 30)),
            text='Reset', manager=self.manager)
        row_y += ROW_HEIGHT

        mode_label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((CONTROL_X, row_y), (CONTROL_WIDTH, 20)),
                text=f"Choose Mode:",
                manager=self.manager,
                object_id="#label")
        row_y += 20

        # Dropdown for selecting special initialization modes
        mode_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=["Random", "Odd Columns Alive", "Odd Diagonals Alive", "Glider"],
            starting_option="Random",
            relative_rect=pygame.Rect((CONTROL_X, row_y), (CONTROL_WIDTH, 30)),
            manager=self.manager)
        row_y += ROW_HEIGHT

        gen_counter = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 520), (200, 30)),
            text="Generation: 0", manager=self.manager, visible=0)

        return {
            "gen_label": gen_label, "gen_slider": gen_slider,
            "prob_label": prob_label, "prob_slider": prob_slider,
            "size_label": size_label, "size_slider": size_slider,
            "wrap_toggle_button": wrap_btn,
            "init_mode_dropdown": mode_dropdown, "mode_label": mode_label,
            "play_pause_button": play_btn, "reset_button": reset_btn,
            "generation_counter_label": gen_counter
        }

    def launch(self):
        # Main GUI event loop
        while True:
            time_delta = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.controls["play_pause_button"]:
                        # Start simulation
                        self.start_simulation()
                    elif event.ui_element == self.controls["wrap_toggle_button"]:
                        # Toggle wraparound logic
                        self.wraparound = not self.wraparound
                        self.controls["wrap_toggle_button"].set_text(f"Wraparound: {'Yes' if self.wraparound else 'No'}")
                self.manager.process_events(event)

            self.manager.update(time_delta)
            self.update_labels()
            self.screen.fill((200, 200, 200))
            pygame.draw.rect(self.screen, (150, 150, 150), self.control_rect)
            self.manager.draw_ui(self.screen)
            pygame.display.update()

    def update_labels(self):
        # Updates GUI text for sliders
        self.controls["gen_label"].set_text(f"Generations: {int(self.controls['gen_slider'].get_current_value())}")
        self.controls["prob_label"].set_text(f"Initial State Probability: {int(self.controls['prob_slider'].get_current_value()) / 100}")
        sz = int(self.controls["size_slider"].get_current_value())
        self.controls["size_label"].set_text(f"Board Size: {sz}×{sz}")

    def start_simulation(self):
        generations = int(self.controls["gen_slider"].get_current_value())
        probability = int(self.controls["prob_slider"].get_current_value()) / 100
        size = int(self.controls["size_slider"].get_current_value())
        size = size if size % 2 == 0 else size + 1

        self.init_mode = self.controls["init_mode_dropdown"].selected_option
        if isinstance(self.init_mode, tuple):
            self.init_mode = self.init_mode[0]  
        print(f"Selected mode: {self.init_mode}")

        if self.init_mode == "Glider":
            probability = 1.0

        self.game_manager = GameManager(N=size, p=probability, max_steps=generations, wraparound=self.wraparound,
                                        glider=self.glider if self.init_mode == "Glider" else None,
                                        init_mode=self.init_mode)
        
        self.controls["generation_counter_label"].show()
        self.controls["play_pause_button"].set_text('Pause')
        for key, widget in self.controls.items():
            if key not in ["play_pause_button", "reset_button"] and hasattr(widget, "disable"):
                widget.disable()

        self.run_simulation()

    def run_simulation(self):
        paused = False
        running = True

        while running:
            time_delta = self.clock.tick(5) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.controls["play_pause_button"]:
                        paused = not paused
                        self.controls["play_pause_button"].set_text('Play' if paused else 'Pause')
                    elif event.ui_element == self.controls["reset_button"]:
                        paused = True
                        self.game_manager.current_step = 0
                        self.game_manager.board = Board(self.game_manager.N, self.game_manager.p, self.game_manager.wraparound,
                                                         glider=self.glider if self.init_mode == "Glider" else None,
                                                         init_mode=self.init_mode)
                        self.controls["play_pause_button"].set_text('Play')
                self.manager.process_events(event)

            if self.game_manager.current_step < self.game_manager.max_steps and not paused:
                self.game_manager.step()
                self.game_manager.draw(self.screen, self.control_rect.width)
            elif self.game_manager.current_step < self.game_manager.max_steps and paused:
                self.game_manager.draw(self.screen, self.control_rect.width)
            else:
                running = False

            pygame.draw.rect(self.screen, (150, 150, 150), self.control_rect)
            self.controls["generation_counter_label"].set_text(f"Generation: {self.game_manager.current_step}")
            self.manager.update(time_delta)
            self.manager.draw_ui(self.screen)
            pygame.display.update()


# ========== Entry Point ==========
if __name__ == "__main__":
    Simulator().launch()
