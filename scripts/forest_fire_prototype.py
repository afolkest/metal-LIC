"""
Drossel-Schwagl forest fire model — pygame + numba + sliders.

Controls:
  1/2/3  — resolution 500 / 1000 / 2000
  R      — reset grid
  SPACE  — pause/resume
  ESC    — quit
  Sliders for growth, lightning, steps/frame
"""

import numpy as np
import numba as nb
import pygame
import sys

# --- Simulation kernel (numba JIT) ---

@nb.njit(cache=True)
def step(g, out, p_val, f_val, rng_seed):
    h, w = g.shape
    rng = rng_seed
    for y in range(h):
        for x in range(w):
            rng ^= rng << np.uint64(13)
            rng ^= rng >> np.uint64(7)
            rng ^= rng << np.uint64(17)
            r = np.float64(rng & np.uint64(0xFFFFFFFF)) / 4294967295.0

            c = g[y, x]
            if c == 2:
                out[y, x] = 0
            elif c == 1:
                has_fire = (g[(y - 1) % h, x] == 2 or
                            g[(y + 1) % h, x] == 2 or
                            g[y, (x - 1) % w] == 2 or
                            g[y, (x + 1) % w] == 2)
                if has_fire:
                    out[y, x] = 2
                elif r < f_val:
                    out[y, x] = 2
                else:
                    out[y, x] = 1
            else:
                if r < p_val:
                    out[y, x] = 1
                else:
                    out[y, x] = 0
    return rng

@nb.njit(cache=True)
def grid_to_rgb(g, rgb):
    h, w = g.shape
    for y in range(h):
        for x in range(w):
            c = g[y, x]
            if c == 0:
                rgb[y, x, 0] = 26;  rgb[y, x, 1] = 26;  rgb[y, x, 2] = 26
            elif c == 1:
                rgb[y, x, 0] = 45;  rgb[y, x, 1] = 138; rgb[y, x, 2] = 78
            else:
                rgb[y, x, 0] = 255; rgb[y, x, 1] = 102; rgb[y, x, 2] = 0

# --- Simple slider class ---

class Slider:
    def __init__(self, x, y, w, label, val_min, val_max, val, fmt="{:.2f}"):
        self.rect = pygame.Rect(x, y, w, 20)
        self.label = label
        self.val_min = val_min
        self.val_max = val_max
        self.val = val
        self.fmt = fmt
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_val(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_val(event.pos[0])

    def _update_val(self, mx):
        t = (mx - self.rect.x) / self.rect.w
        t = max(0.0, min(1.0, t))
        self.val = self.val_min + t * (self.val_max - self.val_min)

    def draw(self, screen, font):
        # Track
        pygame.draw.rect(screen, (60, 60, 60), self.rect)
        # Fill
        t = (self.val - self.val_min) / (self.val_max - self.val_min)
        fill = pygame.Rect(self.rect.x, self.rect.y, int(self.rect.w * t), self.rect.h)
        pygame.draw.rect(screen, (100, 140, 180), fill)
        # Handle
        hx = self.rect.x + int(self.rect.w * t)
        pygame.draw.rect(screen, (220, 220, 220), (hx - 3, self.rect.y - 2, 6, self.rect.h + 4))
        # Label
        txt = font.render(f"{self.label}: {self.fmt.format(self.val)}", True, (220, 220, 220))
        screen.blit(txt, (self.rect.x, self.rect.y - 18))

# --- Init ---

W, H = 512, 512
rng_state = np.uint64(42)

def make_grid(w):
    global rng_state
    g = np.zeros((w, w), dtype=np.int8)
    rng_state = np.uint64(42)
    for y in range(w):
        for x in range(w):
            rng_state ^= rng_state << np.uint64(13)
            rng_state ^= rng_state >> np.uint64(7)
            rng_state ^= rng_state << np.uint64(17)
            if (rng_state & np.uint64(0xFFFFFFFF)) / 4294967295.0 < 0.6:
                g[y, x] = 1
    return g

grid = make_grid(W)
out = np.zeros_like(grid)
rgb = np.zeros((H, W, 3), dtype=np.uint8)

# JIT warmup
print("Compiling JIT kernels...")
rng_state = step(grid, out, 0.02, 5e-7, np.uint64(123))
grid, out = out, grid
grid_to_rgb(grid, rgb)
print("Ready.")

# --- Pygame setup ---

pygame.init()
PANEL_W = 260
screen = pygame.display.set_mode((800 + PANEL_W, 800), pygame.RESIZABLE)
pygame.display.set_caption("Forest Fire CA")
font = pygame.font.SysFont("menlo", 13)
clock = pygame.time.Clock()

# Sliders in right panel
sl_growth = Slider(820, 60, 220, "growth (log10)", -4.0, -0.5, np.log10(0.02), fmt="{:.1f}")
sl_lightning = Slider(820, 120, 220, "lightning (log10)", -8.0, -3.0, np.log10(5e-7), fmt="{:.1f}")
sl_substeps = Slider(820, 180, 220, "steps/frame", 1, 20, 2, fmt="{:.0f}")
sliders = [sl_growth, sl_lightning, sl_substeps]

paused = False
frame = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        for sl in sliders:
            sl.handle_event(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_1:
                W, H = 512, 512
                grid = make_grid(W); out = np.zeros_like(grid)
                rgb = np.zeros((H, W, 3), dtype=np.uint8); frame = 0
            elif event.key == pygame.K_2:
                W, H = 1024, 1024
                grid = make_grid(W); out = np.zeros_like(grid)
                rgb = np.zeros((H, W, 3), dtype=np.uint8); frame = 0
            elif event.key == pygame.K_3:
                W, H = 2048, 2048
                grid = make_grid(W); out = np.zeros_like(grid)
                rgb = np.zeros((H, W, 3), dtype=np.uint8); frame = 0
            elif event.key == pygame.K_r:
                grid = make_grid(W); out = np.zeros_like(grid); frame = 0
            elif event.key == pygame.K_SPACE:
                paused = not paused

    # Simulate
    if not paused:
        p_val = 10 ** sl_growth.val
        f_val = 10 ** sl_lightning.val
        substeps = max(1, int(sl_substeps.val))
        for _ in range(substeps):
            rng_state = step(grid, out, p_val, f_val, rng_state)
            grid, out = out, grid
            frame += 1

    # Render
    grid_to_rgb(grid, rgb)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    sw, sh = screen.get_size()
    img_size = min(sw - PANEL_W, sh)
    scaled = pygame.transform.scale(surf, (img_size, img_size))

    screen.fill((30, 30, 30))
    screen.blit(scaled, (0, 0))

    # Reposition sliders to right panel (adapts to window resize)
    px = img_size + 20
    sl_w = max(140, sw - img_size - 40)
    for j, sl in enumerate(sliders):
        sl.rect.x = px
        sl.rect.y = 60 + j * 60
        sl.rect.w = sl_w
        sl.draw(screen, font)

    # Stats
    n_tree = int(np.sum(grid == 1))
    n_burn = int(np.sum(grid == 2))
    n_total = W * H
    f_val = 10 ** sl_lightning.val
    lines = [
        f"{clock.get_fps():.0f} fps    step {frame}",
        f"",
        f"trees: {n_tree:,} ({100*n_tree/n_total:.0f}%)",
        f"fire:  {n_burn:,}",
        f"f*N:   {f_val * n_tree:.2f} strikes/step",
        f"",
        f"res: {W}  [1/2/3]",
        f"[SPACE] pause  [R] reset",
    ]
    y = 60 + len(sliders) * 60 + 20
    for line in lines:
        txt = font.render(line, True, (180, 180, 180))
        screen.blit(txt, (px, y))
        y += 18

    if paused:
        txt = font.render("PAUSED", True, (255, 200, 100))
        screen.blit(txt, (px, y + 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
