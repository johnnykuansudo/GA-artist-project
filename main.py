"""
polygon-based GA for image approximation
Version: No CLI arguments needed. All settings configured inside the file.

run: main.py
"""

import os
import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List
from PIL import Image, ImageDraw


# ============================================================
#                     USER CONFIG AREA
# ============================================================

TARGET_IMAGE = "target3.jpg"        # 要用的圖片放這裡
OUTPUT_DIR = "output_auto"          # 輸出資料夾

# 若想改 resolution，就改 target_height。width 自動計算比例。
TARGET_HEIGHT = 256                
TARGET_WIDTH = -1                   # 自動算，不要改

N_POLYGONS = 120                    # 多邊形數量（越多越細緻，越慢）
POP_SIZE = 40                       # 每代族群大小
GENERATIONS = 3000                  # 總代數
MUTATION_RATE = 0.08                # 變異率
EVAL_SCALE = 0.5                    # fitness 計算用的縮小比例（0.3~0.7）
SAVE_EVERY = 200                    # 每幾代存一次圖片
RANDOM_SEED = 42                    # 固定隨機種子以利重現

# ============================================================
#                 INTERNAL GA CONFIG & STRUCTURES
# ============================================================

@dataclass
class GAConfig:
    width: int
    height: int
    n_polygons: int = N_POLYGONS
    population_size: int = POP_SIZE
    generations: int = GENERATIONS
    mutation_rate: float = MUTATION_RATE
    vertex_mutation_sigma: float = 0.05
    color_mutation_sigma: float = 20.0
    alpha_min: int = 30
    alpha_max: int = 200
    elitism: int = 2
    tournament_size: int = 3
    save_every: int = SAVE_EVERY
    eval_scale: float = EVAL_SCALE
    seed: int = RANDOM_SEED


@dataclass
class Polygon:
    vertices: np.ndarray
    color: np.ndarray

    @staticmethod
    def random_poly(rng: random.Random):
        verts = np.array(
            [[rng.random(), rng.random()],
             [rng.random(), rng.random()],
             [rng.random(), rng.random()]],
            dtype=np.float32
        )
        r = rng.randint(0, 255)
        g = rng.randint(0, 255)
        b = rng.randint(0, 255)
        a = rng.randint(40, 140)
        return Polygon(vertices=verts, color=np.array([r, g, b, a], dtype=np.int16))


@dataclass
class Individual:
    polygons: List[Polygon] = field(default_factory=list)
    fitness: float = None

    @staticmethod
    def random_ind(config: GAConfig, rng: random.Random):
        return Individual([Polygon.random_poly(rng) for _ in range(config.n_polygons)])

    def clone(self):
        return Individual(
            [Polygon(p.vertices.copy(), p.color.copy()) for p in self.polygons],
            self.fitness
        )


# ============================================================
#                      IMAGE OPERATIONS
# ============================================================

def load_image(path: str, width: int, height: int):
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def render(ind: Individual, width: int, height: int):
    base = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    for poly in ind.polygons:
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        pts = [(int(x * (width - 1)), int(y * (height - 1)))
               for x, y in poly.vertices]
        color = tuple(int(np.clip(c, 0, 255)) for c in poly.color)

        draw.polygon(pts, fill=color)
        base = Image.alpha_composite(base, overlay)

    return np.asarray(base.convert("RGB"), dtype=np.float32)


def mse(a: np.ndarray, b: np.ndarray):
    return float(np.mean((a - b) ** 2))


# ============================================================
#                     GENETIC OPERATORS
# ============================================================

def tournament_select(pop, config, rng):
    cs = rng.sample(pop, config.tournament_size)
    return min(cs, key=lambda i: i.fitness)


def crossover(p1: Individual, p2: Individual, config: GAConfig, rng: random.Random):
    cut = rng.randint(1, config.n_polygons - 1)
    c1 = [p.vertices.copy() for p in p1.polygons[:cut]] + \
         [p.vertices.copy() for p in p2.polygons[cut:]]

    c2 = [p.vertices.copy() for p in p2.polygons[:cut]] + \
         [p.vertices.copy() for p in p1.polygons[cut:]]

    child1 = Individual([Polygon(v.copy(), p.color.copy())
                         for v, p in zip([p.vertices for p in p1.polygons], p1.polygons)])
    child2 = Individual([Polygon(v.copy(), p.color.copy())
                         for v, p in zip([p.vertices for p in p2.polygons], p2.polygons)])

    child1 = p1.clone()
    child2 = p2.clone()

    return child1, child2


def mutate(ind: Individual, config: GAConfig, rng: random.Random, scale: float):
    for poly in ind.polygons:
        # Vertex mutation
        for v in range(3):
            if rng.random() < config.mutation_rate:
                poly.vertices[v] += rng.gauss(0, config.vertex_mutation_sigma * scale)
                poly.vertices[v] = np.clip(poly.vertices[v], 0, 1)

        # Color mutation
        for c in range(4):
            if rng.random() < config.mutation_rate:
                if c < 3:
                    poly.color[c] += rng.gauss(0, config.color_mutation_sigma * scale)
                else:
                    poly.color[c] += rng.gauss(0, 10 * scale)
                poly.color[c] = int(np.clip(poly.color[c],
                                            config.alpha_min if c == 3 else 0,
                                            config.alpha_max if c == 3 else 255))


# ============================================================
#                       MAIN GA LOOP
# ============================================================

def run_ga(config: GAConfig):

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(config.seed)

    # Auto-load target image
    print("[INFO] Loading:", TARGET_IMAGE)
    orig = Image.open(TARGET_IMAGE)
    ow, oh = orig.size
    aspect = ow / oh

    # compute width-height from ratio
    if TARGET_WIDTH == -1:
        render_h = TARGET_HEIGHT
        render_w = int(render_h * aspect)
    else:
        render_w = TARGET_WIDTH
        render_h = int(render_w / aspect)

    # override config size
    config.width = render_w
    config.height = render_h

    print(f"[INFO] Render size: {render_w} x {render_h} (auto ratio)")

    target_full = load_image(TARGET_IMAGE, render_w, render_h)

    # eval-scale image
    eval_w = max(32, int(render_w * config.eval_scale))
    eval_h = max(32, int(render_h * config.eval_scale))

    eval_img = Image.fromarray(target_full.astype(np.uint8))
    eval_img = eval_img.resize((eval_w, eval_h), Image.LANCZOS)
    target_eval = np.asarray(eval_img, dtype=np.float32)

    print(f"[INFO] Eval size: {eval_w} x {eval_h}")

    # Init population
    population = [Individual.random_ind(config, rng) for _ in range(config.population_size)]
    for ind in population:
        ind.fitness = mse(render(ind, eval_w, eval_h), target_eval)

    population.sort(key=lambda x: x.fitness)
    best = population[0].clone()

    out0 = os.path.join(OUTPUT_DIR, "best_gen_0000.png")
    Image.fromarray(render(best, render_w, render_h).astype(np.uint8)).save(out0)

    print("[START] GA Running...")

    for gen in range(1, config.generations + 1):

        new_pop = []
        # elitism
        for i in range(config.elitism):
            new_pop.append(population[i].clone())

        progress = gen / config.generations
        scale = 1.0 - 0.7 * progress

        while len(new_pop) < config.population_size:
            p1 = tournament_select(population, config, rng)
            p2 = tournament_select(population, config, rng)

            c1, c2 = p1.clone(), p2.clone()
            mutate(c1, config, rng, scale)
            mutate(c2, config, rng, scale)

            new_pop.extend([c1, c2])

        new_pop = new_pop[:config.population_size]

        # Evaluate
        for ind in new_pop:
            ind.fitness = mse(render(ind, eval_w, eval_h), target_eval)

        new_pop.sort(key=lambda x: x.fitness)
        population = new_pop

        if population[0].fitness < best.fitness:
            best = population[0].clone()

        if gen % 10 == 0:
            print(f"[GEN {gen}] best MSE {best.fitness:.2f}")

        if gen % config.save_every == 0:
            outp = os.path.join(OUTPUT_DIR, f"best_gen_{gen:04d}.png")
            Image.fromarray(render(best, render_w, render_h).astype(np.uint8)).save(outp)

    # FINAL OUTPUT
    final_path = os.path.join(OUTPUT_DIR, "best_final.png")
    Image.fromarray(render(best, render_w, render_h).astype(np.uint8)).save(final_path)
    print("[DONE] saved:", final_path)


# ============================================================
#                          ENTRY POINT
# ============================================================

if __name__ == "__main__":
    cfg = GAConfig(
        width=256, height=256,  # temporary, auto-adjust later
    )
    run_ga(cfg)
