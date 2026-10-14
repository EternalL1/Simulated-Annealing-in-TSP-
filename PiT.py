"""
Traveling Salesperson Problem (TSP) - Simulated Annealing
Application : Downtown Cagayan de Oro City tourist itinerary
Objective   : Minimize total driving distance (km), start and end at V0

Algorithm   : Simulated Annealing, following Algorithm 2.5 of the assigned paper
              (random start tour -> swap two cities -> accept if better, else
              accept with probability e^(-delta/T) -> cool T = alpha * T ->
              stop when T is low / no improvement / max iterations ->
              output the shortest tour found).

Run         : python tsp_simulated_annealing.py
Output      : console report + distance_matrix.csv (the dataset used)
              + two figures (graph model with best route, SA convergence)
              saved as tsp_route_graph.png and tsp_convergence.png
Requires    : matplotlib (pip install matplotlib) for the figures only
"""

import csv
import itertools
import math
import random
import time

# =============================================================================
# SECTION 1: DATASET  (this is the dataset used by the implementation)
# =============================================================================
# !!! Set to False ONLY after you replace the matrix below with your real
# !!! Google Maps driving distances.
USING_PLACEHOLDER_DATA = True

# Vertex labels and the locations they represent (check each pin on Google Maps).
VERTICES = [
    "V0 Gaston Park / Divisoria (start/end)",
    "V1 St. Augustine Metropolitan Cathedral",
    "V2 Museo de Oro (Xavier University)",
    "V3 Limketkai Center",
    "V4 Cogon Market",
    "V5 Centrio Mall",
    "V6 SM CDO Downtown Premier",
    "V7 Pelaez Sports Center",
    "V8 Cagayan de Oro City Hall",
    "V9 Capitol University",
]

# DIST[i][j] = driving distance in km from vertex i to vertex j (DIRECTED).
# Row = from, column = to. Diagonal = 0.
# Data source (real data): Google Maps, driving mode, read in each direction.
# Assumed data: no traffic effects, fixed route suggested by Google Maps,
#               no time spent at each stop.
# The values below are PLACEHOLDERS for testing only, NOT real measurements.
DIST = [
    [ 0.0,  1.8,  1.5,  2.4,  2.1,  1.3,  3.5,  1.1,  4.6,  1.8],
    [ 1.8,  0.0,  1.7,  3.8,  3.8,  2.9,  4.0,  2.8,  4.7,  1.7],
    [ 1.6,  1.7,  0.0,  2.6,  2.6,  2.9,  2.5,  2.2,  3.1,  0.3],
    [ 2.4,  3.9,  2.6,  0.0,  0.4,  2.1,  2.5,  1.6,  3.7,  2.8],
    [ 2.0,  3.8,  2.7,  0.4,  0.0,  1.9,  2.9,  1.2,  4.1,  2.9],
    [ 1.4,  3.1,  2.9,  2.2,  1.8,  0.0,  4.1,  0.8,  5.4,  3.2],
    [ 3.7,  4.1,  2.5,  2.6,  2.8,  4.3,  0.0,  3.4,  1.2,  2.3],
    [ 1.1,  2.8,  2.2,  1.5,  1.2,  0.7,  3.5,  0.0,  4.7,  2.6],
    [ 4.6,  4.7,  3.0,  3.8,  4.1,  5.5,  1.3,  4.5,  0.0,  2.8],
    [ 1.8,  1.7,  0.3,  2.8,  2.8,  3.0,  2.4,  2.5,  2.9,  0.0],
]

START = 0  # index of the starting (and ending) vertex

# OPTIONAL map coordinates (latitude, longitude) of each vertex, in the same
# order as VERTICES. Right-click a pin in Google Maps to copy them. If left as
# None, the graph is drawn with the vertices evenly spaced on a circle.
COORDS = None
# Example: COORDS = [(8.4780, 124.6470), (8.4772, 124.6465), ...]  # 10 pairs

SHOW_PLOTS = True   # open the figure windows (needed for the screen recording)
SAVE_PLOTS = True   # also save the figures as PNG files

# =============================================================================
# SECTION 2: SIMULATED ANNEALING PARAMETERS
# =============================================================================
SEED = 42              # fixed seed so the demo run is repeatable
T_INITIAL = 10.0       # step 3: initial high temperature (on the scale of tour distance in km)
ALPHA = 0.95           # step 11: cooling schedule T = alpha * T, 0 < alpha < 1
T_MIN = 1e-3           # step 12: stop when temperature is sufficiently low
MOVES_PER_TEMP = 100   # candidate swaps tried at each temperature level
NO_IMPROVE_LIMIT = 40  # step 12: stop after this many temperature levels
                       #          with no new best tour
MAX_ITERATIONS = 200000  # hard cap on total iterations


# =============================================================================
# SECTION 3: HELPER FUNCTIONS
# =============================================================================
def tour_distance(tour, dist):
    """Total distance of a closed tour. tour = list of vertex indices that
    starts at START and does NOT repeat the start at the end; the return leg
    to START is added here. This is the O(n) evaluation step in the paper."""
    total = 0.0
    for k in range(len(tour)):
        a = tour[k]
        b = tour[(k + 1) % len(tour)]
        total += dist[a][b]
    return total


def random_tour(n, start):
    """Step 2: random starting tour (START fixed in position 0)."""
    others = [v for v in range(n) if v != start]
    random.shuffle(others)
    return [start] + others


def swap_neighbor(tour):
    """Step 4: new solution by swapping the positions of two cities.
    START (position 0) is never moved, so every tour begins/ends at START."""
    new = tour[:]
    i, j = random.sample(range(1, len(tour)), 2)
    new[i], new[j] = new[j], new[i]
    return new


# =============================================================================
# SECTION 4: SIMULATED ANNEALING (Algorithm 2.5)
# =============================================================================
def simulated_annealing(dist, start):
    n = len(dist)
    current = random_tour(n, start)                 # step 2
    current_cost = tour_distance(current, dist)
    best, best_cost = current[:], current_cost
    T = T_INITIAL                                   # step 3

    iterations = 0
    accepted_worse = 0
    stale_levels = 0
    history = []  # (temperature, best cost so far) for optional inspection
    trace_current, trace_best = [], []  # per-iteration costs for the figure

    while T > T_MIN and stale_levels < NO_IMPROVE_LIMIT \
            and iterations < MAX_ITERATIONS:        # step 12 (stop rules)
        improved_this_level = False

        for _ in range(MOVES_PER_TEMP):
            iterations += 1
            candidate = swap_neighbor(current)              # step 4
            cand_cost = tour_distance(candidate, dist)      # step 5
            delta = cand_cost - current_cost                # new - current

            if delta < 0:                                   # steps 6-7
                accept = True
            else:                                           # steps 8-9
                accept = random.random() < math.exp(-delta / T)
                if accept and delta > 0:
                    accepted_worse += 1

            if accept:
                current, current_cost = candidate, cand_cost
                # Keep the best tour ever seen (step 13 returns the shortest
                # tour found, which may not be the final current tour).
                if current_cost < best_cost - 1e-12:
                    best, best_cost = current[:], current_cost
                    improved_this_level = True

            trace_current.append(current_cost)
            trace_best.append(best_cost)

            if iterations >= MAX_ITERATIONS:
                break

        stale_levels = 0 if improved_this_level else stale_levels + 1
        history.append((T, best_cost))
        T *= ALPHA                                          # step 11

    stats = {
        "iterations": iterations,
        "final_temperature": T,
        "worse_moves_accepted": accepted_worse,
        "temperature_levels": len(history),
        "trace_current": trace_current,
        "trace_best": trace_best,
    }
    return best, best_cost, stats


# =============================================================================
# SECTION 5: VERIFICATION
# =============================================================================
def validate_tour(tour, n, start):
    """Every vertex exactly once, starting at START (return leg is implied)."""
    ok_start = tour[0] == start
    ok_all = sorted(tour) == list(range(n))
    return ok_start and ok_all


def brute_force(dist, start):
    """Exact optimum by checking all (n-1)! tours. Practical for n <= ~11.
    Used only as evidence for how close SA got; it is NOT part of SA."""
    n = len(dist)
    others = [v for v in range(n) if v != start]
    best_cost, best_tour = float("inf"), None
    for perm in itertools.permutations(others):
        tour = [start] + list(perm)
        cost = tour_distance(tour, dist)
        if cost < best_cost:
            best_cost, best_tour = cost, tour
    return best_tour, best_cost


def print_route(tour, dist, label_list):
    """Print each edge with its weight and recompute the total from them."""
    closed = tour + [tour[0]]
    running = 0.0
    print(f"{'Leg':<5}{'From':<45}{'To':<45}{'km':>6}")
    for k in range(len(closed) - 1):
        a, b = closed[k], closed[k + 1]
        w = dist[a][b]
        running += w
        print(f"{k + 1:<5}{label_list[a]:<45}{label_list[b]:<45}{w:>6.1f}")
    print(f"{'':<95}{'-' * 6}")
    print(f"{'Recalculated total from edge weights:':<95}{running:>6.1f} km")
    return running


def export_dataset(path="distance_matrix.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from\\to"] + [v.split()[0] for v in VERTICES])
        for i, row in enumerate(DIST):
            w.writerow([VERTICES[i].split()[0]] + row)
    return path


# =============================================================================
# SECTION 5B: VISUALIZATION (graph model + SA convergence)
# =============================================================================
def _node_positions(n):
    """(x, y) for each vertex: real coordinates if COORDS is set, otherwise a
    circle with the start vertex V0 at the top."""
    if COORDS is not None:
        assert len(COORDS) == n, "COORDS needs one (lat, lon) pair per vertex."
        return [(lon, lat) for lat, lon in COORDS]
    return [(math.cos(math.pi / 2 - 2 * math.pi * k / n),
             math.sin(math.pi / 2 - 2 * math.pi * k / n)) for k in range(n)]


def plot_graph(tour, cost, dist, labels, path="tsp_route_graph.png"):
    """Weighted graph: faint lines = all possible travel connections,
    bold numbered arrows = the best route found, labels = km per leg."""
    import matplotlib.pyplot as plt
    n = len(dist)
    pos = _node_positions(n)
    fig, ax = plt.subplots(figsize=(14, 8))

    # All possible edges (complete graph) drawn faintly for context.
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                    color="grey", alpha=0.13, lw=0.8, zorder=1)

    # Best route as numbered directed arrows with distance labels.
    closed = tour + [tour[0]]
    for k in range(n):
        a, b = closed[k], closed[k + 1]
        ax.annotate("", xy=pos[b], xytext=pos[a], zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.4,
                                    shrinkA=17, shrinkB=17,
                                    connectionstyle="arc3,rad=0.08"))
        mx = (pos[a][0] + pos[b][0]) / 2
        my = (pos[a][1] + pos[b][1]) / 2
        ax.text(mx, my, f"{k + 1}) {dist[a][b]:.1f} km", fontsize=9,
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#d62728", lw=0.8))

    # Vertices (start vertex highlighted).
    for v in range(n):
        is_start = v == START
        ax.scatter(*pos[v], s=900, zorder=4,
                   color="#2ca02c" if is_start else "#1f77b4",
                   edgecolor="black", linewidth=1.2)
        ax.text(pos[v][0], pos[v][1], f"V{v}", color="white", fontsize=11,
                fontweight="bold", ha="center", va="center", zorder=6)

    ax.set_title(f"Best route found by Simulated Annealing: {cost:.1f} km",
                 fontsize=14, fontweight="bold")
    ax.set_aspect("equal" if COORDS is None else "auto")
    ax.margins(0.12)
    if COORDS is None:
        ax.axis("off")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Legend panel: vertex key.
    key = "\n".join(labels)
    key += "\n\nGreen node = start/end"
    key += "\nRed arrows = best route"
    key += "\nLabel = leg number) distance"
    key += "\nGrey lines = other possible edges"
    fig.text(0.60, 0.5, key, fontsize=9, va="center", family="monospace")
    fig.subplots_adjust(left=0.0, right=0.58)
    if SAVE_PLOTS:
        fig.savefig(path, dpi=150)
    return fig


def plot_convergence(stats, optimum=None, path="tsp_convergence.png"):
    """How SA improved: current tour cost (wanders, accepts worse moves while
    hot) versus the best cost found so far (only ever goes down)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(stats["trace_current"], color="#1f77b4", alpha=0.35, lw=0.8,
            label="Current tour distance")
    ax.plot(stats["trace_best"], color="#d62728", lw=2.2,
            label="Best distance found so far")
    if optimum is not None:
        ax.axhline(optimum, color="#2ca02c", ls="--", lw=1.6,
                   label=f"Exact optimum (brute force): {optimum:.1f} km")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total tour distance (km)")
    ax.set_title("Simulated Annealing convergence", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(path, dpi=150)
    return fig


def make_figures(best, best_cost, stats, optimum):
    try:
        import matplotlib
        if not SHOW_PLOTS:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: run 'pip install matplotlib' "
              "to see the figures.)")
        return
    plot_graph(best, best_cost, DIST, VERTICES)
    plot_convergence(stats, optimum)
    if SAVE_PLOTS:
        print("\nFigures saved: tsp_route_graph.png, tsp_convergence.png")
    if SHOW_PLOTS:
        plt.show()


# =============================================================================
# SECTION 6: MAIN
# =============================================================================
def main():
    n = len(DIST)
    assert all(len(r) == n for r in DIST), "Distance matrix must be square."
    assert len(VERTICES) == n, "One label per vertex is required."

    print("=" * 70)
    print("TSP - SIMULATED ANNEALING | Downtown Cagayan de Oro tourist route")
    print("=" * 70)
    if USING_PLACEHOLDER_DATA:
        print("*** WARNING: PLACEHOLDER DATA. Replace DIST with real Google")
        print("*** Maps distances and set USING_PLACEHOLDER_DATA = False.\n")

    print("INPUT: vertices")
    for i, v in enumerate(VERTICES):
        print(f"  {v}")
    print("\nINPUT: directed distance matrix (km, row = from, col = to)")
    print("      " + "".join(f"{'V' + str(j):>6}" for j in range(n)))
    for i, row in enumerate(DIST):
        print(f"  V{i:<3}" + "".join(f"{x:>6.1f}" for x in row))
    print(f"\nExported dataset: {export_dataset()}")

    random.seed(SEED)
    print(f"\nParameters: seed={SEED}, T0={T_INITIAL}, alpha={ALPHA}, "
          f"Tmin={T_MIN}, moves/temp={MOVES_PER_TEMP}, "
          f"no-improve limit={NO_IMPROVE_LIMIT}")

    t0 = time.time()
    best, best_cost, stats = simulated_annealing(DIST, START)
    sa_time = time.time() - t0

    print("\n" + "-" * 70)
    print("SIMULATED ANNEALING RESULT")
    print("-" * 70)
    print("Route :", " -> ".join(f"V{v}" for v in best + [best[0]]))
    print(f"Total distance (SA reported): {best_cost:.1f} km")
    print(f"Iterations: {stats['iterations']} | temperature levels: "
          f"{stats['temperature_levels']} | worse moves accepted: "
          f"{stats['worse_moves_accepted']} | time: {sa_time:.3f} s\n")

    recalculated = print_route(best, DIST, VERTICES)

    print("\nCHECKS")
    print("  Every vertex visited exactly once, starts/ends at V0:",
          validate_tour(best, n, START))
    print("  Recalculated total matches SA total:",
          abs(recalculated - best_cost) < 1e-9)

    opt_cost = None
    if n <= 11:
        print("\nBRUTE-FORCE VERIFICATION (exact optimum, for comparison only)")
        t0 = time.time()
        opt_tour, opt_cost = brute_force(DIST, START)
        print("  Optimal route :",
              " -> ".join(f"V{v}" for v in opt_tour + [opt_tour[0]]))
        print(f"  Optimal total : {opt_cost:.1f} km "
              f"(computed in {time.time() - t0:.1f} s)")
        gap = (best_cost - opt_cost) / opt_cost * 100
        print(f"  SA total      : {best_cost:.1f} km | gap to optimum: "
              f"{gap:.2f}%")
        print("  SA found the optimum on this run." if gap < 1e-9
              else "  SA is approximate: it did not reach the optimum.")
    print("\nNote: SA is probabilistic and gives an approximate solution;")
    print("optimality is only claimed where the brute-force check confirms it.")

    make_figures(best, best_cost, stats, opt_cost)


if __name__ == "__main__":
    main()