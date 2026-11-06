from collections import deque
import warnings

import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel

# Suppress annoying matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

MAX_HISTORY_STEPS = 100


class TerminalUI:
    def __init__(self):
        # Terminal Setup
        try:
            console.clear()
        except Exception:
            pass
        console.print(Panel.fit("[bold #00bfff]KANFAI OS v1.0[/]\n[#e6edf3]Neural Link Established.[/]", border_style="#4169e1"))

        # Matplotlib Blue/Ice Dark Setup
        plt.style.use("dark_background")
        try:
            plt.ion()
        except Exception:
            pass
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 10))
        if getattr(self.fig.canvas, "manager", None) is not None:
            try:
                self.fig.canvas.manager.set_window_title("KANFAI BIOMETRICS [DEEP BLUE]")
            except Exception:
                pass

        # Deep space/ocean blue background
        self.bg_color = '#090c10'
        self.panel_color = '#161b22'
        self.text_color = '#e6edf3'
        self.grid_color = '#30363d'

        self.fig.patch.set_facecolor(self.bg_color)
        self.fig.suptitle("REAL-TIME NEURO-MATRIX", fontsize=16, color='#00bfff', fontweight='bold', fontfamily='monospace')

        self.groups = {
            "EXCITATORY & DRIVE": ["dopamine", "norepinephrine", "glutamate", "histamine", "adrenaline"],
            "INHIBITORY & MOOD": ["serotonin", "gaba", "glycine", "prolactin", "dhea"],
            "STRESS & SURVIVAL": ["cortisol", "oxytocin", "vasopressin", "endorphin", "acetylcholine"],
            "METABOLIC & SLEEP": ["adenosine", "melatonin", "ghrelin", "leptin", "insulin"]
        }

        # Highly readable, distinct neon colors for the data lines
        self.colors = ['#00FFFF', '#FFA500', '#39FF14', '#FF00FF', '#FFFF00']
        self.history = {
            k: {chem: deque(maxlen=MAX_HISTORY_STEPS) for chem in v}
            for k, v in self.groups.items()
        }
        self.time_steps = deque(maxlen=MAX_HISTORY_STEPS)
        self.lines = {}

        # Initialize axes, grids, legends, and persistent Line2D artists once to avoid full redraws every turn
        for idx, (title, chemicals) in enumerate(self.groups.items()):
            row = idx // 2
            col = idx % 2
            ax = self.axs[row, col]

            ax.set_title(title, color=self.text_color, fontfamily='monospace', fontsize=10)
            ax.set_facecolor(self.panel_color)
            ax.set_ylim(0, 1.05)
            ax.grid(True, color=self.grid_color, linestyle='-', alpha=0.8)
            ax.tick_params(colors=self.text_color)

            for spine in ax.spines.values():
                spine.set_color(self.grid_color)

            self.lines[title] = {}
            for c_idx, chem in enumerate(chemicals):
                color = self.colors[c_idx % len(self.colors)]
                (line,) = ax.plot(
                    [], [],
                    label=chem.upper(),
                    color=color,
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    markeredgecolor=self.bg_color,
                    markeredgewidth=1
                )
                self.lines[title][chem] = line

            ax.legend(
                loc="upper left",
                fontsize="x-small",
                bbox_to_anchor=(1.02, 1),
                facecolor=self.panel_color,
                edgecolor=self.grid_color,
                labelcolor=self.text_color
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_plot(self, state_dict, step):
        self.time_steps.append(step)
        x_data = list(self.time_steps)
        x_min = x_data[0]
        x_max = max(x_min + 1, x_data[-1])

        for idx, (title, chemicals) in enumerate(self.groups.items()):
            row = idx // 2
            col = idx % 2
            ax = self.axs[row, col]
            ax.set_xlim(x_min - 0.2, x_max + 0.2)

            for chem in chemicals:
                self.history[title][chem].append(state_dict[chem])
                self.lines[title][chem].set_data(x_data, list(self.history[title][chem]))

        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def get_input(self):
        return console.input("\n[bold #c28b62]USER >[/] ")

    def print_ai_start(self):
        console.print("[bold #e6c19c]KANFAI >[/] ", end="")

    def print_ai_chunk(self, chunk):
        print(chunk, end="", flush=True)

    def print_ai_end(self):
        print()
