import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
import warnings

# Suppress annoying matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

class TerminalUI:
    def __init__(self):
        # Terminal Setup
        console.clear()
        console.print(Panel.fit("[bold #00bfff]KANFAI OS v1.0[/]\n[#e6edf3]Neural Link Established.[/]", border_style="#4169e1"))
        
        # Matplotlib Blue/Ice Dark Setup
        plt.style.use("dark_background")
        plt.ion()
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.canvas.manager.set_window_title("KANFAI BIOMETRICS [DEEP BLUE]")
        
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
        self.history = {k: {chem: [] for chem in v} for k, v in self.groups.items()}
        self.time_steps = []

    def update_plot(self, state_dict, step):
        self.time_steps.append(step)
        for ax in self.axs.flat:
            ax.clear()
            
        for idx, (title, chemicals) in enumerate(self.groups.items()):
            row = idx // 2
            col = idx % 2
            ax = self.axs[row, col]
            
            ax.set_title(title, color=self.text_color, fontfamily='monospace', fontsize=10)
            ax.set_facecolor(self.panel_color)
            ax.set_ylim(0, 1.05)
            ax.grid(True, color=self.grid_color, linestyle='-', alpha=0.8)
            ax.tick_params(colors=self.text_color)
            
            # Hide outer spines for a cleaner look
            for spine in ax.spines.values():
                spine.set_color(self.grid_color)
            
            for c_idx, chem in enumerate(chemicals):
                self.history[title][chem].append(state_dict[chem])
                color = self.colors[c_idx % len(self.colors)]
                ax.plot(self.time_steps, self.history[title][chem], label=chem.upper(), 
                        color=color, linewidth=2, marker='o', markersize=4, 
                        markeredgecolor=self.bg_color, markeredgewidth=1)
            
            legend = ax.legend(loc="upper left", fontsize="x-small", bbox_to_anchor=(1.02, 1), 
                               facecolor=self.panel_color, edgecolor=self.grid_color, labelcolor=self.text_color)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def get_input(self):
        return console.input("\n[bold #c28b62]USER >[/] ")

    def print_ai_start(self):
        console.print("[bold #e6c19c]KANFAI >[/] ", end="")
        
    def print_ai_chunk(self, chunk):
        print(chunk, end="", flush=True)
    
    def print_ai_end(self):
        print()
