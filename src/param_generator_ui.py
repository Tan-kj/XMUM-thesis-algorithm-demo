import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os

class ToolTip:
    def __init__(self, widget, text, root):
        self.widget = widget
        self.text = text
        self.root = root
        self.tw = None
        self.widget.bind("<Button-1>", self.show_tip)
        self.root.bind("<Button-1>", self.hide_tip, add="+")

    def show_tip(self, event=None):
        if self.tw:
            return
        # Calculate position
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 10
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("Arial", 9), padx=8, pady=6,
                         wraplength=460)
        label.pack()

    def hide_tip(self, event):
        if self.tw:
            if event.widget != self.widget:
                self.tw.destroy()
                self.tw = None

class ParamGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PCB GA Parameters Generator")
        self.root.geometry("600x700")

        self.params_def = [
            {"category": "1. Shared Parameters (Weights & Penalties)", "params": [
                {"name": "num_layers", "default": 2, "type": int,
                 "desc": "Number of PCB routing layers (Default: 2 = double-sided). More layers provide more 3D space for A* routing, reducing crossovers and congestion. However, routing grid size scales with layer count, increasing memory usage and pathfinding time. Used in both intra-group and global routing."},
                {"name": "grid_res", "default": 1.0, "type": float,
                 "desc": "Routing grid resolution (mm). A* pathfinding operates on grid points of size grid_res × grid_res. Smaller values yield finer paths and better obstacle avoidance, but grid size grows quadratically, significantly increasing memory and pathfinding time. For larger layouts, increase this value to keep execution time reasonable."},
                {"name": "wirelength_weight", "default": 1.0, "type": float,
                 "desc": "Weight of total wire length (total_wire_length) in the fitness function. Increasing this prioritizes shorter total traces; decreasing it reduces emphasis on wire length. Applied in both intra-group GA and two-stage global GA."},
                {"name": "capacitance_weight", "default": 1.5, "type": float,
                 "desc": "Weight of estimated parasitic capacitance (total_cap) in the fitness function. Calculated by summing coupling lengths of parallel trace segments between different nets. Increasing this minimizes parallel coupling but may conflict with wire length minimization. Set to 0 to disable capacitance calculations for faster speed. Shared by intra-group and global evaluations."},
                {"name": "base_wire_cross_penalty", "default": 500.0, "type": float,
                 "desc": "Base penalty score for wire crossovers. The actual penalty scales dynamically with generation count up to a 3x cap: dynamic_cross = min(base_wire_cross_penalty × 10^(gen/total_gens), 3 × base_wire_cross_penalty). The cap prevents fitness cliffs late in evolution, allowing layouts with minor crossovers to be repaired gradually. Crossovers are also constrained by A* occupancy penalties and Layout Score congestion terms."},
                {"name": "base_component_overlap_penalty", "default": 6000.0, "type": float,
                 "desc": "Base penalty score for component overlap, applied in two places: 1) Intra-group GA fitness evaluation; 2) Transition target value for overlap penalty in Stage 2 fine-grained GA. Computed based on overlap area and center distance, using large values (thousands) to strictly enforce non-overlapping components."},
                {"name": "group_base_component_overlap_penalty", "default": 6000.0, "type": float,
                 "desc": "Starting value of component overlap penalty during Stage 1 global evaluation (group transformation phase). In Stage 2, the penalty transitions linearly to base_component_overlap_penalty over 100 generations, enabling a smooth shift from group-level tolerance to component-level strictness."},
                {"name": "wire_comp_overlap_penalty", "default": 400.0, "type": float,
                 "desc": "Penalty score for traces passing through component body geometries (wire-component overlap). Accumulated by A* pathfinding for each grid point traversed inside a component, pushing traces away from components to prevent short circuits. Higher values create safer, wider bypass paths."},
                {"name": "via_penalty", "default": 3.5, "type": float,
                 "desc": "Penalty score for placing a Via. Serves as both a step cost for A* cross-layer movement and an equivalent wire length conversion factor in fitness (each cross-layer segment counts as this value towards total length). Overly high values restrict routing to single layers and increase congestion; overly low values produce excessive vias. Consider reducing for multi-layer boards."},
                {"name": "distance_weight", "default": 0.5, "type": float,
                 "desc": "Weight of pin MST straight-line distance (total_pin_dist) in fitness, decaying linearly across generations: current = distance_weight × max(0, 1 - gen/total_gens). Mainly pulls connected components closer during early evolution, fading out later to favor actual routing metrics."},
                {"name": "centripetal_factor", "default": 0.1, "type": float,
                 "desc": "Centripetal attraction coefficient for single-terminal nets within a group (global nets with only 1 pin in this group), effective only in intra-group GA. Centripetal Force = distance_weight × factor. Weakly pulls external-facing pins toward the group's geometric center (0,0), preventing components without internal net attraction from being repelled outside group boundaries."},
                {"name": "layout_score_cap_weight", "default": 1.5, "type": float,
                 "desc": "Weight of parasitic capacitance in Layout Score calculation. Layout Score is used strictly for output statistics (per-generation top score history and final Top-N ranking) and does NOT affect GA optimization target. Formula: score = total_cap × cap_weight + total_wire_length × wire_weight + congestion × cong_weight."},
                {"name": "layout_score_wire_weight", "default": 1.0, "type": float,
                 "desc": "Weight of wire length in Layout Score. Used only for output statistics and final ranking without affecting the GA optimization process. Defaults to match wirelength_weight (1.0)."},
                {"name": "layout_score_congestion_weight", "default": 5.0, "type": float,
                 "desc": "Weight of congestion in Layout Score. Used only for output statistics and final ranking without affecting the GA optimization process. Congestion is defined as: wire crossover count × 10 + component overlap score."},
                {"name": "component_repulsion_weight", "default": 30.0, "type": float,
                 "desc": "Weight of component repulsion penalty, applied in two places: 1) Intra-group GA fitness evaluation; 2) Transition target value for repulsion penalty in Stage 2 fine-grained GA. When center distance between two components falls below safety threshold (2 × sum of average semi-side lengths), penalty is computed as (threshold - distance) / threshold. Higher values spread components more evenly but increase routing distance."},
                {"name": "group_repulsion_weight", "default": 30.0, "type": float,
                 "desc": "Starting value of component repulsion penalty during Stage 1 global evaluation. In Stage 2, repulsion penalty transitions linearly to component_repulsion_weight over 100 generations. A larger difference between these values results in a more pronounced tightening of component spacing during Stage 2."}
            ]},
            {"category": "2. Group Iteration Parameters (Group GA)", "params": [
                {"name": "group_pop_size", "default": 300, "type": int,
                 "desc": "Population size for each intra-group GA. Larger populations allow more thorough search and higher local solution quality, but evaluation time and memory grow linearly. Note that each group runs an independent GA instance, so total evaluation load scales as Groups × Population × Generations. Recommended: 30–50 for quick tests."},
                {"name": "group_generations", "default": 500, "type": int,
                 "desc": "Total generations for each intra-group GA. More generations lead to better local convergence. If no improvement occurs for 5 consecutive generations, the engine automatically lowers mutation rate to accelerate convergence. Recommended: 50–100 for quick tests."},
                {"name": "group_routing_start_gen", "default": 200, "type": int,
                 "desc": "Generation at which intra-group GA begins actual physical routing (A* pathfinding). Prior generations evaluate only lightweight metrics like pin MST distance, overlap, and repulsion for fast execution. Setting to 0 starts routing from generation 1. Recommended: 20 for quick tests."},
                {"name": "group_mutation_rate", "default": 0.2, "type": float,
                 "desc": "Mutation rate for intra-group GA (0~1). Each component's coordinates and rotation angle undergo Gaussian translation (std dev = 5% of component movable range) or 90° rotation with this probability. Automatically reduces to max(original - 0.1, 0.01) if best fitness stalls for 5 generations."},
                {"name": "group_crossover_rate", "default": 0.7, "type": float,
                 "desc": "Crossover rate for intra-group GA (0~1). Performs two-point crossover on two parent chromosomes with this probability, swapping component genes (x/y/angle) within a random slice."},
                {"name": "group_tournament_size", "default": 3, "type": int,
                 "desc": "Number of competitors in intra-group tournament selection. Larger sizes increase selection pressure and convergence speed but risk premature convergence. Typical range: 2–5."},
                {"name": "group_area_coef", "default": 3.0, "type": float,
                 "desc": "Maximum allowable layout area expansion coefficient for a group: Allowed Area = Total Component Area in Group × Coefficient (square root gives local bounding side length). Values too small cause severe overlaps; values too large lead to longer intra-group routing and looser global assembly."},
                {"name": "top_n", "default": 5, "type": int,
                 "desc": "Number of topologically distinct local optimal solutions extracted per group for selection in global combination phase. Larger values broaden global exploration space in Stage 1, but increase global evaluation costs and Softmax mutation overhead. Must be ≤ group_pop_size and ≥ 1."},
                {"name": "diversity_threshold", "default": 1.0, "type": float,
                 "desc": "Threshold for topological similarity filtering of local solutions (mm). Two solutions are deemed identical and filtered out if all corresponding component coordinate differences are below this value and rotations match, ensuring diversity among Top-N solutions. Higher values relax filtering."}
            ]},
            {"category": "3. Global Outer Iteration Parameters (Two-Stage GA)", "params": [
                {"name": "out_pop_size", "default": 300, "type": int,
                 "desc": "Population size for the global two-stage GA. Larger populations improve global exploration, but each evaluation may include actual global routing, increasing runtime linearly. Elite retention ratio is set to 1/10. Recommended: 30–50 for quick tests."},
                {"name": "out_generations", "default": 3000, "type": int,
                 "desc": "Total generations for global two-stage GA (Stage 1 + Stage 2). Stage 1 explores group transformations and local solution combinations, while Stage 2 performs fine-grained component-level tuning. Both share this total generation count. Recommended: 100–300 for quick tests."},
                {"name": "out_routing_start_gen", "default": 600, "type": int,
                 "desc": "Generation at which global GA begins actual global routing (A* pathfinding for inter-group nets). Prior to this, evaluations use only group transformations and lightweight metrics. Typically set to ~20% of out_generations. Setting to 0 starts routing from generation 1. Recommended to set earlier than out_combine_start_gen to supply routed individuals to Stage 1."},
                {"name": "out_combine_start_gen", "default": 1200, "type": int,
                 "desc": "Generation at which the algorithm transitions to Stage 2 (fine-grained GA fine-tuning). Abstract group genes for all individuals are flattened into concrete component coordinates (comp_genes). Crossover/mutation then operate directly at component level, switching evaluation to incremental PR&R (Rip-up and Reroute) to preserve unaffected trace segments. Must be ≥ 1 and satisfy out_routing_start_gen ≤ value < out_generations. Recommended: 50 for quick tests."},
                {"name": "out_mutation_rate", "default": 0.2, "type": float,
                 "desc": "Global GA mutation rate (0~1). Stage 1: Trigger probability for group transformation translation/rotation and Softmax local solution re-selection. Stage 2: Trigger probability for component-level perturbation (randomly perturbing 1–3 components via Gaussian noise). Stage 1 diversity monitoring may temporarily increase this value."},
                {"name": "out_crossover_rate", "default": 0.7, "type": float,
                 "desc": "Global GA crossover rate (0~1). Stage 1: Two-point group-level crossover probability (swapping group transformations and local solution indices). Stage 2: Uniform component-level crossover probability (50% chance to swap x/y/angle per component). Stage 1 diversity monitoring may temporarily decrease this value."},
                {"name": "out_tournament_size", "default": 3, "type": int,
                 "desc": "Number of competitors in global tournament selection. Larger sizes increase selection pressure and convergence speed but risk premature convergence. Typical range: 2–5."},
                {"name": "ema_alpha", "default": 0.3, "type": float,
                 "desc": "Exponential Moving Average (EMA) smoothing coefficient (0~1], effective only in Stage 1. Tracks historical fitness of each local solution during global evaluations. Higher values lessen historical influence, favoring recent performance. Used for Softmax index selection during Stage 1 mutation."},
                {"name": "softmax_t0", "default": 5000.0, "type": float,
                 "desc": "Softmax initial temperature (Stage 1 only). When mutating local solution indices in Stage 1, higher temperature yields a flatter probability distribution, encouraging exploration of different local solutions. Temperature decays exponentially from t0 to tend."},
                {"name": "softmax_tend", "default": 1.0, "type": float,
                 "desc": "Softmax final temperature (Stage 1 only). As temperature drops to this value near the end of Stage 1, local solution selection leans almost entirely towards solutions with highest EMA scores (exploitation)."},
                {"name": "stage2_step_max_factor", "default": 0.05, "type": float,
                 "desc": "Upper bound factor for mutation step size in Stage 2 fine-tuning: Initial Step Size = global_side × Factor (global_side is estimated global layout dimension). Typical values range from 0.01 to 0.1; larger values allow bolder early exploration."},
                {"name": "stage2_step_min_factor", "default": 0.0005, "type": float,
                 "desc": "Lower bound factor for mutation step size in Stage 2 fine-tuning (linear decay endpoint): Step Size = global_side × [max_factor + (min_factor - max_factor) × progress], where progress goes from 0 to 1. Reaching this factor at the end of Stage 2 ensures coarse-to-fine convergence while preserving optimized global topology."},
                {"name": "out_full_reroute_interval", "default": 25, "type": int,
                 "desc": "Interval (in generations) for full global rerouting during Stage 2 (Default: 25). When gen % interval == 0 in Stage 2, the engine bypasses incremental PR&R and performs full global rerouting (evaluate_global) on all individuals. This refreshes true Layout Scores and removes stale, sub-optimal routing path artifacts, preventing routing deadlocks. Smaller values increase score accuracy at the expense of higher computation time. Recommended: 10–50, and must be < (out_generations - out_combine_start_gen)."}
            ]},
            {"category": "4. Diversity Control", "params": [
                {"name": "diversity_std_threshold", "default": 0.0001, "type": float,
                 "desc": "Fitness standard deviation threshold (Stage 1 only). Calculates fitness std dev for the top half of the population. If value falls below threshold, population convergence is detected and diversity emergency control triggers: mutation_rate += diversity_mutation_delta, crossover_rate -= diversity_crossover_delta. Set to 0 to disable monitoring."},
                {"name": "diversity_mutation_delta", "default": 0.005, "type": float,
                 "desc": "Step increment added to mutation rate upon population convergence (Stage 1 only). Once diversity recovers, mutation rate gradually decays back to its original value at a rate of 10% per generation."},
                {"name": "diversity_crossover_delta", "default": 0.2, "type": float,
                 "desc": "Step decrement subtracted from crossover rate upon population convergence (Stage 1 only). Reducing crossover and boosting mutation helps escape local optima. Crossover rate gradually recovers once population diversity is restored."}
            ]}
        ]
        
        self.entries = {}
        self.create_widgets()

    def create_widgets(self):
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="top", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        for category in self.params_def:
            labelframe = ttk.LabelFrame(scrollable_frame, text=category["category"], padding=(10, 5))
            labelframe.pack(fill="x", padx=10, pady=10, expand=True)
            
            for row, p in enumerate(category["params"]):
                lbl = ttk.Label(labelframe, text=p["name"] + ":")
                lbl.grid(row=row, column=0, sticky="e", padx=5, pady=5)
                
                entry = ttk.Entry(labelframe, width=15)
                entry.insert(0, str(p["default"]))
                entry.grid(row=row, column=1, sticky="w", padx=5, pady=5)
                
                self.entries[p["name"]] = (entry, p["type"])
                
                btn_help = tk.Label(labelframe, text="❓", fg="#0055ff", cursor="hand2", font=("Arial", 11))
                btn_help.grid(row=row, column=2, sticky="w", padx=5, pady=5)
                
                ToolTip(btn_help, p["desc"], self.root)

        bottom_frame = ttk.Frame(self.root, padding=(10, 10))
        bottom_frame.pack(side="bottom", fill="x")
        
        ttk.Label(bottom_frame, text="Filename:").grid(row=0, column=0, sticky="w", pady=5)
        self.filename_entry = ttk.Entry(bottom_frame, width=25)
        self.filename_entry.insert(0, "custom_params.json")
        self.filename_entry.grid(row=0, column=1, sticky="w", pady=5, padx=5)

        ttk.Label(bottom_frame, text="Save Path:").grid(row=1, column=0, sticky="w", pady=5)
        self.path_entry = ttk.Entry(bottom_frame, width=40)
        self.path_entry.insert(0, os.getcwd()) 
        self.path_entry.grid(row=1, column=1, sticky="w", pady=5, padx=5)
        
        ttk.Button(bottom_frame, text="Browse...", command=self.browse_dir).grid(row=1, column=2, padx=5, pady=5)
        
        generate_btn = ttk.Button(bottom_frame, text="✔ Generate JSON Parameter File", command=self.generate_json)
        generate_btn.grid(row=2, column=0, columnspan=3, pady=15)

    def browse_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, dir_path.replace("\\\\", "/").replace("\\", "/"))

    def generate_json(self):
        result = {}
        for name, (entry, val_type) in self.entries.items():
            raw_val = entry.get().strip()
            try:
                result[name] = val_type(raw_val)
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid value '{raw_val}' for parameter '{name}'.\nExpected format: {val_type.__name__}")
                return
        
        raw_path = self.path_entry.get().strip()
        clean_path = raw_path.replace("\\\\", "/").replace("\\", "/")
        
        filename = self.filename_entry.get().strip()
        
        if not clean_path or not filename:
            messagebox.showerror("Error", "Please fill in both the save path and filename!")
            return
            
        full_path = os.path.join(clean_path, filename)
        
        try:
            os.makedirs(clean_path, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Success", f"Parameter file generated successfully!\n\nSaved at:\n{full_path}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Failed to save file. Check file permissions:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ParamGeneratorApp(root)
    root.mainloop()