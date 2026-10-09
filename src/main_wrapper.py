import json
import os
import csv
import pcb_engine
import time
import argparse
import sys
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely import affinity


#  scoring weights
LAYOUT_SCORE_WEIGHTS = {
    'parasitic_capacitance': 1.5,
    'wire_length': 1.0,
    'congestion': 5.0
}

class DualLogger:
    def __init__(self, filepath, stream):
        self.terminal = stream
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_config(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_fitness_history(history, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'fitness'])
        for gen, fit in enumerate(history):
            writer.writerow([gen, fit])

def save_fitness_components(history, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'generation', 'total_wire_length', 'total_cap', 'overlap_score',
            'wire_cross_count', 'wire_overlap_len', 'total_pin_dist',
            'repulsion_penalty', 'fitness'
        ])
        for gen, comps in enumerate(history):
            writer.writerow([gen] + list(comps))

def save_individual_json(ind, idx, filepath, num_layers):
    layer_names = ["F.Cu"] + [f"In{i}.Cu" for i in range(1, num_layers)]
    ind_data = {
        "metadata": {
            "id": idx, "target": "KiCad 10.0",
            "fitness": ind.get("fitness", 0),
            "total_wire_length": ind.get("total_wire_length", 0),
            "total_cap": ind.get("total_cap", 0),
            "overlap_score": ind.get("overlap_score", 0),
            "wire_cross_count": ind.get("wire_cross_count", 0)
        },
        "nets_info": [], "footprints": [], "tracks": [], "vias": []
    }

    for comp_id, comp in ind["comps"].items():
        comp_info = {
            "reference": comp_id,
            "footprint_lib_link": "Package_DIP:DIP-8_W7.62mm",
            "position": {"x": comp["x"], "y": comp["y"]},
            "rotation_deg": comp["angle"],
            "layer": "F.Cu", "pads": []
        }
        if "pins" in ind and comp_id in ind["pins"]:
            for pin_name, pos in ind["pins"][comp_id].items():
                comp_info["pads"].append({
                    "pad_name": pin_name, "position": {"x": pos[0], "y": pos[1]}
                })
        ind_data["footprints"].append(comp_info)

    net_index = 1
    for net_id, segments in ind["net_segments"].items():
        ind_data["nets_info"].append({"net_name": net_id, "net_code": net_index})
        for seg in segments:
            x1, y1, l1 = seg[0]
            x2, y2, l2 = seg[1]
            if l1 == l2:
                layer_name = layer_names[l1] if l1 < len(layer_names) else f"Layer{l1}"
                ind_data["tracks"].append({
                    "net_name": net_id, "net_code": net_index,
                    "start": {"x": x1, "y": y1}, "end": {"x": x2, "y": y2},
                    "width": 0.25, "layer": layer_name
                })
            else:
                ind_data["vias"].append({
                    "net_name": net_id, "net_code": net_index,
                    "position": {"x": x1, "y": y1}, "size": 0.8,
                    "drill": 0.4, "layers": layer_names.copy()
                })
        net_index += 1

    with open(filepath, 'w') as f:
        json.dump(ind_data, f, indent=2)

def save_layout_score_history(history, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'layout_score'])
        for gen, score in history:
            writer.writerow([gen, score])

def component_polygon(cx, cy, width, height, angle_deg):
    rect = box(-width/2, -height/2, width/2, height/2)
    rotated = affinity.rotate(rect, angle_deg, origin=(0, 0))
    translated = affinity.translate(rotated, cx, cy)
    return translated

def read_fitness_history(csv_path):
    generations, fitness = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row['generation']))
            fitness.append(float(row['fitness']))
    return generations, fitness

def plot_fitness(history_csv, output_path, title=None):
    gens, fits = read_fitness_history(history_csv)
    plt.figure(figsize=(10, 6))
    plt.plot(gens, fits, label='Best Fitness', color='blue', linewidth=2)
    plt.title(title if title else 'Best Fitness Value of the PCB Layout')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Fitness history saved to {output_path}")

def plot_fitness_components(components_csv, output_dir):
    components_info = {
        'total_wire_length': 'Total Wire Length (mm)',
        'total_cap': 'Total Capacity Score',
        'overlap_score': 'Overlap Score',
        'wire_cross_count': 'Wire Cross Count',
        'wire_overlap_len': 'Wire and Component Overlap Length (mm)',
        'total_pin_dist': 'Total Pin Distance (in mm)',
        'repulsion_penalty': 'Repulsion Penalty',
        'fitness': 'Total Fitness'
    }
    data = {key: ([], []) for key in components_info}
    with open(components_csv, 'r') as f:
        for row in csv.DictReader(f):
            gen = int(row['generation'])
            for key in components_info:
                if key in row:
                    data[key][0].append(gen)
                    data[key][1].append(float(row[key]))

    for key, title in components_info.items():
        gens, vals = data[key]
        if not gens: continue
        plt.figure(figsize=(10, 6))
        plt.plot(gens, vals, label=title, color='blue', linewidth=2)
        plt.title(title)
        plt.xlabel('Generation')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"{key}.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[*] {title} plot saved to {filepath}")

def plot_layout_score_history(history_csv, output_path):
    generations, scores = [], []
    with open(history_csv, 'r') as f:
        for row in csv.DictReader(f):
            generations.append(int(row['generation']))
            scores.append(float(row['layout_score']))
    if not generations:
        print("[!] No layout score history to plot, skipping.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(generations, scores, label='Best Layout Score', color='green', linewidth=2)
    plt.title('Best Layout Score of the PCB Layout')
    plt.xlabel('Generation')
    plt.ylabel('Layout Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Layout score history saved to {output_path}")

def plot_individual_layout(config, json_path, output_path, num_layers=2):
    with open(json_path, 'r') as f:
        ind_data = json.load(f)

    meta = ind_data["metadata"]
    layer_names = ["F.Cu"] + [f"In{i}.Cu" for i in range(1, num_layers)]
    layer_index = {name: idx for idx, name in enumerate(layer_names)}

    ind_comps = {}
    ind_pins = {}
    for fp in ind_data.get("footprints", []):
        cid = fp["reference"]
        ind_comps[cid] = {
            'x': fp["position"]["x"], 'y': fp["position"]["y"],
            'angle': fp["rotation_deg"]
        }
        ind_pins[cid] = {
            pad["pad_name"]: (pad["position"]["x"], pad["position"]["y"])
            for pad in fp.get("pads", [])
        }

    ind_segs = {}
    for tr in ind_data.get("tracks", []):
        layer = layer_index.get(tr.get("layer"), 0)
        seg = ((tr["start"]["x"], tr["start"]["y"], layer),
               (tr["end"]["x"], tr["end"]["y"], layer))
        ind_segs.setdefault(tr["net_name"], []).append(seg)
    for via in ind_data.get("vias", []):
        x, y = via["position"]["x"], via["position"]["y"]
        ind_segs.setdefault(via["net_name"], []).append(((x, y, 0), (x, y, 1)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10.colors
    layer_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]

    for comp in config["components"]:
        cid = comp["id"]
        if cid not in ind_comps:
            continue
        params = ind_comps[cid]
        poly = component_polygon(params['x'], params['y'], comp['width'], comp['height'], params['angle'])
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.3, color='lightblue', edgecolor='black', linewidth=1)
        ax.text(params['x'], params['y'], cid, ha='center', va='center', fontsize=8)

        for pin_name, (px, py) in ind_pins[cid].items():
            ax.plot(px, py, 'ro', markersize=4)
            ax.text(px, py + 0.2, f"{cid}.{pin_name}", fontsize=6, ha='center')

    net_ids = list(ind_segs.keys())
    for idx, net_id in enumerate(net_ids):
        color = colors[idx % len(colors)]
        segs = ind_segs[net_id]
        current_layer, current_path_x, current_path_y = None, [], []

        def flush_path():
            if len(current_path_x) > 1:
                style = layer_linestyles[current_layer % len(layer_linestyles)]
                ax.plot(current_path_x, current_path_y, color=color, linewidth=1.5, linestyle=style)

        for seg in segs:
            (x1, y1, l1), (x2, y2, l2) = seg
            if l1 == l2:
                if current_layer != l1 or (current_path_x and (current_path_x[-1] != x1 or current_path_y[-1] != y1)):
                    flush_path()
                    current_layer, current_path_x, current_path_y = l1, [x1, x2], [y1, y2]
                else:
                    current_path_x.append(x2)
                    current_path_y.append(y2)
            else:
                flush_path()
                current_layer, current_path_x, current_path_y = None, [], []
                ax.plot(x1, y1, marker='o', markersize=4, color='black', markerfacecolor='white', zorder=5)
        flush_path()

    title_str = f"Fitness: {round(float(meta.get('fitness', 0.0)), 2)}"
    ax.set_title(title_str)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Individual layout saved to {output_path}")


# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB Routing Engine CLI with Plotting")
    parser.add_argument("-c", "--config", required=True, help="输入布局 JSON 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出主目录路径")
    parser.add_argument("-p", "--params", required=False, help="外部全局参数 JSON 文件路径")
    args = parser.parse_args()
    
    config_path = args.config
    base_dir = args.output
    params_path = args.params

    #   base_dir/logs/            -> log and terminal output
    #   base_dir/csv/             -> each generation's best fitness and components data
    #   base_dir/csv/group_fitness/ -> each group GA's best fitness per generation
    #   base_dir/json/            -> eacdh best individual (only save once per individual)
    #   base_dir/png/             -> all plots
    #   base_dir/png/group_fitness/ -> each group GA's best fitness plot
    logs_dir = os.path.join(base_dir, "logs")
    csv_dir = os.path.join(base_dir, "csv")
    group_csv_dir = os.path.join(csv_dir, "group_fitness")
    json_dir = os.path.join(base_dir, "json")
    png_dir = os.path.join(base_dir, "png")
    png_component_dir = os.path.join(png_dir, "fitness_components")
    group_png_dir = os.path.join(png_dir, "group_fitness")
    for d in [logs_dir, csv_dir, group_csv_dir, json_dir, png_dir, png_component_dir, group_png_dir]:
        os.makedirs(d, exist_ok=True)

    log_file_path = os.path.join(logs_dir, "terminal_log.txt")
    sys.stdout = DualLogger(log_file_path, sys.stdout)
    sys.stderr = DualLogger(log_file_path, sys.stderr)

    print("=" * 50)
    print("PCB Auto-Routing & GA Engine (Two-Level)")
    print("=" * 50)
    
    if not os.path.exists(config_path):
        print(f"[!] Error: The specified config file does not exist: {config_path}")
        sys.exit(1)

    # default GLOBAL_PARAMS
    GROUP_GENERATIONS = 500
    OUT_GENERATIONS = 3000
    GLOBAL_PARAMS = {
        "num_layers": 2, "grid_res": 1.0, "wirelength_weight": 1.0, "capacitance_weight": 1.5,
        "base_wire_cross_penalty": 500.0, "base_component_overlap_penalty": 6000.0,
        "wire_comp_overlap_penalty": 400.0, "via_penalty": 3.5, "distance_weight": 0.5,
        "centripetal_factor": 0.35,
        "layout_score_cap_weight": LAYOUT_SCORE_WEIGHTS['parasitic_capacitance'],
        "layout_score_wire_weight": LAYOUT_SCORE_WEIGHTS['wire_length'],
        "layout_score_congestion_weight": LAYOUT_SCORE_WEIGHTS['congestion'],
        "component_repulsion_weight": 30.0, "group_pop_size": 300, "group_generations": GROUP_GENERATIONS,
        "group_mutation_rate": 0.2, "group_crossover_rate": 0.7, "group_tournament_size": 3,
        "group_routing_start_gen": int(GROUP_GENERATIONS * 0.4), "group_area_coef": 2, "top_n": 5,
        "diversity_threshold": 1.0, "out_pop_size": 300, "out_generations": OUT_GENERATIONS,
        "out_mutation_rate": 0.2, "out_crossover_rate": 0.7, "out_tournament_size": 3,
        "ema_alpha": 0.3, "softmax_t0": 5000.0, "softmax_tend": 1.0,
        "out_routing_start_gen": int(OUT_GENERATIONS * 0.2), "out_combine_start_gen": int(OUT_GENERATIONS * 0.4),
        "group_base_component_overlap_penalty": 6000, "group_repulsion_weight": 30.0,
        "diversity_std_threshold": 1e-4, "diversity_mutation_delta": 0.005, "diversity_crossover_delta": 0.2,
        "stage2_step_max_factor": 0.05, "stage2_step_min_factor": 0.0005,
        "out_full_reroute_interval": 25
    }

    if params_path and os.path.exists(params_path):
        print(f"[*] Loading custom parameters from: {params_path}")
        with open(params_path, 'r') as f:
            GLOBAL_PARAMS.update(json.load(f))

    print(f"[*] Input configuration: {config_path}")
    print(f"[*] Output directory: {base_dir}")
    print(f"[*] Genetic Algorithm Configuration:\n{json.dumps(GLOBAL_PARAMS, indent=2)}")

    start_time = time.time()
    config = load_config(config_path)
    config["PARAMS"] = GLOBAL_PARAMS

    print("[*] Starting optimization...")
    cxx_result = pcb_engine.run_ga_optimization(config)
    
    best_individuals = cxx_result["best_individuals"]
    local_groups_best = cxx_result.get("local_groups_best_individuals", [])
    fitness_history = cxx_result["fitness_history"]
    fitness_components_history = cxx_result["fitness_components_history"]
    group_fitness_histories = cxx_result.get("group_fitness_histories", [])
    layout_score_history = cxx_result.get("layout_score_history", [])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n[*] C++ calculation completed in {elapsed_time:.2f} seconds!")

    # save results
    history_csv = os.path.join(csv_dir, "fitness_history.csv")
    save_fitness_history(fitness_history, history_csv)
    
    components_csv = os.path.join(csv_dir, "fitness_components.csv")
    save_fitness_components(fitness_components_history, components_csv)

    num_layers = GLOBAL_PARAMS.get("num_layers", 2)
    best_json_files = []
    for rank, ind in enumerate(best_individuals):
        json_path = os.path.join(json_dir, f"best_{rank:02d}.json")
        save_individual_json(ind, rank, json_path, num_layers)
        best_json_files.append(json_path)
        
    scores_csv = os.path.join(csv_dir, "layout_scores.csv")
    save_layout_score_history(layout_score_history, scores_csv)
    print(f"[*] Layout score history saved to {scores_csv}")

    for group_idx, history in enumerate(group_fitness_histories):
        group_csv_path = os.path.join(group_csv_dir, f"group_fitness_{group_idx:02d}.csv")
        save_fitness_history(history, group_csv_path)
        plot_fitness(group_csv_path,
                     os.path.join(group_png_dir, f"group_fitness_{group_idx:02d}.png"),
                     title=f"Group {group_idx} Best Fitness")

    group_json_files = []
    for group_idx, ind in enumerate(local_groups_best):
        json_path = os.path.join(json_dir, f"group_best_{group_idx:02d}.json")
        save_individual_json(ind, group_idx, json_path, num_layers)
        group_json_files.append(json_path)

    print("\n[*] Generating plots...")
    plot_fitness(history_csv, os.path.join(png_dir, "fitness_history.png"))
    plot_fitness_components(components_csv, png_component_dir)
    plot_layout_score_history(scores_csv, os.path.join(png_dir, "layout_scores.png"))

    for rank, json_path in enumerate(best_json_files):
        plot_individual_layout(config, json_path, os.path.join(png_dir, f"best_{rank:02d}.png"), num_layers)
    for group_idx, json_path in enumerate(group_json_files):
        plot_individual_layout(config, json_path, os.path.join(png_dir, f"group_best_{group_idx:02d}.png"), num_layers)
    print("[*] All plots generated successfully.")

    run_summary = {
        "input_config": config_path, "output_directory": base_dir,
        "custom_params_file": params_path, "start_time": start_time,
        "end_time": end_time, "elapsed_seconds": elapsed_time,
        "GLOBAL_PARAMS": GLOBAL_PARAMS
    }
    with open(os.path.join(logs_dir, "run_summary.json"), 'w') as f:
        json.dump(run_summary, f, indent=4)
        
    print(f"\n[*] Optimization process completely finished!")
