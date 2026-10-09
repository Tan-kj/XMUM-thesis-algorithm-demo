import os
import json
import random
import math

# (Configurable Parameters)

CONFIG = {
    "num_jsons_to_generate": 20,
    "output_directory": "D:/Data_File/xmum 4.2 semester/Final_year_project_2/PCB_sample/level_7",  

    "complexity_limits": {
        "V_score": {"target": 16.8, "tolerance": 1.2},   
        "E_score": {"target": 4, "tolerance": 0.5},   
        "P_score": {"target": 8,  "tolerance": 1.5}    
    },
    "weights": {                 
        "alpha": 1.2,
        "beta": 0.5,
        "gamma": 2.0
    },

    "component_limits": {
        "C": (0, 8),  "D": (0, 8),  "R": (0, 8),  "J": (0, 4),
        "T": (0, 2),  "U": (0, 4),  "Q": (0, 4)
    },
    "max_pins_per_net": 9,
    "enable_grouping": True,
    "max_group_size": 7
}

COMPONENT_TEMPLATES = {
    "C": [
        {"width": 5.55, "height": 7.0965, "pins": [{"name": "1", "dx": -1.325, "dy": 0.0}, {"name": "2", "dx": 1.325, "dy": 0.0}]},
        {"width": 8.95, "height": 10.3965, "pins": [{"name": "1", "dx": -2.5375, "dy": 0.0}, {"name": "2", "dx": 2.5375, "dy": 0.0}]}
    ],
    "D": [
        {"width": 12.91, "height": 6.6365, "pins": [{"name": "1", "dx": -5.08, "dy": 0.0}, {"name": "2", "dx": 5.08, "dy": 0.0}]},
        {"width": 15.45, "height": 6.6365, "pins": [{"name": "1", "dx": -6.35, "dy": -0.0}, {"name": "2", "dx": 6.35, "dy": -0.0}]},
        {"width": 14.85, "height": 5.9365, "pins": [{"name": "1", "dx": -6.35, "dy": 0.0}, {"name": "2", "dx": 6.35, "dy": 0.0}]}
    ],
    "R": [
        {"width": 14.85, "height": 7.1365, "pins": [{"name": "1", "dx": -6.35, "dy": 0.0}, {"name": "2", "dx": 6.35, "dy": 0.0}]}
    ],
    "J": [
        {"width": 9.3452, "height": 8.9965, "pins": [{"name": "1", "dx": -0.0, "dy": -1.27}, {"name": "2", "dx": -0.0, "dy": 1.27}]},
        {"width": 10.55, "height": 7.2565, "pins": [{"name": "1", "dx": -3.0, "dy": -0.0}, {"name": "2", "dx": 3.0, "dy": -0.0}]}
    ],
    "T": [
        {"width": 14.5357, "height": 12.3965, "pins": [{"name": "1", "dx": -5.62, "dy": -2.39}, {"name": "2", "dx": -5.62, "dy": 0.15}, {"name": "3", "dx": -5.62, "dy": 2.69}, {"name": "4", "dx": 5.62, "dy": 2.69}, {"name": "5", "dx": 5.62, "dy": 0.15}, {"name": "6", "dx": 5.62, "dy": -2.39}]}
    ],
    "U": [
        {"width": 7.45, "height": 8.4965, "pins": [{"name": "1", "dx": -2.475, "dy": -1.905}, {"name": "2", "dx": -2.475, "dy": -0.635}, {"name": "3", "dx": -2.475, "dy": 0.635}, {"name": "4", "dx": -2.475, "dy": 1.905}, {"name": "5", "dx": 2.475, "dy": 1.905}, {"name": "6", "dx": 2.475, "dy": 0.635}, {"name": "7", "dx": 2.475, "dy": -0.635}, {"name": "8", "dx": 2.475, "dy": -1.905}]},
        {"width": 9.78, "height": 8.8965, "pins": [{"name": "1", "dx": -3.805, "dy": -1.27}, {"name": "2", "dx": -3.805, "dy": 1.27}, {"name": "3", "dx": 3.815, "dy": 1.27}, {"name": "4", "dx": 3.815, "dy": -1.27}]}
    ],
    "Q": [
        {"width": 5.51, "height": 8.0465, "pins": [{"name": "1", "dx": -1.27, "dy": 0.385}, {"name": "2", "dx": 0.0, "dy": 0.385}, {"name": "3", "dx": 1.27, "dy": 0.385}]},
        {"width": 16.45, "height": 8.6265, "pins": [{"name": "1", "dx": -5.45, "dy": -0.185}, {"name": "2", "dx": 0.0, "dy": -0.185}, {"name": "3", "dx": 5.45, "dy": -0.185}]}
    ]
}

def auto_extract_groups(nets, footprint_refs, max_group_size):
    edges = {ref: {} for ref in footprint_refs}
    total_comps = len(footprint_refs)

    for net in nets:
        conns = net["connections"]
        comps_in_net = list(set(conn["comp_id"] for conn in conns))
        n = len(comps_in_net)
        if n < 2 or n > max(15, total_comps * 0.2): continue
        weight = 1.0 / (n - 1)
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = comps_in_net[i], comps_in_net[j]
                edges[c1][c2] = edges[c1].get(c2, 0) + weight
                edges[c2][c1] = edges[c2].get(c1, 0) + weight

    groups = {ref: [ref] for ref in footprint_refs}
    group_id_map = {ref: ref for ref in footprint_refs}

    edge_list = []
    for c1, neighbors in edges.items():
        for c2, w in neighbors.items():
            if c1 < c2: edge_list.append((w, c1, c2))
    edge_list.sort(key=lambda x: x[0], reverse=True)

    for w, c1, c2 in edge_list:
        g1, g2 = group_id_map[c1], group_id_map[c2]
        if g1 != g2 and (len(groups[g1]) + len(groups[g2]) <= max_group_size):
            for member in groups[g2]:
                groups[g1].append(member)
                group_id_map[member] = g1
            del groups[g2]

    out_groups = []
    g_idx = 1
    for members in groups.values():
        if len(members) >= 2:
            out_groups.append({"group_name": f"AutoGroup_{g_idx}", "members": members})
            g_idx += 1

    single_comps = [members[0] for members in groups.values() if len(members) < 2]
    if single_comps:
        group_lookup = {}
        for group in out_groups:
            for member in group["members"]:
                group_lookup[member] = group
        for comp in single_comps:
            if comp in group_lookup:
                continue
            neighbors = edges.get(comp, {})
            if not neighbors:
                continue
            best_neighbor = max(neighbors, key=neighbors.get)
            target = group_lookup.get(best_neighbor)
            if target is None:
                continue
            target["members"].append(comp)
            group_lookup[comp] = target

    return out_groups

def _select_active_pins(all_pins, P_total):
    random.shuffle(all_pins)
    if P_total >= len(all_pins):
        return all_pins

    pins_by_comp = {}
    for pin in all_pins:
        pins_by_comp.setdefault(pin["comp_id"], []).append(pin)

    selected = []
    for pins in pins_by_comp.values():
        selected.append(pins.pop(random.randrange(len(pins))))
    if len(selected) > P_total:
        random.shuffle(selected)
        return selected[:P_total]

    extra = [pin for pins in pins_by_comp.values() for pin in pins]
    random.shuffle(extra)
    selected.extend(extra[:P_total - len(selected)])
    random.shuffle(selected)
    return selected

def _repair_nets(nets):
    def net_comps(net):
        return list(dict.fromkeys(c["comp_id"] for c in net["connections"]))

    for _ in range(len(nets)):
        self_nets = [n for n in nets if len(net_comps(n)) < 2]
        if not self_nets:
            break
        multi_nets = [n for n in nets if len(net_comps(n)) >= 2]

        fixed = False
        for s_net in self_nets:
            s_comp = net_comps(s_net)[0]
            s_pin = s_net["connections"][0]
            s_conn = s_net["connections"]

            for m_net in multi_nets:
                m_conn = m_net["connections"]
                for m_pin in m_conn:
                    if m_pin["comp_id"] == s_comp:
                        continue
                    s_conn.remove(s_pin)
                    m_conn.remove(m_pin)
                    m_conn.insert(0, s_pin)
                    s_conn.insert(0, m_pin)
                    if len(net_comps(m_net)) >= 2:
                        fixed = True
                        break
                    m_conn.remove(s_pin)
                    s_conn.remove(m_pin)
                    m_conn.insert(0, m_pin)
                    s_conn.insert(0, s_pin)
                if fixed:
                    break
            if fixed:
                break

            for m_net in self_nets:
                if m_net is s_net or net_comps(m_net)[0] == s_comp:
                    continue
                m_conn = m_net["connections"]
                m_pin = m_conn[0]
                s_conn.remove(s_pin)
                m_conn.remove(m_pin)
                m_conn.insert(0, s_pin)
                s_conn.insert(0, m_pin)
                fixed = True
                break
            if fixed:
                break

        if not fixed:
            break
    return nets

def _connect_islands(nets, footprint_refs):
    max_iters = 50
    for _ in range(max_iters):
        adj = {ref: set() for ref in footprint_refs}
        for net in nets:
            comps = list(dict.fromkeys(c["comp_id"] for c in net["connections"]))
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    adj[comps[i]].add(comps[j])
                    adj[comps[j]].add(comps[i])

        visited = set()
        islands = []
        for ref in footprint_refs:
            if ref not in visited:
                island = set()
                queue = [ref]
                visited.add(ref)
                while queue:
                    curr = queue.pop(0)
                    island.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                islands.append(island)

        active_islands = []
        for island in islands:
            island_nets = []
            for i, net in enumerate(nets):
                if any(c["comp_id"] in island for c in net["connections"]):
                    island_nets.append(i)
            if island_nets:
                active_islands.append((island, island_nets))

        if len(active_islands) <= 1:
            break

        comp_pin_counts = {}
        for net in nets:
            for c in net["connections"]:
                comp_id = c["comp_id"]
                comp_pin_counts[comp_id] = comp_pin_counts.get(comp_id, 0) + 1

        merged = False
        for i in range(len(active_islands)):
            for j in range(i + 1, len(active_islands)):
                island1, nets1 = active_islands[i]
                island2, nets2 = active_islands[j]

                bridge_comp = None
                bridge_net_idx = -1
                bridge_pin_idx = -1

                for comp in island1:
                    if comp_pin_counts.get(comp, 0) > 1:
                        for n_idx in nets1:
                            for p_idx, conn in enumerate(nets[n_idx]["connections"]):
                                if conn["comp_id"] == comp:
                                    bridge_comp = comp
                                    bridge_net_idx = n_idx
                                    bridge_pin_idx = p_idx
                                    break
                            if bridge_comp: break
                    if bridge_comp: break

                target_island_nets = nets2
                target_island = island2

                if not bridge_comp:
                    for comp in island2:
                        if comp_pin_counts.get(comp, 0) > 1:
                            for n_idx in nets2:
                                for p_idx, conn in enumerate(nets[n_idx]["connections"]):
                                    if conn["comp_id"] == comp:
                                        bridge_comp = comp
                                        bridge_net_idx = n_idx
                                        bridge_pin_idx = p_idx
                                        break
                                if bridge_comp: break
                        if bridge_comp: break
                    target_island_nets = nets1
                    target_island = island1

                if bridge_comp:
                    target_net_idx = target_island_nets[0]
                    target_pin_idx = -1
                    for p_idx, conn in enumerate(nets[target_net_idx]["connections"]):
                        if conn["comp_id"] in target_island:
                            target_pin_idx = p_idx
                            break

                    if target_pin_idx != -1:
                        pin1 = nets[bridge_net_idx]["connections"].pop(bridge_pin_idx)
                        pin2 = nets[target_net_idx]["connections"].pop(target_pin_idx)
                        nets[bridge_net_idx]["connections"].append(pin2)
                        nets[target_net_idx]["connections"].append(pin1)
                        merged = True
                        break
            if merged:
                break
        if not merged:
            break

    return nets


def find_feasible_configuration():
    w = CONFIG["weights"]
    limits = CONFIG["complexity_limits"]

    alpha, beta, gamma = w["alpha"], w["beta"], w["gamma"]
    V_target, V_tol = limits["V_score"]["target"], limits["V_score"]["tolerance"]
    E_target, E_tol = limits["E_score"]["target"], limits["E_score"]["tolerance"]
    P_target, P_tol = limits["P_score"]["target"], limits["P_score"]["tolerance"]

    for _ in range(50000):
        comp_counts = {p: random.randint(c[0], c[1]) for p, c in CONFIG["component_limits"].items()}

        selected_comps = []
        total_pins = 0
        for prefix, count in comp_counts.items():
            for _ in range(count):
                template = random.choice(COMPONENT_TEMPLATES[prefix])
                selected_comps.append({"prefix": prefix, "template": template})
                total_pins += len(template["pins"])

        V = len(selected_comps)
        V_score = alpha * V

        if abs(V_score - V_target) > V_tol:
            continue

        max_E = total_pins // 2
        if max_E < 1:
            continue

        E_min = math.ceil((E_target - E_tol) / beta)
        E_max = math.floor((E_target + E_tol) / beta)
        E_min = max(1, E_min)
        E_max = min(max_E, E_max)

        if E_min > E_max:
            continue

        valid_options = []
        for E in range(E_min, E_max + 1):
            P_low = math.ceil((P_target - P_tol) * E / gamma)
            P_high = math.floor((P_target + P_tol) * E / gamma)

            P_low = max(P_low, 2 * E)
            P_high = min(P_high, total_pins, E * CONFIG["max_pins_per_net"])

            if P_low <= P_high:
                for P in range(P_low, P_high + 1):
                    valid_options.append((E, P))

        if valid_options:
            E, P = random.choice(valid_options)
            actual_scores = {
                "V_score": alpha * V,
                "E_score": beta * E,
                "P_score": gamma * (P / E)
            }
            return selected_comps, E, P, actual_scores

    raise RuntimeError(
        f"Cannot find feasible configuration after 50000 attempts. "
    )

def construct_circuit(index):
    result = find_feasible_configuration()
    if result is None:
        raise RuntimeError("build_circuit: No feasible configuration found.")
    selected_comps, E, P_total, scores = result

    components = []
    all_pins = []
    footprint_refs = set()

    prefix_counters = {p: 1 for p in CONFIG["component_limits"].keys()}
    for comp in selected_comps:
        prefix = comp["prefix"]
        comp_id = f"{prefix}{prefix_counters[prefix]}"
        prefix_counters[prefix] += 1

        template = comp["template"]
        components.append({
            "id": comp_id, "width": template["width"],
            "height": template["height"], "pins": template["pins"]
        })
        footprint_refs.add(comp_id)
        for pin in template["pins"]:
            all_pins.append({"comp_id": comp_id, "pin_name": pin["name"]})

    active_pins = _select_active_pins(all_pins, P_total)

    net_sizes = [2] * E
    remaining_pins = P_total - 2 * E
    while remaining_pins > 0:
        idx = random.randint(0, E - 1)
        if net_sizes[idx] < CONFIG["max_pins_per_net"]:
            net_sizes[idx] += 1
            remaining_pins -= 1

    nets = []
    pin_index = 0
    for i, size in enumerate(net_sizes):
        connections = active_pins[pin_index: pin_index + size]
        pin_index += size
        nets.append({
            "net_id": f"Net-Construct_{i+1}",
            "connections": connections
        })

    _repair_nets(nets)
    
    _connect_islands(nets, footprint_refs)

    groups = []
    if CONFIG["enable_grouping"]:
        groups = auto_extract_groups(nets, footprint_refs, CONFIG["max_group_size"])

    metadata = {
        "V_score": round(scores["V_score"], 3),
        "E_score": round(scores["E_score"], 3),
        "P_score": round(scores["P_score"], 3),
        "V_count": len(components),
        "E_count": E,
        "P_count": P_total,
        "V_target": CONFIG["complexity_limits"]["V_score"]["target"],
        "E_target": CONFIG["complexity_limits"]["E_score"]["target"],
        "P_target": CONFIG["complexity_limits"]["P_score"]["target"]
    }

    return {
        "metadata": metadata,
        "pcb_bounds": {"xmin": 0.0, "ymin": 0.0, "xmax": 300.0, "ymax": 300.0},
        "components": components,
        "groups": groups,
        "nets": nets
    }

def main():
    os.makedirs(CONFIG["output_directory"], exist_ok=True)

    limits = CONFIG["complexity_limits"]
    print("start generating PCB JSON files with the following complexity limits:")
    for key, val in limits.items():
        print(f"  {key}: target={val['target']} ± {val['tolerance']}")

    for i in range(1, CONFIG["num_jsons_to_generate"] + 1):
        try:
            layout_data = construct_circuit(i)
            filename = f"layout_{i:02d}.json"
            filepath = os.path.join(CONFIG["output_directory"], filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(layout_data, f, indent=2)

            meta = layout_data["metadata"]
            print(
                f"[{i:02d}/{CONFIG['num_jsons_to_generate']}] generated {filename} | "
                f"V_score={meta['V_score']} (V={meta['V_count']}), "
                f"E_score={meta['E_score']} (E={meta['E_count']}), "
                f"P_score={meta['P_score']} (P={meta['P_count']})"
            )
        except Exception as e:
            print(f"Failed to generate {filename}: {e}")
            break

if __name__ == "__main__":
    main()