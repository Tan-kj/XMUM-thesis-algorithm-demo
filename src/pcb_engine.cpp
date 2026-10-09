#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <queue>
#include <random>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <limits>
#include <iostream>
#include <utility>
#include <set>
#include <atomic>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;


struct Point2D { double x, y; };

struct Point3D { double x, y; int l; };

struct Segment3D { Point3D p1, p2; };

struct PinDef { std::string name; double dx, dy; };

struct ComponentDef {
    std::string id;
    double width, height;
    double min_x, max_x, min_y, max_y;
    std::vector<PinDef> pins;
};

struct NetConnection { int comp_idx; std::string pin_name; };

struct NetDef { std::string net_id; std::vector<NetConnection> connections; };

struct CompGene { double x, y; int angle; };


struct Individual {
    std::vector<CompGene> genes;
    double fitness = std::numeric_limits<double>::max();
    double total_wire_length = 0;
    double total_cap = 0;
    double overlap_score = 0;
    int wire_cross_count = 0;
    double wire_overlap_len = 0;
    double total_pin_dist = 0;
    double repulsion_penalty = 0;
    double centripetal_dist = 0;
    double eval_overlap_penalty = 0;
    double eval_repulsion_weight = 0;
    std::vector<std::vector<Segment3D>> net_segments;
    std::vector<std::unordered_map<std::string, Point2D>> abs_pins;
};

struct Context {
    std::vector<ComponentDef> comps;
    std::vector<NetDef> nets;
    std::unordered_map<std::string, int> comp_name_to_idx;

    int pop_size, generations, tournament_size;
    double mutation_rate, crossover_rate, grid_res;
    double wirelength_weight, capacitance_weight;
    double base_component_overlap_penalty, base_wire_cross_penalty;
    double wire_comp_overlap_penalty, via_penalty, distance_weight;
    double dynamic_cross_penalty = 0;
    double current_distance_weight = 0;

    int routing_start_gen = 0;
    double component_repulsion_weight = 0.0;
    double repulsion_threshold = 0.0;
    int num_layers = 2;

    std::vector<NetDef> single_pin_nets;
    std::vector<Point2D> single_pin_ghosts;
    double centripetal_weight = 0.0;
};

struct GroupDef {
    std::string name;
    std::vector<int> comp_indices;
    std::set<int> comp_set;
    double total_area = 0;
};

struct GroupSolution {
    std::vector<CompGene> local_genes;
    std::unordered_map<int, std::vector<Segment3D>> local_net_segments;
    std::vector<std::unordered_map<std::string, Point2D>> local_abs_pins;
    std::vector<int> comp_global_indices;
    double fitness;
};

struct GroupGAResult {
    std::vector<GroupSolution> solutions;
    std::vector<double> fitness_history;
};

Point2D rotate_point(double x, double y, int angle_deg) {
    double rad = angle_deg * M_PI / 180.0;
    double c = std::cos(rad), s = std::sin(rad);
    return { x * c - y * s, x * s + y * c };
}

inline double capped_cross_penalty(double base_wire_cross_penalty, double progress) {
    return std::min(3.0 * base_wire_cross_penalty,
                    base_wire_cross_penalty * std::pow(10.0, progress));
}

struct AABB { double xmin, ymin, xmax, ymax; };

AABB get_component_aabb(double cx, double cy, double w, double h, int angle) {
    Point2D p1 = rotate_point(-w / 2, -h / 2, angle);
    Point2D p2 = rotate_point(w / 2, -h / 2, angle);
    Point2D p3 = rotate_point(w / 2, h / 2, angle);
    Point2D p4 = rotate_point(-w / 2, h / 2, angle);
    double xmin = cx + std::min({ p1.x, p2.x, p3.x, p4.x });
    double xmax = cx + std::max({ p1.x, p2.x, p3.x, p4.x });
    double ymin = cy + std::min({ p1.y, p2.y, p3.y, p4.y });
    double ymax = cy + std::max({ p1.y, p2.y, p3.y, p4.y });
    return { xmin, ymin, xmax, ymax };
}

double cross_product(Point2D a, Point2D b, Point2D c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool is_intersecting(Point2D p1, Point2D p2, Point2D p3, Point2D p4) {
    double cp1 = cross_product(p3, p4, p1);
    double cp2 = cross_product(p3, p4, p2);
    double cp3 = cross_product(p1, p2, p3);
    double cp4 = cross_product(p1, p2, p4);
    return (cp1 * cp2 < 0) && (cp3 * cp4 < 0);
}

std::vector<std::pair<int, int>> mst(const std::vector<Point2D>& points) {
    int n = static_cast<int>(points.size());
    if (n <= 1) return {};
    std::vector<bool> visited(n, false);
    std::vector<double> min_edge(n, std::numeric_limits<double>::infinity());
    std::vector<int> parent(n, -1);
    min_edge[0] = 0.0;
    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i < n; ++i) {
        int u = -1;
        for (int j = 0; j < n; ++j)
            if (!visited[j] && (u == -1 || min_edge[j] < min_edge[u])) u = j;
        if (u == -1) break;
        visited[u] = true;
        if (parent[u] != -1) edges.push_back({ parent[u], u });
        for (int v = 0; v < n; ++v) {
            if (!visited[v]) {
                double dist = std::hypot(points[u].x - points[v].x, points[u].y - points[v].y);
                if (dist < min_edge[v]) { min_edge[v] = dist; parent[v] = u; }
            }
        }
    }
    return edges;
}

struct AStarNode { int c, r, l; double f; bool operator>(const AStarNode& o) const { return f > o.f; } };

std::vector<Point3D> a_star_3d(
    Point3D start, Point3D goal, const std::vector<AABB>& comp_boxes,
    std::vector<int>& grid, const std::vector<double>& history_cost,
    int cols, int rows, double min_x, double min_y,
    const Context& ctx, int net_id)
{
    int start_c = static_cast<int>(std::round((start.x - min_x) / ctx.grid_res));
    int start_r = static_cast<int>(std::round((start.y - min_y) / ctx.grid_res));
    int goal_c  = static_cast<int>(std::round((goal.x  - min_x) / ctx.grid_res));
    int goal_r  = static_cast<int>(std::round((goal.y  - min_y) / ctx.grid_res));
    int n_layers = ctx.num_layers;
    auto get_idx = [&](int c, int r, int l) { return l * (cols * rows) + r * cols + c; };

    if (start_c < 0 || start_c >= cols || start_r < 0 || start_r >= rows)
        return { start, {start.x + 20000, start.y + 20000, start.l}, goal };

    std::vector<double> g_score(cols * rows * n_layers, std::numeric_limits<double>::infinity());
    std::vector<int> came_from(cols * rows * n_layers, -1);
    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> pq;

    int start_idx = get_idx(start_c, start_r, start.l);
    g_score[start_idx] = 0.0;
    pq.push({ start_c, start_r, start.l, 0.0 });

    int dc[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
    int dr[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };
    bool found = false;
    int iters = 0;
    double dist_grid = (std::abs(goal.x - start.x) + std::abs(goal.y - start.y)) / ctx.grid_res;
    int max_iters = std::min(50000, static_cast<int>(5000 + dist_grid * 60));

    while (!pq.empty() && iters < max_iters) {
        iters++;
        auto curr = pq.top(); pq.pop();
        int curr_idx = get_idx(curr.c, curr.r, curr.l);
        if (curr.c == goal_c && curr.r == goal_r && curr.l == goal.l) { found = true; break; }

        for (int d = 0; d < 8; ++d) {
            int nc = curr.c + dc[d], nr = curr.r + dr[d], nl = curr.l;
            if (nc < 0 || nc >= cols || nr < 0 || nr >= rows) continue;
            double step_cost = (d < 4) ? ctx.grid_res : 1.41421356 * ctx.grid_res;
            if (d < 4) {
                if (nl == 0 && dc[d] == 0) step_cost += ctx.grid_res * 1.2;
                if (nl == 1 && dr[d] == 0) step_cost += ctx.grid_res * 1.2;
            } else step_cost += ctx.grid_res * 1.5;

            double nx = min_x + nc * ctx.grid_res, ny = min_y + nr * ctx.grid_res;
            if (nl == 0 && std::abs(nx - start.x) + std::abs(ny - start.y) > 10 &&
                std::abs(nx - goal.x) + std::abs(ny - goal.y) > 10) {
                for (const auto& box : comp_boxes) {
                    if (nx >= box.xmin && nx <= box.xmax && ny >= box.ymin && ny <= box.ymax)
                        { step_cost += ctx.wire_comp_overlap_penalty; break; }
                }
            }

            int n_idx = get_idx(nc, nr, nl);
            int occ = grid[n_idx];
            if (occ != -1 && occ != net_id) step_cost += ctx.dynamic_cross_penalty;
            step_cost += history_cost[n_idx];

            double tentative_g = g_score[curr_idx] + step_cost;
            if (tentative_g < g_score[n_idx]) {
                came_from[n_idx] = curr_idx;
                g_score[n_idx] = tentative_g;
                double dx = std::abs(nx - goal.x), dy = std::abs(ny - goal.y);
                double h = (dx + dy + (1.414 - 2.0) * std::min(dx, dy) + std::abs(nl - goal.l) * ctx.via_penalty) * 1.5;
                pq.push({ nc, nr, nl, tentative_g + h });
            }
        }
        for (int delta_l = -1; delta_l <= 1; delta_l += 2) {
            int nl = curr.l + delta_l;
            if (nl < 0 || nl >= n_layers) continue;
            double step_cost = ctx.via_penalty;
            int nc = curr.c, nr = curr.r;
            double nx = min_x + nc * ctx.grid_res, ny = min_y + nr * ctx.grid_res;
            if (nl == 0 && std::abs(nx - start.x) + std::abs(ny - start.y) > 10 &&
                std::abs(nx - goal.x) + std::abs(ny - goal.y) > 10) {
                for (const auto& box : comp_boxes) {
                    if (nx >= box.xmin && nx <= box.xmax && ny >= box.ymin && ny <= box.ymax)
                        { step_cost += ctx.wire_comp_overlap_penalty; break; }
                }
            }
            int n_idx = get_idx(nc, nr, nl);
            int occ = grid[n_idx];
            if (occ != -1 && occ != net_id) step_cost += ctx.dynamic_cross_penalty;
            step_cost += history_cost[n_idx];
            double tentative_g = g_score[curr_idx] + step_cost;
            if (tentative_g < g_score[n_idx]) {
                came_from[n_idx] = curr_idx;
                g_score[n_idx] = tentative_g;
                double dx = std::abs(nx - goal.x), dy = std::abs(ny - goal.y);
                double h = (dx + dy + (1.414 - 2.0) * std::min(dx, dy) + std::abs(nl - goal.l) * ctx.via_penalty) * 1.5;
                pq.push({ nc, nr, nl, tentative_g + h });
            }
        }
    }

    if (!found && came_from[get_idx(goal_c, goal_r, goal.l)] == -1)
        return { start, {start.x + 20000, start.y + 20000, start.l}, goal };

    std::vector<Point3D> path;
    int curr_idx = get_idx(goal_c, goal_r, goal.l);
    while (curr_idx != -1) {
        int l = curr_idx / (cols * rows);
        int rem = curr_idx % (cols * rows);
        int r = rem / cols;
        int c = rem % cols;
        path.push_back({ min_x + c * ctx.grid_res, min_y + r * ctx.grid_res, l });
        curr_idx = came_from[curr_idx];
    }
    std::reverse(path.begin(), path.end());
    path[0] = start;
    path.back() = goal;
    return path;
}

void rasterize_line(std::vector<int>& grid, int cols, int rows, double min_x, double min_y, double res,
    double x0, double y0, double x1, double y1, int layer, int val) {
    int c0 = static_cast<int>(std::round((x0 - min_x) / res));
    int r0 = static_cast<int>(std::round((y0 - min_y) / res));
    int c1 = static_cast<int>(std::round((x1 - min_x) / res));
    int r1 = static_cast<int>(std::round((y1 - min_y) / res));
    int dc = std::abs(c1 - c0), dr = -std::abs(r1 - r0);
    int sc = c0 < c1 ? 1 : -1, sr = r0 < r1 ? 1 : -1;
    int err = dc + dr;
    while (true) {
        if (c0 >= 0 && c0 < cols && r0 >= 0 && r0 < rows)
            grid[layer * (cols * rows) + r0 * cols + c0] = val;
        if (c0 == c1 && r0 == r1) break;
        int e2 = 2 * err;
        if (e2 >= dr) { err += dr; c0 += sc; }
        if (e2 <= dc) { err += dc; r0 += sr; }
    }
}

void manage_grid(std::vector<int>& grid, int cols, int rows, double min_x, double min_y, double res,
    const std::vector<Segment3D>& segs, int net_id, bool add) {
    int val = add ? net_id : -1;
    for (const auto& seg : segs) {
        if (seg.p1.l == seg.p2.l)
            rasterize_line(grid, cols, rows, min_x, min_y, res, seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y, seg.p1.l, val);
        else {
            int c = static_cast<int>(std::round((seg.p1.x - min_x) / res));
            int r = static_cast<int>(std::round((seg.p1.y - min_y) / res));
            if (c >= 0 && c < cols && r >= 0 && r < rows) {
                grid[0 * (cols * rows) + r * cols + c] = val;
                grid[1 * (cols * rows) + r * cols + c] = val;
            }
        }
    }
}

void evaluate(Individual& ind, const Context& ctx, bool do_routing, unsigned eval_seed = 0) {
    int num_comps = static_cast<int>(ctx.comps.size());
    int num_nets  = static_cast<int>(ctx.nets.size());

    double min_x = std::numeric_limits<double>::infinity(), max_x = -min_x;
    double min_y = min_x, max_y = -min_x;
    std::vector<AABB> comp_boxes(num_comps);
    ind.abs_pins.resize(num_comps);

    for (int i = 0; i < num_comps; ++i) {
        const auto& c = ctx.comps[i];
        const auto& gene = ind.genes[i];
        comp_boxes[i] = get_component_aabb(gene.x, gene.y, c.width, c.height, gene.angle);
        min_x = std::min(min_x, comp_boxes[i].xmin); max_x = std::max(max_x, comp_boxes[i].xmax);
        min_y = std::min(min_y, comp_boxes[i].ymin); max_y = std::max(max_y, comp_boxes[i].ymax);
        for (const auto& pin : c.pins) {
            ind.abs_pins[i][pin.name] = rotate_point(pin.dx, pin.dy, gene.angle);
            ind.abs_pins[i][pin.name].x += gene.x;
            ind.abs_pins[i][pin.name].y += gene.y;
        }
    }

    double overlap_score = 0.0, repulsion_penalty = 0.0;
    for (int i = 0; i < num_comps; ++i) {
        for (int j = i + 1; j < num_comps; ++j) {
            double dx = std::max(0.0, std::min(comp_boxes[i].xmax, comp_boxes[j].xmax) - std::max(comp_boxes[i].xmin, comp_boxes[j].xmin));
            double dy = std::max(0.0, std::min(comp_boxes[i].ymax, comp_boxes[j].ymax) - std::max(comp_boxes[i].ymin, comp_boxes[j].ymin));
            if (dx > 0 && dy > 0) {
                double inter_area = dx * dy;
                double act_d = std::hypot(ind.genes[i].x - ind.genes[j].x, ind.genes[i].y - ind.genes[j].y);
                double safe_d = std::hypot(ctx.comps[i].width / 2, ctx.comps[i].height / 2) + std::hypot(ctx.comps[j].width / 2, ctx.comps[j].height / 2);
                overlap_score += (std::max(0.1, safe_d - act_d) * 10 + inter_area) * ctx.base_component_overlap_penalty;
            } else if (ctx.component_repulsion_weight > 0.0) {
                double D = std::hypot(ind.genes[i].x - ind.genes[j].x, ind.genes[i].y - ind.genes[j].y);
                if (D < ctx.repulsion_threshold)
                    repulsion_penalty += (ctx.repulsion_threshold - D) / ctx.repulsion_threshold;
            }
        }
    }
    repulsion_penalty *= ctx.component_repulsion_weight;
    ind.overlap_score = overlap_score;

    double total_pin_dist = 0.0;
    for (int i = 0; i < num_nets; ++i) {
        std::vector<Point2D> pts;
        for (const auto& conn : ctx.nets[i].connections)
            pts.push_back(ind.abs_pins[conn.comp_idx][conn.pin_name]);
        auto edges = mst(pts);
        for (auto e : edges)
            total_pin_dist += std::hypot(pts[e.first].x - pts[e.second].x, pts[e.first].y - pts[e.second].y);
    }
    ind.total_pin_dist = total_pin_dist;
    ind.repulsion_penalty = repulsion_penalty;

    double centripetal_dist = 0.0;
    for (size_t sni = 0; sni < ctx.single_pin_nets.size(); ++sni) {
        const auto& net = ctx.single_pin_nets[sni];
        Point2D ghost = (sni < ctx.single_pin_ghosts.size()) ? ctx.single_pin_ghosts[sni]
                                                             : Point2D{ 0.0, 0.0 };
        for (const auto& conn : net.connections) {
            const auto& p = ind.abs_pins[conn.comp_idx][conn.pin_name];
            centripetal_dist += std::hypot(p.x - ghost.x, p.y - ghost.y);
        }
    }
    ind.centripetal_dist = centripetal_dist;

    if (!do_routing) {
        ind.total_wire_length = 0;
        ind.total_cap = 0;
        ind.wire_cross_count = 0;
        ind.wire_overlap_len = 0;
        ind.net_segments.assign(num_nets, {});
        ind.fitness = overlap_score + total_pin_dist + repulsion_penalty +
                      centripetal_dist * ctx.centripetal_weight;
        return;
    }

    min_x -= 30.0; max_x += 30.0; min_y -= 30.0; max_y += 30.0;
    int cols = static_cast<int>((max_x - min_x) / ctx.grid_res + 1);
    int rows = static_cast<int>((max_y - min_y) / ctx.grid_res + 1);
    int n_layers = ctx.num_layers;

    std::vector<int> occupancy_grid(cols * rows * n_layers, -1);
    std::vector<double> history_cost(cols * rows * n_layers, 0.0);
    std::vector<int> unrouted_nets(num_nets);
    std::iota(unrouted_nets.begin(), unrouted_nets.end(), 0);
    std::vector<double> net_estimated_len(num_nets, 0.0);
    for (int i = 0; i < num_nets; ++i) {
        std::vector<Point2D> pts;
        for (const auto& conn : ctx.nets[i].connections) pts.push_back(ind.abs_pins[conn.comp_idx][conn.pin_name]);
        auto edges = mst(pts);
        for (auto e : edges) net_estimated_len[i] += std::hypot(pts[e.first].x - pts[e.second].x, pts[e.first].y - pts[e.second].y);
    }
    std::sort(unrouted_nets.begin(), unrouted_nets.end(), [&](int a, int b) {
        return net_estimated_len[a] < net_estimated_len[b]; });
    std::vector<int> rip_up_counts(num_nets, 0);
    ind.net_segments.assign(num_nets, {});
    std::mt19937 local_rng(eval_seed ^ 0xDEADBEEF);

    while (!unrouted_nets.empty()) {
        int net_idx = unrouted_nets.front();
        unrouted_nets.erase(unrouted_nets.begin());
        std::vector<Point2D> pins_list;
        for (const auto& conn : ctx.nets[net_idx].connections)
            pins_list.push_back(ind.abs_pins[conn.comp_idx][conn.pin_name]);
        auto edges = mst(pins_list);
        std::vector<Segment3D> segments;
        for (auto e : edges) {
            Point3D p1 = { pins_list[e.first].x,  pins_list[e.first].y,  0 };
            Point3D p2 = { pins_list[e.second].x, pins_list[e.second].y, 0 };
            auto path = a_star_3d(p1, p2, comp_boxes, occupancy_grid, history_cost, cols, rows, min_x, min_y, ctx, net_idx);
            for (size_t k = 1; k < path.size(); ++k)
                segments.push_back({ path[k - 1], path[k] });
        }

        std::vector<int> crossed_nets;
        for (const auto& seg : segments) {
            if (seg.p1.l == seg.p2.l) {
                int c0 = static_cast<int>(std::round((seg.p1.x - min_x) / ctx.grid_res));
                int r0 = static_cast<int>(std::round((seg.p1.y - min_y) / ctx.grid_res));
                int c1 = static_cast<int>(std::round((seg.p2.x - min_x) / ctx.grid_res));
                int r1 = static_cast<int>(std::round((seg.p2.y - min_y) / ctx.grid_res));
                int sc = c0 < c1 ? 1 : -1, sr = r0 < r1 ? 1 : -1;
                int dc = std::abs(c1 - c0), dr = -std::abs(r1 - r0);
                int err = dc + dr;
                while (true) {
                    if (c0 >= 0 && c0 < cols && r0 >= 0 && r0 < rows) {
                        int cell_idx = seg.p1.l * (cols * rows) + r0 * cols + c0;
                        int occ = occupancy_grid[cell_idx];
                        if (occ != -1 && occ != net_idx) {
                            crossed_nets.push_back(occ);
                            history_cost[cell_idx] += 3.0;
                        }
                    }
                    if (c0 == c1 && r0 == r1) break;
                    int e2 = 2 * err;
                    if (e2 >= dr) { err += dr; c0 += sc; }
                    if (e2 <= dc) { err += dc; r0 += sr; }
                }
            }
        }
        std::sort(crossed_nets.begin(), crossed_nets.end());
        crossed_nets.erase(std::unique(crossed_nets.begin(), crossed_nets.end()), crossed_nets.end());
        std::shuffle(crossed_nets.begin(), crossed_nets.end(), local_rng);
        for (int cint : crossed_nets) {
            if (rip_up_counts[cint] < 15) {
                manage_grid(occupancy_grid, cols, rows, min_x, min_y, ctx.grid_res, ind.net_segments[cint], cint, false);
                ind.net_segments[cint].clear();
                unrouted_nets.push_back(cint);
                rip_up_counts[cint]++;
            }
        }
        manage_grid(occupancy_grid, cols, rows, min_x, min_y, ctx.grid_res, segments, net_idx, true);
        ind.net_segments[net_idx] = segments;
    }

    ind.total_wire_length = 0;
    for (const auto& segs : ind.net_segments)
        for (const auto& seg : segs)
            if (seg.p1.l == seg.p2.l) ind.total_wire_length += std::hypot(seg.p1.x - seg.p2.x, seg.p1.y - seg.p2.y);
            else ind.total_wire_length += ctx.via_penalty;

    ind.wire_cross_count = 0;
    double wire_comp_overlap_len = 0;
    for (int i = 0; i < num_nets; ++i) {
        for (const auto& seg : ind.net_segments[i]) {
            if (seg.p1.l != seg.p2.l) continue;
            if (seg.p1.l == 0) {
                for (const auto& box : comp_boxes) {
                    if (std::max(seg.p1.x, seg.p2.x) > box.xmin && std::min(seg.p1.x, seg.p2.x) < box.xmax &&
                        std::max(seg.p1.y, seg.p2.y) > box.ymin && std::min(seg.p1.y, seg.p2.y) < box.ymax)
                        wire_comp_overlap_len += ctx.grid_res;
                }
            }
            for (int j = i + 1; j < num_nets; ++j) {
                for (const auto& seg2 : ind.net_segments[j]) {
                    if (seg.p1.l == seg2.p1.l &&
                        is_intersecting({ seg.p1.x, seg.p1.y }, { seg.p2.x, seg.p2.y },
                                        { seg2.p1.x, seg2.p1.y }, { seg2.p2.x, seg2.p2.y }))
                        ind.wire_cross_count++;
                }
            }
        }
    }
    ind.wire_overlap_len = wire_comp_overlap_len;

    ind.total_cap = 0.0;
    if (ctx.capacitance_weight > 0.0) {
        for (int i = 0; i < num_nets; ++i)
            for (int j = i + 1; j < num_nets; ++j)
                for (const auto& seg1 : ind.net_segments[i]) {
                    if (seg1.p1.l != seg1.p2.l) continue;
                    for (const auto& seg2 : ind.net_segments[j]) {
                        if (seg2.p1.l != seg2.p2.l) continue;
                        {
                            double dx1 = seg1.p2.x - seg1.p1.x, dy1 = seg1.p2.y - seg1.p1.y;
                            double dx2 = seg2.p2.x - seg2.p1.x, dy2 = seg2.p2.y - seg2.p1.y;
                            if (std::abs(dx1 * dy2 - dx2 * dy1) < 1e-3) {
                                double min_x = std::max(std::min(seg1.p1.x, seg1.p2.x), std::min(seg2.p1.x, seg2.p2.x));
                                double max_x = std::min(std::max(seg1.p1.x, seg1.p2.x), std::max(seg2.p1.x, seg2.p2.x));
                                double min_y = std::max(std::min(seg1.p1.y, seg1.p2.y), std::min(seg2.p1.y, seg2.p2.y));
                                double max_y = std::min(std::max(seg1.p1.y, seg1.p2.y), std::max(seg2.p1.y, seg2.p2.y));
                                if (max_x > min_x || max_y > min_y)
                                    ind.total_cap += std::hypot(max_x - min_x, max_y - min_y) * ctx.grid_res;
                            }
                        }
                    }
                }
    }

    ind.fitness = ctx.wirelength_weight * ind.total_wire_length +
                  ctx.capacitance_weight * ind.total_cap +
                  overlap_score +
                  ctx.dynamic_cross_penalty * ind.wire_cross_count +
                  ctx.wire_comp_overlap_penalty * wire_comp_overlap_len +
                  ctx.current_distance_weight * total_pin_dist +
                  repulsion_penalty +
                  centripetal_dist * ctx.centripetal_weight;
}

std::vector<Individual> init_population(const Context& ctx, std::mt19937& rng) {
    std::vector<Individual> pop(ctx.pop_size);
    std::uniform_real_distribution<> d_rand(0.0, 1.0);
    int angles[] = { 0, 90, 180, 270 };
    for (int i = 0; i < ctx.pop_size; ++i) {
        pop[i].genes.resize(ctx.comps.size());
        for (size_t j = 0; j < ctx.comps.size(); ++j) {
            pop[i].genes[j].x = ctx.comps[j].min_x + d_rand(rng) * (ctx.comps[j].max_x - ctx.comps[j].min_x);
            pop[i].genes[j].y = ctx.comps[j].min_y + d_rand(rng) * (ctx.comps[j].max_y - ctx.comps[j].min_y);
            pop[i].genes[j].angle = angles[std::uniform_int_distribution<>(0, 3)(rng)];
        }
    }
    return pop;
}

Individual tournament_select(const std::vector<Individual>& pop, int size, std::mt19937& rng) {
    std::uniform_int_distribution<> dist(0, static_cast<int>(pop.size()) - 1);
    Individual best = pop[dist(rng)];
    for (int i = 1; i < size; ++i) {
        Individual cand = pop[dist(rng)];
        if (cand.fitness < best.fitness) best = cand;
    }
    Individual clone; clone.genes = best.genes;
    return clone;
}

void mutate(Individual& ind, const Context& ctx, std::mt19937& rng) {
    std::uniform_real_distribution<> d_rand(0.0, 1.0);
    int angles[] = { 0, 90, 180, 270 };
    for (size_t i = 0; i < ind.genes.size(); ++i) {
        if (d_rand(rng) < ctx.mutation_rate) {
            double dx = std::normal_distribution<>(0, (ctx.comps[i].max_x - ctx.comps[i].min_x) * 0.05)(rng);
            double dy = std::normal_distribution<>(0, (ctx.comps[i].max_y - ctx.comps[i].min_y) * 0.05)(rng);
            ind.genes[i].x = std::clamp(ind.genes[i].x + dx, ctx.comps[i].min_x, ctx.comps[i].max_x);
            ind.genes[i].y = std::clamp(ind.genes[i].y + dy, ctx.comps[i].min_y, ctx.comps[i].max_y);
        }
        if (d_rand(rng) < ctx.mutation_rate)
            ind.genes[i].angle = angles[std::uniform_int_distribution<>(0, 3)(rng)];
    }
}

bool is_topologically_similar(const GroupSolution& a, const GroupSolution& b, double threshold) {
    if (a.comp_global_indices != b.comp_global_indices) return false;
    for (size_t i = 0; i < a.local_genes.size(); ++i) {
        const auto& ga = a.local_genes[i];
        const auto& gb = b.local_genes[i];
        if (std::abs(ga.x - gb.x) > threshold || std::abs(ga.y - gb.y) > threshold) return false;
        if (ga.angle != gb.angle) return false;
    }
    return true;
}

GroupGAResult run_group_ga(
    const std::vector<int>& comp_indices,
    const std::vector<ComponentDef>& global_comps,
    const std::vector<NetDef>& global_nets,
    const py::dict& params,
    std::mt19937& rng,
    int group_id = 0,
    int top_n = 5,
    double diversity_threshold = 1.0
) {
    Context local_ctx;
    auto get_param = [&](const std::string& key, double def) -> double {
        if (params.contains(key)) return params[key.c_str()].cast<double>();
        std::cerr << "[WARNING] Parameter '" << key << "' missing, using default " << def << std::endl;
        return def;
    };

    local_ctx.num_layers = (int)get_param("num_layers", 2);
    local_ctx.grid_res = get_param("grid_res", 1.0);
    local_ctx.wirelength_weight = get_param("wirelength_weight", 1.0);
    local_ctx.capacitance_weight = get_param("capacitance_weight", 1.0);
    local_ctx.base_component_overlap_penalty = get_param("base_component_overlap_penalty", 8000.0);
    local_ctx.base_wire_cross_penalty = get_param("base_wire_cross_penalty", 500.0);
    local_ctx.wire_comp_overlap_penalty = get_param("wire_comp_overlap_penalty", 1000.0);
    local_ctx.via_penalty = get_param("via_penalty", 3.5);
    local_ctx.distance_weight = get_param("distance_weight", 0.5);
    local_ctx.component_repulsion_weight = get_param("component_repulsion_weight", 30.0);

    local_ctx.pop_size = (int)get_param("group_pop_size", 100);
    local_ctx.generations = (int)get_param("group_generations", 300);
    local_ctx.mutation_rate = get_param("group_mutation_rate", 0.2);
    local_ctx.crossover_rate = get_param("group_crossover_rate", 0.7);
    local_ctx.tournament_size = (int)get_param("group_tournament_size", 3);
    local_ctx.routing_start_gen = (int)get_param("group_routing_start_gen", 60);
    double group_area_coef = get_param("group_area_coef", 3.0);

    double total_area = 0.0;
    for (int idx : comp_indices) total_area += global_comps[idx].width * global_comps[idx].height;
    double allowed_area = total_area * group_area_coef;
    double local_side = std::sqrt(allowed_area);
    double half_side = local_side / 2.0;

    local_ctx.comps.reserve(comp_indices.size());
    std::unordered_map<int, int> global_to_local_idx;
    for (size_t i = 0; i < comp_indices.size(); ++i) {
        int g_idx = comp_indices[i];
        ComponentDef cdef = global_comps[g_idx];
        cdef.min_x = -half_side;
        cdef.max_x = half_side;
        cdef.min_y = -half_side;
        cdef.max_y = half_side;
        local_ctx.comps.push_back(cdef);
        local_ctx.comp_name_to_idx[cdef.id] = static_cast<int>(i);
        global_to_local_idx[g_idx] = static_cast<int>(i);
    }

    std::vector<int> local_to_global_net;
    double centripetal_factor = get_param("centripetal_factor", 0.1);
    local_ctx.centripetal_weight = local_ctx.distance_weight * centripetal_factor;
    for (size_t net_idx = 0; net_idx < global_nets.size(); ++net_idx) {
        const auto& net = global_nets[net_idx];
        NetDef lnet;
        lnet.net_id = net.net_id;
        for (const auto& conn : net.connections) {
            auto it = global_to_local_idx.find(conn.comp_idx);
            if (it != global_to_local_idx.end())
                lnet.connections.push_back({ it->second, conn.pin_name });
        }
        if (lnet.connections.size() >= 2) {
            local_ctx.nets.push_back(lnet);
            local_to_global_net.push_back(static_cast<int>(net_idx));
        } else if (lnet.connections.size() == 1) {
            local_ctx.single_pin_nets.push_back(lnet);
        }
    }

    std::uniform_real_distribution<> ghost_pos_dist(-half_side, half_side);
    local_ctx.single_pin_ghosts.reserve(local_ctx.single_pin_nets.size());
    for (size_t sni = 0; sni < local_ctx.single_pin_nets.size(); ++sni) {
        double t = ghost_pos_dist(rng);
        Point2D ghost;
        switch (sni % 4) {
            case 0: ghost = { t,  half_side }; break;  
            case 1: ghost = { half_side,  t }; break;  
            case 2: ghost = { t, -half_side }; break; 
            default: ghost = { -half_side, t }; break; 
        }
        local_ctx.single_pin_ghosts.push_back(ghost);
    }

    if (local_ctx.component_repulsion_weight > 0.0) {
        double sum_dim = 0.0;
        for (const auto& comp : local_ctx.comps) sum_dim += (comp.width + comp.height) / 2.0;
        local_ctx.repulsion_threshold = 2.0 * (sum_dim / local_ctx.comps.size());
    } else {
        local_ctx.repulsion_threshold = 0.0;
    }

    auto pop = init_population(local_ctx, rng);
    bool initial_routing = (local_ctx.routing_start_gen == 0);
    #pragma omp parallel for
    for (int i = 0; i < local_ctx.pop_size; ++i)
        evaluate(pop[i], local_ctx, initial_routing, static_cast<unsigned int>(i * 7919u + 0x9E3779B9u));

    int elite_size = std::max(1, local_ctx.pop_size / 10);
    double original_mutation_rate = local_ctx.mutation_rate;
    double best_fitness = std::numeric_limits<double>::max();
    int stale_generations = 0;
    std::vector<double> fitness_history;
    fitness_history.reserve(local_ctx.generations);

    std::cout << "[Group GA " << group_id << "] Start: " << local_ctx.comps.size() << " comps, "
              << local_ctx.nets.size() << " sub-nets, " << local_ctx.single_pin_nets.size()
              << " single-pin nets (" << local_ctx.single_pin_ghosts.size() << " ghost pins)"
              << ", routing starts at gen " << local_ctx.routing_start_gen << std::endl;

    for (int gen = 0; gen < local_ctx.generations; ++gen) {
        bool routing_enabled = (gen >= local_ctx.routing_start_gen);
        if (gen == local_ctx.routing_start_gen && gen > 0) {
            #pragma omp parallel for
            for (int i = 0; i < (int)pop.size(); ++i)
                evaluate(pop[i], local_ctx, true, static_cast<unsigned int>(i * 7919u + gen * 104729u + 11u));
        }

        double progress = (double)gen / std::max(1, local_ctx.generations - 1);
        local_ctx.dynamic_cross_penalty = capped_cross_penalty(local_ctx.base_wire_cross_penalty, progress);
        local_ctx.current_distance_weight = local_ctx.distance_weight * std::max(0.0, 1.0 - progress);

        std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
        double current_best = pop[0].fitness;
        fitness_history.push_back(current_best);
        if (current_best < best_fitness - 1e-9) {
            best_fitness = current_best;
            stale_generations = 0;
            local_ctx.mutation_rate = original_mutation_rate;
        } else {
            stale_generations++;
            if (stale_generations >= 5) local_ctx.mutation_rate = std::max(0.01, original_mutation_rate - 0.1);
        }

        std::vector<Individual> new_pop(pop.begin(), pop.begin() + elite_size);
        std::vector<Individual> offspring;
        while (new_pop.size() + offspring.size() < (size_t)local_ctx.pop_size) {
            Individual p1 = tournament_select(pop, local_ctx.tournament_size, rng);
            Individual p2 = tournament_select(pop, local_ctx.tournament_size, rng);
            Individual c1 = p1, c2 = p2;
            if (local_ctx.comps.size() >= 2 &&
                std::uniform_real_distribution<>(0.0, 1.0)(rng) < local_ctx.crossover_rate) {
                int max_idx = static_cast<int>(local_ctx.comps.size()) - 1;
                int pt1 = std::uniform_int_distribution<>(1, max_idx)(rng);
                int pt2 = std::uniform_int_distribution<>(pt1, max_idx)(rng);
                for (int i = pt1; i <= pt2; ++i) { c1.genes[i] = p2.genes[i]; c2.genes[i] = p1.genes[i]; }
            }
            mutate(c1, local_ctx, rng); mutate(c2, local_ctx, rng);
            offspring.push_back(c1);
            if (new_pop.size() + offspring.size() < (size_t)local_ctx.pop_size) offspring.push_back(c2);
        }
        #pragma omp parallel for
        for (int i = 0; i < (int)offspring.size(); ++i)
            evaluate(offspring[i], local_ctx, routing_enabled, static_cast<unsigned int>(i * 7919u + gen * 104729u + 13u));
        new_pop.insert(new_pop.end(), offspring.begin(), offspring.end());
        pop = std::move(new_pop);

        if (gen % 10 == 0) {
            std::cout << "[Group GA " << group_id << "] Gen " << gen << "/" << local_ctx.generations
                      << " | Best Fitness: " << pop[0].fitness << std::endl;
        }
    }

    int candidate_count = std::min((int)pop.size(), top_n * 10);
    #pragma omp parallel for
    for (int i = 0; i < candidate_count; ++i) {
        evaluate(pop[i], local_ctx, true, static_cast<unsigned int>(i * 7919u + local_ctx.generations * 104729u + 17u));  
    }
    std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });

    std::vector<GroupSolution> result;
    for (int i = 0; i < (int)pop.size() && result.size() < (size_t)top_n; ++i) {
        const Individual& ind = pop[i];
        GroupSolution gsol;
        gsol.comp_global_indices = comp_indices;
        gsol.local_genes = ind.genes;
        gsol.local_abs_pins = ind.abs_pins;
        gsol.fitness = ind.fitness;
        for (size_t ni = 0; ni < local_ctx.nets.size(); ++ni) {
            int global_net_idx = local_to_global_net[ni];
            gsol.local_net_segments[global_net_idx] = ind.net_segments[ni];
        }

        bool similar = false;
        for (const auto& existing : result) {
            if (is_topologically_similar(existing, gsol, diversity_threshold)) {
                similar = true;
                break;
            }
        }
        if (!similar) result.push_back(std::move(gsol));
    }
    if (result.empty() && !pop.empty()) { 
        const Individual& ind = pop[0];
        GroupSolution gsol;
        gsol.comp_global_indices = comp_indices;
        gsol.local_genes = ind.genes;
        gsol.local_abs_pins = ind.abs_pins;
        gsol.fitness = ind.fitness;
        for (size_t ni = 0; ni < local_ctx.nets.size(); ++ni) {
            int global_net_idx = local_to_global_net[ni];
            gsol.local_net_segments[global_net_idx] = ind.net_segments[ni];
        }
        result.push_back(std::move(gsol));
    }

    GroupGAResult ga_result;
    ga_result.solutions = std::move(result);
    ga_result.fitness_history = std::move(fitness_history);
    return ga_result;
}

py::dict run_ga_optimization(py::dict config_dict) {
    py::dict params = config_dict["PARAMS"];

    double board_cx = 0.0, board_cy = 0.0, board_half_x = 0.0, board_half_y = 0.0;
    bool has_board_bounds = false;
    if (config_dict.contains("pcb_bounds")) {
        py::dict b = config_dict["pcb_bounds"].cast<py::dict>();
        double bxmin = b["xmin"].cast<double>(), bymin = b["ymin"].cast<double>();
        double bxmax = b["xmax"].cast<double>(), bymax = b["ymax"].cast<double>();
        board_cx = (bxmin + bxmax) / 2.0; board_cy = (bymin + bymax) / 2.0;
        board_half_x = (bxmax - bxmin) / 2.0; board_half_y = (bymax - bymin) / 2.0;
        has_board_bounds = true;
    }

    Context global_ctx_base;
    int comp_idx = 0;
    for (auto item : config_dict["components"].cast<py::list>()) {
        py::dict c = item.cast<py::dict>();
        ComponentDef cdef;
        cdef.id = c["id"].cast<std::string>();
        cdef.width = c["width"].cast<double>(); cdef.height = c["height"].cast<double>();
        cdef.min_x = cdef.max_x = cdef.min_y = cdef.max_y = 0.0;
        for (auto pin_item : c["pins"].cast<py::list>()) {
            py::dict p = pin_item.cast<py::dict>();
            cdef.pins.push_back({ p["name"].cast<std::string>(), p["dx"].cast<double>(), p["dy"].cast<double>() });
        }
        global_ctx_base.comps.push_back(cdef);
        global_ctx_base.comp_name_to_idx[cdef.id] = comp_idx++;
    }

    for (auto item : config_dict["nets"].cast<py::list>()) {
        py::dict n = item.cast<py::dict>();
        NetDef ndef;
        ndef.net_id = n["net_id"].cast<std::string>();
        for (auto conn_item : n["connections"].cast<py::list>()) {
            py::dict conn = conn_item.cast<py::dict>();
            ndef.connections.push_back({ global_ctx_base.comp_name_to_idx[conn["comp_id"].cast<std::string>()],
                                         conn["pin_name"].cast<std::string>() });
        }
        global_ctx_base.nets.push_back(ndef);
    }

    std::vector<GroupDef> groups;
    std::unordered_map<std::string, int> comp_to_group;
    std::set<int> assigned_comps;
    for (auto item : config_dict["groups"].cast<py::list>()) {
        py::dict g = item.cast<py::dict>();
        GroupDef gdef;
        gdef.name = g["group_name"].cast<std::string>();
        for (auto mem : g["members"].cast<py::list>()) {
            std::string cmp_name = mem.cast<std::string>();
            if (global_ctx_base.comp_name_to_idx.find(cmp_name) == global_ctx_base.comp_name_to_idx.end())
                throw std::runtime_error("Component in group not found: " + cmp_name);
            int idx = global_ctx_base.comp_name_to_idx[cmp_name];
            gdef.comp_indices.push_back(idx);
            gdef.comp_set.insert(idx);
            assigned_comps.insert(idx);
            comp_to_group[cmp_name] = static_cast<int>(groups.size());
        }
        groups.push_back(gdef);
    }

    for (size_t i = 0; i < global_ctx_base.comps.size(); ++i) {
        if (assigned_comps.find(static_cast<int>(i)) == assigned_comps.end()) {
            GroupDef gdef;
            gdef.name = "ungrouped_" + global_ctx_base.comps[i].id;
            gdef.comp_indices.push_back(static_cast<int>(i));
            gdef.comp_set.insert(static_cast<int>(i));
            groups.push_back(gdef);
            comp_to_group[global_ctx_base.comps[i].id] = static_cast<int>(groups.size()) - 1;
        }
    }

    auto get_out_param = [&](const std::string& key, double def) -> double {
        if (params.contains(key)) return params[key.c_str()].cast<double>();
        std::cerr << "[WARNING] Outer param '" << key << "' missing, using default " << def << std::endl;
        return def;
    };

    int top_n = (int)get_out_param("top_n", 5);
    double diversity_threshold = get_out_param("diversity_threshold", 1.0);
    double ema_alpha = get_out_param("ema_alpha", 0.3);
    double softmax_t0 = get_out_param("softmax_t0", 5000.0);
    double softmax_tend = get_out_param("softmax_tend", 1.0);

    std::mt19937 rng(93 + 1);
    std::vector<std::vector<GroupSolution>> group_solutions_pool(groups.size());
    std::vector<std::vector<double>> group_fitness_histories(groups.size());
    for (size_t g = 0; g < groups.size(); ++g) {
        std::mt19937 local_rng(rng() + static_cast<unsigned int>(g * 1000));
        GroupGAResult ga_result = run_group_ga(groups[g].comp_indices, global_ctx_base.comps, global_ctx_base.nets,
                                                params, local_rng, static_cast<int>(g), top_n, diversity_threshold);
        group_solutions_pool[g] = std::move(ga_result.solutions);
        group_fitness_histories[g] = std::move(ga_result.fitness_history);
        if (group_solutions_pool[g].empty()) {
            throw std::runtime_error("Group GA returned no solution for group " + groups[g].name);
        }
    }

    int out_pop_size = (int)get_out_param("out_pop_size", 200);
    int out_generations = (int)get_out_param("out_generations", 500);
    double out_mutation_rate = get_out_param("out_mutation_rate", 0.2);
    double out_crossover_rate = get_out_param("out_crossover_rate", 0.7);
    int out_tournament_size = (int)get_out_param("out_tournament_size", 3);
    int out_routing_start_gen = (int)get_out_param("out_routing_start_gen", 100);
    int out_combine_start_gen = (int)get_out_param("out_combine_start_gen", 450);
    out_combine_start_gen = std::max(1, out_combine_start_gen);

    int out_full_reroute_interval = (int)get_out_param("out_full_reroute_interval", 25);
    out_full_reroute_interval = std::max(1, out_full_reroute_interval);

    double stage2_step_max_factor = get_out_param("stage2_step_max_factor", 0.05);   
    double stage2_step_min_factor = get_out_param("stage2_step_min_factor", 0.0005); 

    double diversity_std_threshold = get_out_param("diversity_std_threshold", 1e-6);
    double diversity_mutation_delta = get_out_param("diversity_mutation_delta", 0.05);
    double diversity_crossover_delta = get_out_param("diversity_crossover_delta", 0.1);
    double original_out_mutation_rate = out_mutation_rate;
    double original_out_crossover_rate = out_crossover_rate;

    double group_base_component_overlap_penalty = get_out_param("group_base_component_overlap_penalty", 1.0);
    double group_repulsion_weight = get_out_param("group_repulsion_weight", 30.0);
    double component_repulsion_weight = get_out_param("component_repulsion_weight", 30.0);
    double base_component_overlap_penalty = get_out_param("base_component_overlap_penalty", 8000.0);
    double wire_comp_overlap_penalty = get_out_param("wire_comp_overlap_penalty", 1000.0);
    double base_wire_cross_penalty = get_out_param("base_wire_cross_penalty", 500.0);
    double distance_weight = get_out_param("distance_weight", 0.5);
    double wirelength_weight = get_out_param("wirelength_weight", 1.0);
    double capacitance_weight = get_out_param("capacitance_weight", 1.0);
    double via_penalty = get_out_param("via_penalty", 3.5);
    double grid_res = get_out_param("grid_res", 1.0);
    int num_layers = (int)get_out_param("num_layers", 2);

    auto compute_layout_score = [](const Individual& ind) -> double {
        double congestion = static_cast<double>(ind.wire_cross_count) * 10.0 + ind.overlap_score;
        return ind.total_cap * 1.5 +
               ind.total_wire_length * 1.0 +
               congestion * 5.0;
    };
    std::vector<std::pair<int, double>> layout_score_history;
    layout_score_history.reserve(out_generations);

    double max_group_span = 0.0;
    for (const auto& sols : group_solutions_pool) {
        const auto& gsol = sols[0];
        double min_x = std::numeric_limits<double>::infinity(), max_x = -min_x;
        double min_y = min_x, max_y = -min_x;
        for (size_t i = 0; i < gsol.local_genes.size(); ++i) {
            const auto& gene = gsol.local_genes[i];
            const auto& comp = global_ctx_base.comps[gsol.comp_global_indices[i]];
            auto box = get_component_aabb(gene.x, gene.y, comp.width, comp.height, gene.angle);
            min_x = std::min(min_x, box.xmin); max_x = std::max(max_x, box.xmax);
            min_y = std::min(min_y, box.ymin); max_y = std::max(max_y, box.ymax);
        }
        max_group_span = std::max(max_group_span, max_x - min_x);
        max_group_span = std::max(max_group_span, max_y - min_y);
    }
    double global_side, global_half;
    if (has_board_bounds) {
        global_side = std::max(board_half_x * 2.0, board_half_y * 2.0);
        global_half = global_side / 2.0;
    } else {
        global_side = max_group_span * std::sqrt(static_cast<double>(groups.size())) * 1.5 + 100.0;
        global_half = global_side / 2.0;
        board_cx = 0.0; board_cy = 0.0;
        board_half_x = global_half; board_half_y = global_half;
    }


    struct OutTransform { double x, y; int angle; };
    struct OutChromosome {
        std::vector<OutTransform> group_transforms;
        std::vector<int> group_solution_indices;   
        std::vector<CompGene> comp_genes;       
    };

    std::vector<std::vector<double>> ema_scores(groups.size(), std::vector<double>(top_n, 0.0));

    auto temperature = [&](int gen) {
        double progress = (double)gen / std::max(1, out_generations);
        return softmax_t0 * std::pow(softmax_tend / softmax_t0, progress);
    };

    auto evaluate_global = [&](const OutChromosome& chromo, int gen, std::mt19937& eval_rng,
                               bool independent_scoring = false, bool update_ema = true) -> Individual {
        int num_global_comps = static_cast<int>(global_ctx_base.comps.size());
        int num_global_nets = static_cast<int>(global_ctx_base.nets.size());

        const double eff_grid_res = independent_scoring ? 1.0 : grid_res;
        const double eff_via_penalty = independent_scoring ? 3.5 : via_penalty;
        const double eff_wire_comp_penalty = independent_scoring ? 400.0 : wire_comp_overlap_penalty;
        const double eff_cross_penalty = independent_scoring ? 1500.0
            : capped_cross_penalty(base_wire_cross_penalty, (double)gen / out_generations);

        Individual global_ind;
        global_ind.genes.resize(num_global_comps);
        global_ind.abs_pins.resize(num_global_comps);

        if (gen < out_combine_start_gen) {
            for (size_t g = 0; g < groups.size(); ++g) {
                const auto& gsol = group_solutions_pool[g][chromo.group_solution_indices[g]];
                const auto& tr = chromo.group_transforms[g];
                for (size_t i = 0; i < gsol.comp_global_indices.size(); ++i) {
                    int global_idx = gsol.comp_global_indices[i];
                    const auto& local_gene = gsol.local_genes[i];
                    Point2D rotated = rotate_point(local_gene.x, local_gene.y, tr.angle);
                    global_ind.genes[global_idx].x = rotated.x + tr.x;
                    global_ind.genes[global_idx].y = rotated.y + tr.y;
                    global_ind.genes[global_idx].angle = (local_gene.angle + tr.angle) % 360;
                    if (global_ind.genes[global_idx].angle < 0) global_ind.genes[global_idx].angle += 360;
                    int ang = global_ind.genes[global_idx].angle;
                    if (ang % 90 != 0) ang = ((ang + 45) / 90) * 90;
                    global_ind.genes[global_idx].angle = ang % 360;
                }
            }
        } else {
            global_ind.genes = chromo.comp_genes;
        }

        double global_min_x = std::numeric_limits<double>::infinity(), global_max_x = -global_min_x;
        double global_min_y = global_min_x, global_max_y = -global_min_x;
        std::vector<AABB> global_comp_boxes(num_global_comps);
        for (int i = 0; i < num_global_comps; ++i) {
            const auto& c = global_ctx_base.comps[i];
            const auto& gene = global_ind.genes[i];
            global_comp_boxes[i] = get_component_aabb(gene.x, gene.y, c.width, c.height, gene.angle);
            global_min_x = std::min(global_min_x, global_comp_boxes[i].xmin);
            global_max_x = std::max(global_max_x, global_comp_boxes[i].xmax);
            global_min_y = std::min(global_min_y, global_comp_boxes[i].ymin);
            global_max_y = std::max(global_max_y, global_comp_boxes[i].ymax);
            for (const auto& pin : c.pins) {
                global_ind.abs_pins[i][pin.name] = rotate_point(pin.dx, pin.dy, gene.angle);
                global_ind.abs_pins[i][pin.name].x += gene.x;
                global_ind.abs_pins[i][pin.name].y += gene.y;
            }
        }

        double overlap_score = 0.0, repulsion_penalty = 0.0;

        double current_overlap_penalty = independent_scoring ? 6000.0 : group_base_component_overlap_penalty;
        double current_repulsion_weight = independent_scoring ? 30.0 : group_repulsion_weight;
        if (!independent_scoring && gen >= out_combine_start_gen) {
            int transition_gens = 100;
            double ratio = 1.0;
            if (gen < out_combine_start_gen + transition_gens) {
                ratio = (double)(gen - out_combine_start_gen) / transition_gens;
            }
            current_overlap_penalty = group_base_component_overlap_penalty +
                                      ratio * (base_component_overlap_penalty - group_base_component_overlap_penalty);
            current_repulsion_weight = group_repulsion_weight +
                                       ratio * (component_repulsion_weight - group_repulsion_weight);
        }

        for (int i = 0; i < num_global_comps; ++i) {
            for (int j = i + 1; j < num_global_comps; ++j) {
                double dx = std::max(0.0, std::min(global_comp_boxes[i].xmax, global_comp_boxes[j].xmax) - std::max(global_comp_boxes[i].xmin, global_comp_boxes[j].xmin));
                double dy = std::max(0.0, std::min(global_comp_boxes[i].ymax, global_comp_boxes[j].ymax) - std::max(global_comp_boxes[i].ymin, global_comp_boxes[j].ymin));
                if (dx > 0 && dy > 0) {
                    double inter_area = dx * dy;
                    double act_d = std::hypot(global_ind.genes[i].x - global_ind.genes[j].x, global_ind.genes[i].y - global_ind.genes[j].y);
                    double safe_d = std::hypot(global_ctx_base.comps[i].width / 2, global_ctx_base.comps[i].height / 2) +
                                    std::hypot(global_ctx_base.comps[j].width / 2, global_ctx_base.comps[j].height / 2);
                    overlap_score += (std::max(0.1, safe_d - act_d) * 10 + inter_area) * current_overlap_penalty;
                } else if (current_repulsion_weight > 0.0) {
                    double D = std::hypot(global_ind.genes[i].x - global_ind.genes[j].x, global_ind.genes[i].y - global_ind.genes[j].y);
                    double rep_threshold = 0.0;
                    { double sum_dim = 0.0; for (const auto& comp : global_ctx_base.comps) sum_dim += (comp.width + comp.height) / 2.0; rep_threshold = 2.0 * (sum_dim / global_ctx_base.comps.size()); }
                    if (D < rep_threshold) repulsion_penalty += (rep_threshold - D) / rep_threshold * current_repulsion_weight;
                }
            }
        }
        global_ind.overlap_score = overlap_score;
        global_ind.repulsion_penalty = repulsion_penalty;

        double total_pin_dist = 0.0;
        for (int i = 0; i < num_global_nets; ++i) {
            std::vector<Point2D> pts;
            for (const auto& conn : global_ctx_base.nets[i].connections)
                pts.push_back(global_ind.abs_pins[conn.comp_idx][conn.pin_name]);
            auto edges = mst(pts);
            for (auto e : edges) total_pin_dist += std::hypot(pts[e.first].x - pts[e.second].x, pts[e.first].y - pts[e.second].y);
        }
        global_ind.total_pin_dist = total_pin_dist;

        bool do_routing = independent_scoring || (gen >= out_routing_start_gen);
        global_ind.net_segments.assign(num_global_nets, {});
        double total_wire_length = 0.0;
        int wire_cross_count = 0;
        double wire_overlap_len = 0.0;
        double total_cap = 0.0;

        if (do_routing) {
            double route_min_x = global_min_x - 30.0, route_max_x = global_max_x + 30.0;
            double route_min_y = global_min_y - 30.0, route_max_y = global_max_y + 30.0;
            int cols = static_cast<int>((route_max_x - route_min_x) / eff_grid_res + 1);
            int rows = static_cast<int>((route_max_y - route_min_y) / eff_grid_res + 1);
            std::vector<int> occupancy_grid(cols * rows * num_layers, -1);
            std::vector<double> history_cost(cols * rows * num_layers, 0.0);

            std::vector<int> unprocessed;

            if (gen < out_combine_start_gen) {
                for (size_t g = 0; g < groups.size(); ++g) {
                    const auto& gsol = group_solutions_pool[g][chromo.group_solution_indices[g]];
                    const auto& tr = chromo.group_transforms[g];
                    for (const auto& kv : gsol.local_net_segments) {
                        int net_idx = kv.first;
                        std::vector<Segment3D> g_segs;
                        for (const auto& seg : kv.second) {
                            Point2D rp1 = rotate_point(seg.p1.x, seg.p1.y, tr.angle);
                            Point2D rp2 = rotate_point(seg.p2.x, seg.p2.y, tr.angle);
                            g_segs.push_back({ { rp1.x + tr.x, rp1.y + tr.y, seg.p1.l },
                                               { rp2.x + tr.x, rp2.y + tr.y, seg.p2.l } });
                        }
                        global_ind.net_segments[net_idx] = g_segs;
                    }
                }

                for (int ni = 0; ni < num_global_nets; ++ni)
                    if (!global_ind.net_segments[ni].empty())
                        manage_grid(occupancy_grid, cols, rows, route_min_x, route_min_y, grid_res, global_ind.net_segments[ni], ni, true);

                std::set<int> cross_net_set;
                for (int ni = 0; ni < num_global_nets; ++ni) {
                    const auto& net = global_ctx_base.nets[ni];
                    int first_group = -1;
                    for (const auto& conn : net.connections) {
                        auto it = comp_to_group.find(global_ctx_base.comps[conn.comp_idx].id);
                        if (it == comp_to_group.end()) { cross_net_set.insert(ni); break; }
                        if (first_group == -1) first_group = it->second;
                        else if (it->second != first_group) { cross_net_set.insert(ni); break; }
                    }
                }
                std::vector<int> cross_nets(cross_net_set.begin(), cross_net_set.end());
                std::vector<double> est_len(cross_nets.size(), 0.0);
                for (size_t i = 0; i < cross_nets.size(); ++i) {
                    int ni = cross_nets[i];
                    std::vector<Point2D> pts;
                    for (const auto& conn : global_ctx_base.nets[ni].connections)
                        pts.push_back(global_ind.abs_pins[conn.comp_idx][conn.pin_name]);
                    auto edges = mst(pts);
                    for (auto e : edges) est_len[i] += std::hypot(pts[e.first].x - pts[e.second].x, pts[e.first].y - pts[e.second].y);
                }
                std::vector<int> order(cross_nets.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return est_len[a] < est_len[b]; });
                for (int idx : order) unprocessed.push_back(cross_nets[idx]);
            } else {
                std::vector<double> est_len(num_global_nets, 0.0);
                for (int ni = 0; ni < num_global_nets; ++ni) {
                    std::vector<Point2D> pts;
                    for (const auto& conn : global_ctx_base.nets[ni].connections)
                        pts.push_back(global_ind.abs_pins[conn.comp_idx][conn.pin_name]);
                    auto edges = mst(pts);
                    for (auto e : edges) est_len[ni] += std::hypot(pts[e.first].x - pts[e.second].x, pts[e.first].y - pts[e.second].y);
                }
                std::vector<int> order(num_global_nets);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return est_len[a] < est_len[b]; });
                for (int idx : order) unprocessed.push_back(idx);
            }

            std::mt19937 route_rng(eval_rng() ^ 0xABCD);
            std::vector<int> rip_up_counts(num_global_nets, 0);
            while (!unprocessed.empty()) {
                int ni = unprocessed.front(); unprocessed.erase(unprocessed.begin());
                if (!global_ind.net_segments[ni].empty()) {
                    manage_grid(occupancy_grid, cols, rows, route_min_x, route_min_y, eff_grid_res, global_ind.net_segments[ni], ni, false);
                    global_ind.net_segments[ni].clear();
                }
                std::vector<Point2D> pts;
                for (const auto& conn : global_ctx_base.nets[ni].connections)
                    pts.push_back(global_ind.abs_pins[conn.comp_idx][conn.pin_name]);
                auto edges = mst(pts);
                std::vector<Segment3D> segs;
                for (auto e : edges) {
                    Point3D p1 = { pts[e.first].x,  pts[e.first].y,  0 };
                    Point3D p2 = { pts[e.second].x, pts[e.second].y, 0 };
                    Context tmp_ctx;
                    tmp_ctx.num_layers = num_layers;
                    tmp_ctx.grid_res = eff_grid_res;
                    tmp_ctx.wire_comp_overlap_penalty = eff_wire_comp_penalty;
                    tmp_ctx.dynamic_cross_penalty = eff_cross_penalty;
                    tmp_ctx.via_penalty = eff_via_penalty;
                    auto path = a_star_3d(p1, p2, global_comp_boxes, occupancy_grid, history_cost, cols, rows, route_min_x, route_min_y, tmp_ctx, ni);
                    for (size_t k = 1; k < path.size(); ++k)
                        segs.push_back({ path[k - 1], path[k] });
                }

                std::vector<int> crossed_nets;
                for (const auto& seg : segs) {
                    if (seg.p1.l == seg.p2.l) {
                        int c0 = static_cast<int>(std::round((seg.p1.x - route_min_x) / eff_grid_res));
                        int r0 = static_cast<int>(std::round((seg.p1.y - route_min_y) / eff_grid_res));
                        int c1 = static_cast<int>(std::round((seg.p2.x - route_min_x) / eff_grid_res));
                        int r1 = static_cast<int>(std::round((seg.p2.y - route_min_y) / eff_grid_res));
                        int sc = c0 < c1 ? 1 : -1, sr = r0 < r1 ? 1 : -1;
                        int dc = std::abs(c1 - c0), dr = -std::abs(r1 - r0);
                        int err = dc + dr;
                        while (true) {
                            if (c0 >= 0 && c0 < cols && r0 >= 0 && r0 < rows) {
                                int cell_idx = seg.p1.l * (cols * rows) + r0 * cols + c0;
                                int occ = occupancy_grid[cell_idx];
                                if (occ != -1 && occ != ni && std::find(crossed_nets.begin(), crossed_nets.end(), occ) == crossed_nets.end())
                                    crossed_nets.push_back(occ);
                            }
                            if (c0 == c1 && r0 == r1) break;
                            int e2 = 2 * err;
                            if (e2 >= dr) { err += dr; c0 += sc; }
                            if (e2 <= dc) { err += dc; r0 += sr; }
                        }
                    }
                }
                std::shuffle(crossed_nets.begin(), crossed_nets.end(), route_rng);
                for (int cn : crossed_nets) {
                    if (rip_up_counts[cn] < 10) {
                        manage_grid(occupancy_grid, cols, rows, route_min_x, route_min_y, eff_grid_res, global_ind.net_segments[cn], cn, false);
                        global_ind.net_segments[cn].clear();
                        unprocessed.push_back(cn);
                        rip_up_counts[cn]++;
                    }
                }
                manage_grid(occupancy_grid, cols, rows, route_min_x, route_min_y, eff_grid_res, segs, ni, true);
                global_ind.net_segments[ni] = segs;
            }
        }

        for (const auto& segs : global_ind.net_segments)
            for (const auto& seg : segs)
                total_wire_length += (seg.p1.l == seg.p2.l) ? std::hypot(seg.p1.x - seg.p2.x, seg.p1.y - seg.p2.y) : eff_via_penalty;

        for (int i = 0; i < num_global_nets; ++i) {
            for (const auto& seg : global_ind.net_segments[i]) {
                if (seg.p1.l != seg.p2.l) continue;
                if (seg.p1.l == 0)
                    for (const auto& box : global_comp_boxes)
                        if (std::max(seg.p1.x, seg.p2.x) > box.xmin && std::min(seg.p1.x, seg.p2.x) < box.xmax &&
                            std::max(seg.p1.y, seg.p2.y) > box.ymin && std::min(seg.p1.y, seg.p2.y) < box.ymax)
                            wire_overlap_len += eff_grid_res;
                for (int j = i + 1; j < num_global_nets; ++j)
                    for (const auto& seg2 : global_ind.net_segments[j])
                        if (seg.p1.l == seg2.p1.l &&
                            is_intersecting({ seg.p1.x, seg.p1.y }, { seg.p2.x, seg.p2.y },
                                            { seg2.p1.x, seg2.p1.y }, { seg2.p2.x, seg2.p2.y }))
                            wire_cross_count++;
            }
        }

        if (do_routing && (independent_scoring || capacitance_weight > 0.0)) {
            for (int i = 0; i < num_global_nets; ++i)
                for (int j = i + 1; j < num_global_nets; ++j)
                    for (const auto& s1 : global_ind.net_segments[i]) {
                        if (s1.p1.l != s1.p2.l) continue;
                        for (const auto& s2 : global_ind.net_segments[j]) {
                            if (s2.p1.l != s2.p2.l) continue;
                            double dx1 = s1.p2.x - s1.p1.x, dy1 = s1.p2.y - s1.p1.y;
                            double dx2 = s2.p2.x - s2.p1.x, dy2 = s2.p2.y - s2.p1.y;
                            if (std::abs(dx1 * dy2 - dx2 * dy1) < 1e-3) {
                                double min_x = std::max(std::min(s1.p1.x, s1.p2.x), std::min(s2.p1.x, s2.p2.x));
                                double max_x = std::min(std::max(s1.p1.x, s1.p2.x), std::max(s2.p1.x, s2.p2.x));
                                double min_y = std::max(std::min(s1.p1.y, s1.p2.y), std::min(s2.p1.y, s2.p2.y));
                                double max_y = std::min(std::max(s1.p1.y, s1.p2.y), std::max(s2.p1.y, s2.p2.y));
                                if (max_x > min_x || max_y > min_y)
                                    total_cap += std::hypot(max_x - min_x, max_y - min_y) * eff_grid_res;
                            }
                        }
                    }
        }

        global_ind.total_wire_length = total_wire_length;
        global_ind.wire_cross_count = wire_cross_count;
        global_ind.wire_overlap_len = wire_overlap_len;
        global_ind.total_cap = total_cap;

        double dynamic_cross = eff_cross_penalty;
        double cur_dist_weight = distance_weight * std::max(0.0, 1.0 - (double)gen / out_generations);
        global_ind.fitness = wirelength_weight * total_wire_length +
                             capacitance_weight * total_cap +
                             overlap_score +
                             dynamic_cross * wire_cross_count +
                             eff_wire_comp_penalty * wire_overlap_len +
                             cur_dist_weight * total_pin_dist +
                             repulsion_penalty;

        global_ind.eval_overlap_penalty = current_overlap_penalty;
        global_ind.eval_repulsion_weight = current_repulsion_weight;

        if (update_ema && !independent_scoring && gen < out_combine_start_gen) {
            #pragma omp critical(ema_update)
            {
                for (size_t g = 0; g < groups.size(); ++g) {
                    int idx = chromo.group_solution_indices[g];
                    ema_scores[g][idx] = (1 - ema_alpha) * ema_scores[g][idx] + ema_alpha * global_ind.fitness;
                }
            }
        }

        return global_ind;
    };

    auto evaluate_incremental = [&](const Individual& old_ind, const OutChromosome& chromo,
                                    int gen, std::mt19937& eval_rng) -> Individual {
        int num_global_comps = static_cast<int>(global_ctx_base.comps.size());
        int num_global_nets = static_cast<int>(global_ctx_base.nets.size());

        if (old_ind.genes.size() != (size_t)num_global_comps || old_ind.abs_pins.size() != (size_t)num_global_comps)
            return evaluate_global(chromo, gen, eval_rng);

        Individual ind;
        ind.genes = chromo.comp_genes;
        ind.abs_pins.resize(num_global_comps);

        std::vector<bool> comp_moved(num_global_comps, false);
        for (int i = 0; i < num_global_comps; ++i) {
            const auto& ng = chromo.comp_genes[i];
            const auto& og = old_ind.genes[i];
            if (std::abs(ng.x - og.x) > 1e-9 || std::abs(ng.y - og.y) > 1e-9 || ng.angle != og.angle)
                comp_moved[i] = true;
        }

        std::vector<bool> net_affected(num_global_nets, false);
        for (int ni = 0; ni < num_global_nets; ++ni) {
            for (const auto& conn : global_ctx_base.nets[ni].connections) {
                if (comp_moved[conn.comp_idx]) { net_affected[ni] = true; break; }
            }
        }

        std::vector<AABB> boxes(num_global_comps), old_boxes(num_global_comps);
        std::vector<AABB> mov_boxes, old_mov_boxes;
        double gmin_x = std::numeric_limits<double>::infinity(), gmax_x = -gmin_x;
        double gmin_y = gmin_x, gmax_y = -gmin_x;
        for (int i = 0; i < num_global_comps; ++i) {
            const auto& c = global_ctx_base.comps[i];
            const auto& gene = ind.genes[i];
            boxes[i] = get_component_aabb(gene.x, gene.y, c.width, c.height, gene.angle);
            gmin_x = std::min(gmin_x, boxes[i].xmin); gmax_x = std::max(gmax_x, boxes[i].xmax);
            gmin_y = std::min(gmin_y, boxes[i].ymin); gmax_y = std::max(gmax_y, boxes[i].ymax);
            for (const auto& pin : c.pins) {
                ind.abs_pins[i][pin.name] = rotate_point(pin.dx, pin.dy, gene.angle);
                ind.abs_pins[i][pin.name].x += gene.x;
                ind.abs_pins[i][pin.name].y += gene.y;
            }
            const auto& og = old_ind.genes[i];
            old_boxes[i] = get_component_aabb(og.x, og.y, c.width, c.height, og.angle);
            if (comp_moved[i]) {
                old_mov_boxes.push_back(old_boxes[i]);
                mov_boxes.push_back(boxes[i]);
            }
        }

        double current_overlap_penalty = group_base_component_overlap_penalty;
        double current_repulsion_weight = group_repulsion_weight;
        if (gen >= out_combine_start_gen) {
            int transition_gens = 100;
            double ratio = 1.0;
            if (gen < out_combine_start_gen + transition_gens)
                ratio = (double)(gen - out_combine_start_gen) / transition_gens;
            current_overlap_penalty = group_base_component_overlap_penalty +
                                      ratio * (base_component_overlap_penalty - group_base_component_overlap_penalty);
            current_repulsion_weight = group_repulsion_weight +
                                       ratio * (component_repulsion_weight - group_repulsion_weight);
        }

        double rep_threshold = 0.0;
        {
            double sum_dim = 0.0;
            for (const auto& comp : global_ctx_base.comps) sum_dim += (comp.width + comp.height) / 2.0;
            rep_threshold = 2.0 * (sum_dim / global_ctx_base.comps.size());
        }

        auto pair_contribution = [&](int i, int j, const CompGene& gi, const CompGene& gj,
                                     double overlap_pen, double repul_w) -> std::pair<double, double> {
            const auto& ci = global_ctx_base.comps[i];
            const auto& cj = global_ctx_base.comps[j];
            AABB bi = get_component_aabb(gi.x, gi.y, ci.width, ci.height, gi.angle);
            AABB bj = get_component_aabb(gj.x, gj.y, cj.width, cj.height, gj.angle);
            double dx = std::max(0.0, std::min(bi.xmax, bj.xmax) - std::max(bi.xmin, bj.xmin));
            double dy = std::max(0.0, std::min(bi.ymax, bj.ymax) - std::max(bi.ymin, bj.ymin));
            if (dx > 0 && dy > 0) {
                double inter_area = dx * dy;
                double act_d = std::hypot(gi.x - gj.x, gi.y - gj.y);
                double safe_d = std::hypot(ci.width / 2, ci.height / 2) + std::hypot(cj.width / 2, cj.height / 2);
                return { (std::max(0.1, safe_d - act_d) * 10 + inter_area) * overlap_pen, 0.0 };
            }
            if (repul_w > 0.0) {
                double D = std::hypot(gi.x - gj.x, gi.y - gj.y);
                if (D < rep_threshold) return { 0.0, (rep_threshold - D) / rep_threshold * repul_w };
            }
            return { 0.0, 0.0 };
        };

        double overlap_score = old_ind.overlap_score;
        double repulsion_penalty = old_ind.repulsion_penalty;
        for (int i = 0; i < num_global_comps; ++i) {
            if (!comp_moved[i]) continue;
            for (int k = 0; k < i; ++k) {
                if (comp_moved[k]) continue; 
                auto old_c = pair_contribution(k, i, old_ind.genes[k], old_ind.genes[i],
                                               old_ind.eval_overlap_penalty, old_ind.eval_repulsion_weight);
                auto new_c = pair_contribution(k, i, ind.genes[k], ind.genes[i],
                                               current_overlap_penalty, current_repulsion_weight);
                overlap_score += new_c.first - old_c.first;
                repulsion_penalty += new_c.second - old_c.second;
            }
            for (int j = i + 1; j < num_global_comps; ++j) {
                auto old_c = pair_contribution(i, j, old_ind.genes[i], old_ind.genes[j],
                                               old_ind.eval_overlap_penalty, old_ind.eval_repulsion_weight);
                auto new_c = pair_contribution(i, j, ind.genes[i], ind.genes[j],
                                               current_overlap_penalty, current_repulsion_weight);
                overlap_score += new_c.first - old_c.first;
                repulsion_penalty += new_c.second - old_c.second;
            }
        }
        ind.overlap_score = overlap_score;
        ind.repulsion_penalty = repulsion_penalty;

        double total_pin_dist = old_ind.total_pin_dist;
        for (int ni = 0; ni < num_global_nets; ++ni) {
            if (!net_affected[ni]) continue;
            auto mst_len_of = [&](const std::vector<std::unordered_map<std::string, Point2D>>& pins_map) {
                std::vector<Point2D> pts;
                for (const auto& conn : global_ctx_base.nets[ni].connections)
                    pts.push_back(pins_map[conn.comp_idx].at(conn.pin_name));
                double s = 0.0;
                auto edges = mst(pts);
                for (auto e : edges) s += std::hypot(pts[e.first].x - pts[e.second].x, pts[e.first].y - pts[e.second].y);
                return s;
            };
            total_pin_dist += mst_len_of(ind.abs_pins) - mst_len_of(old_ind.abs_pins);
        }
        ind.total_pin_dist = total_pin_dist;

        bool do_routing = (gen >= out_routing_start_gen);
        ind.net_segments = old_ind.net_segments;
        double total_wire_length = old_ind.total_wire_length;
        int wire_cross_count = old_ind.wire_cross_count;
        double wire_overlap_len = old_ind.wire_overlap_len;
        double total_cap = old_ind.total_cap;

        auto crosses_between = [&](const std::vector<Segment3D>& segs1, const std::vector<Segment3D>& segs2) -> int {
            int cnt = 0;
            for (const auto& s1 : segs1) {
                if (s1.p1.l != s1.p2.l) continue;
                for (const auto& s2 : segs2) {
                    if (s1.p1.l == s2.p1.l &&
                        is_intersecting({ s1.p1.x, s1.p1.y }, { s1.p2.x, s1.p2.y },
                                        { s2.p1.x, s2.p1.y }, { s2.p2.x, s2.p2.y }))
                        cnt++;
                }
            }
            return cnt;
        };

        auto cap_between = [&](const std::vector<Segment3D>& segs1, const std::vector<Segment3D>& segs2) -> double {
            double c = 0.0;
            for (const auto& s1 : segs1) {
                if (s1.p1.l != s1.p2.l) continue;
                for (const auto& s2 : segs2) {
                    if (s2.p1.l != s2.p2.l) continue;
                    double dx1 = s1.p2.x - s1.p1.x, dy1 = s1.p2.y - s1.p1.y;
                    double dx2 = s2.p2.x - s2.p1.x, dy2 = s2.p2.y - s2.p1.y;
                    if (std::abs(dx1 * dy2 - dx2 * dy1) < 1e-3) {
                        double mx = std::max(std::min(s1.p1.x, s1.p2.x), std::min(s2.p1.x, s2.p2.x));
                        double Mx = std::min(std::max(s1.p1.x, s1.p2.x), std::max(s2.p1.x, s2.p2.x));
                        double my = std::max(std::min(s1.p1.y, s1.p2.y), std::min(s2.p1.y, s2.p2.y));
                        double My = std::min(std::max(s1.p1.y, s1.p2.y), std::max(s2.p1.y, s2.p2.y));
                        if (Mx > mx || My > my) c += std::hypot(Mx - mx, My - my) * grid_res;
                    }
                }
            }
            return c;
        };

        auto wire_overlap_between = [&](const std::vector<Segment3D>& segs, const std::vector<AABB>& boxlist) -> double {
            double s = 0.0;
            for (const auto& seg : segs) {
                if (seg.p1.l != seg.p2.l || seg.p1.l != 0) continue;
                for (const auto& box : boxlist)
                    if (std::max(seg.p1.x, seg.p2.x) > box.xmin && std::min(seg.p1.x, seg.p2.x) < box.xmax &&
                        std::max(seg.p1.y, seg.p2.y) > box.ymin && std::min(seg.p1.y, seg.p2.y) < box.ymax)
                        s += grid_res;
            }
            return s;
        };

        std::vector<int> affected_nets;
        if (do_routing) {
            for (int ni = 0; ni < num_global_nets; ++ni)
                if (net_affected[ni] || old_ind.net_segments[ni].empty())
                    affected_nets.push_back(ni);
        }

        if (do_routing && !affected_nets.empty()) {
            std::vector<Segment3D> old_aff_segs;
            for (int a : affected_nets)
                for (const auto& seg : old_ind.net_segments[a])
                    old_aff_segs.push_back(seg);

            for (int a : affected_nets) {
                for (const auto& seg : old_ind.net_segments[a])
                    total_wire_length -= (seg.p1.l == seg.p2.l) ? std::hypot(seg.p1.x - seg.p2.x, seg.p1.y - seg.p2.y) : via_penalty;
                for (int k = 0; k < a; ++k) {
                    if (net_affected[k]) continue; 
                    wire_cross_count -= crosses_between(old_ind.net_segments[k], old_ind.net_segments[a]);
                    if (capacitance_weight > 0.0) total_cap -= cap_between(old_ind.net_segments[k], old_ind.net_segments[a]);
                }
                for (int j = a + 1; j < num_global_nets; ++j) {
                    wire_cross_count -= crosses_between(old_ind.net_segments[a], old_ind.net_segments[j]);
                    if (capacitance_weight > 0.0) total_cap -= cap_between(old_ind.net_segments[a], old_ind.net_segments[j]);
                }
            }
            double old_wire_comp = 0.0;
            for (int ni = 0; ni < num_global_nets; ++ni)
                old_wire_comp += wire_overlap_between(old_ind.net_segments[ni], old_mov_boxes);
            old_wire_comp += wire_overlap_between(old_aff_segs, old_boxes);
            old_wire_comp -= wire_overlap_between(old_aff_segs, old_mov_boxes);
            wire_overlap_len -= old_wire_comp;

            double route_min_x = gmin_x - 30.0, route_max_x = gmax_x + 30.0;
            double route_min_y = gmin_y - 30.0, route_max_y = gmax_y + 30.0;
            int cols = static_cast<int>((route_max_x - route_min_x) / grid_res + 1);
            int rows = static_cast<int>((route_max_y - route_min_y) / grid_res + 1);
            std::vector<int> occupancy_grid(cols * rows * num_layers, -1);
            std::vector<double> history_cost(cols * rows * num_layers, 0.0);

            for (int ni = 0; ni < num_global_nets; ++ni)
                if (!net_affected[ni] && !old_ind.net_segments[ni].empty())
                    manage_grid(occupancy_grid, cols, rows, route_min_x, route_min_y, grid_res, old_ind.net_segments[ni], ni, true);

            std::vector<double> est_len(affected_nets.size(), 0.0);
            for (size_t ai = 0; ai < affected_nets.size(); ++ai) {
                int ni = affected_nets[ai];
                std::vector<Point2D> pts;
                for (const auto& conn : global_ctx_base.nets[ni].connections)
                    pts.push_back(ind.abs_pins[conn.comp_idx][conn.pin_name]);
                auto edges = mst(pts);
                for (auto e : edges) est_len[ai] += std::hypot(pts[e.first].x - pts[e.second].x, pts[e.first].y - pts[e.second].y);
            }
            std::vector<int> order(affected_nets.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return est_len[a] < est_len[b]; });
            std::vector<int> unprocessed;
            for (int idx : order) unprocessed.push_back(affected_nets[idx]);

            std::mt19937 route_rng(eval_rng() ^ 0xABCD);
            std::vector<int> rip_up_counts(num_global_nets, 0);
            while (!unprocessed.empty()) {
                int ni = unprocessed.front(); unprocessed.erase(unprocessed.begin());
                if (!ind.net_segments[ni].empty()) {
                    manage_grid(occupancy_grid, cols, rows, route_min_x, route_min_y, grid_res, ind.net_segments[ni], ni, false);
                    ind.net_segments[ni].clear();
                }
                std::vector<Point2D> pts;
                for (const auto& conn : global_ctx_base.nets[ni].connections)
                    pts.push_back(ind.abs_pins[conn.comp_idx][conn.pin_name]);
                auto edges = mst(pts);
                std::vector<Segment3D> segs;
                for (auto e : edges) {
                    Point3D p1 = { pts[e.first].x,  pts[e.first].y,  0 };
                    Point3D p2 = { pts[e.second].x, pts[e.second].y, 0 };
                    Context tmp_ctx;
                    tmp_ctx.num_layers = num_layers;
                    tmp_ctx.grid_res = grid_res;
                    tmp_ctx.wire_comp_overlap_penalty = wire_comp_overlap_penalty;
                    tmp_ctx.dynamic_cross_penalty = capped_cross_penalty(base_wire_cross_penalty, (double)gen / out_generations);
                    tmp_ctx.via_penalty = via_penalty;
                    auto path = a_star_3d(p1, p2, boxes, occupancy_grid, history_cost, cols, rows, route_min_x, route_min_y, tmp_ctx, ni);
                    for (size_t k = 1; k < path.size(); ++k)
                        segs.push_back({ path[k - 1], path[k] });
                }

                std::vector<int> crossed_nets;
                for (const auto& seg : segs) {
                    if (seg.p1.l != seg.p2.l) continue;
                    int c0 = static_cast<int>(std::round((seg.p1.x - route_min_x) / grid_res));
                    int r0 = static_cast<int>(std::round((seg.p1.y - route_min_y) / grid_res));
                    int c1 = static_cast<int>(std::round((seg.p2.x - route_min_x) / grid_res));
                    int r1 = static_cast<int>(std::round((seg.p2.y - route_min_y) / grid_res));
                    int sc = c0 < c1 ? 1 : -1, sr = r0 < r1 ? 1 : -1;
                    int dc = std::abs(c1 - c0), dr = -std::abs(r1 - r0);
                    int err = dc + dr;
                    while (true) {
                        if (c0 >= 0 && c0 < cols && r0 >= 0 && r0 < rows) {
                            int cell_idx = seg.p1.l * (cols * rows) + r0 * cols + c0;
                            int occ = occupancy_grid[cell_idx];
                            if (occ != -1 && occ != ni &&
                                std::find(crossed_nets.begin(), crossed_nets.end(), occ) == crossed_nets.end())
                                crossed_nets.push_back(occ);
                        }
                        if (c0 == c1 && r0 == r1) break;
                        int e2 = 2 * err;
                        if (e2 >= dr) { err += dr; c0 += sc; }
                        if (e2 <= dc) { err += dc; r0 += sr; }
                    }
                }
                std::shuffle(crossed_nets.begin(), crossed_nets.end(), route_rng);
                for (int cn : crossed_nets) {
                    if (net_affected[cn] && rip_up_counts[cn] < 10) {
                        manage_grid(occupancy_grid, cols, rows, route_min_x, route_min_y, grid_res, ind.net_segments[cn], cn, false);
                        ind.net_segments[cn].clear();
                        unprocessed.push_back(cn);
                        rip_up_counts[cn]++;
                    }
                }
                manage_grid(occupancy_grid, cols, rows, route_min_x, route_min_y, grid_res, segs, ni, true);
                ind.net_segments[ni] = segs;
            }

            for (int a : affected_nets) {
                for (const auto& seg : ind.net_segments[a])
                    total_wire_length += (seg.p1.l == seg.p2.l) ? std::hypot(seg.p1.x - seg.p2.x, seg.p1.y - seg.p2.y) : via_penalty;
                for (int k = 0; k < a; ++k) {
                    if (net_affected[k]) continue;
                    wire_cross_count += crosses_between(ind.net_segments[k], ind.net_segments[a]);
                    if (capacitance_weight > 0.0) total_cap += cap_between(ind.net_segments[k], ind.net_segments[a]);
                }
                for (int j = a + 1; j < num_global_nets; ++j) {
                    const auto& segs_j = net_affected[j] ? ind.net_segments[j] : old_ind.net_segments[j];
                    wire_cross_count += crosses_between(ind.net_segments[a], segs_j);
                    if (capacitance_weight > 0.0) total_cap += cap_between(ind.net_segments[a], segs_j);
                }
            }
            std::vector<Segment3D> new_aff_segs;
            for (int a : affected_nets)
                for (const auto& seg : ind.net_segments[a])
                    new_aff_segs.push_back(seg);
            double new_wire_comp = 0.0;
            for (int ni = 0; ni < num_global_nets; ++ni)
                new_wire_comp += wire_overlap_between(ind.net_segments[ni], mov_boxes);
            new_wire_comp += wire_overlap_between(new_aff_segs, boxes);
            new_wire_comp -= wire_overlap_between(new_aff_segs, mov_boxes);
            wire_overlap_len += new_wire_comp;
        }

        double dynamic_cross = capped_cross_penalty(base_wire_cross_penalty, (double)gen / out_generations);
        double cur_dist_weight = distance_weight * std::max(0.0, 1.0 - (double)gen / out_generations);
        ind.total_wire_length = total_wire_length;
        ind.wire_cross_count = wire_cross_count;
        ind.wire_overlap_len = wire_overlap_len;
        ind.total_cap = total_cap;
        ind.eval_overlap_penalty = current_overlap_penalty;
        ind.eval_repulsion_weight = current_repulsion_weight;

        ind.fitness = wirelength_weight * total_wire_length +
                      capacitance_weight * total_cap +
                      overlap_score +
                      dynamic_cross * wire_cross_count +
                      wire_comp_overlap_penalty * wire_overlap_len +
                      cur_dist_weight * total_pin_dist +
                      repulsion_penalty;
        return ind;
    };

    auto init_out_pop = [&](std::mt19937& rng_init) {
        std::vector<OutChromosome> pop(out_pop_size);
        std::uniform_real_distribution<> pos_dist_x(board_cx - board_half_x, board_cx + board_half_x);
        std::uniform_real_distribution<> pos_dist_y(board_cy - board_half_y, board_cy + board_half_y);
        int angles[] = { 0, 90, 180, 270 };
        for (int i = 0; i < out_pop_size; ++i) {
            pop[i].group_transforms.resize(groups.size());
            pop[i].group_solution_indices.resize(groups.size());
            for (size_t g = 0; g < groups.size(); ++g) {
                pop[i].group_transforms[g].x = pos_dist_x(rng_init);
                pop[i].group_transforms[g].y = pos_dist_y(rng_init);
                pop[i].group_transforms[g].angle = angles[std::uniform_int_distribution<>(0, 3)(rng_init)];
                pop[i].group_solution_indices[g] = std::uniform_int_distribution<>(0, top_n - 1)(rng_init);
            }
            pop[i].comp_genes.resize(global_ctx_base.comps.size());
        }
        return pop;
    };

    std::mt19937 out_rng(201 + out_pop_size);
    auto out_pop = init_out_pop(out_rng);
    int elite_size = std::max(1, out_pop_size / 10);
    std::vector<double> fitness_history;
    std::vector<std::vector<double>> fitness_components_history;

    struct IndCache { Individual ind; OutChromosome chromo; };
    std::vector<IndCache> cache(out_pop_size);
    for (int i = 0; i < out_pop_size; ++i) {
        std::mt19937 thread_rng(static_cast<unsigned int>(out_rng() + i * 100));
        cache[i].ind = evaluate_global(out_pop[i], 0, thread_rng);
        cache[i].chromo = out_pop[i];
    }

    std::cout << "[Two-Stage GA] Routing starts at gen " << out_routing_start_gen
              << ", stage-2 fine-tuning starts at gen " << out_combine_start_gen
              << ", periodic full reroute interval " << out_full_reroute_interval << std::endl;

    for (int gen = 0; gen < out_generations; ++gen) {

        bool stage2 = (gen >= out_combine_start_gen);
        bool periodic_full_reroute = stage2 && (gen % out_full_reroute_interval == 0);
        if (periodic_full_reroute) {
            std::cout << "[Two-Stage GA] Gen " << gen << ": periodic full rerouting (interval="
                      << out_full_reroute_interval << ")..." << std::endl;
        }

        if (gen == out_combine_start_gen) {
            std::cout << "[Two-Stage GA] Flattening group-level genes into component-level genes for all individuals..." << std::endl;

            for (int i = 0; i < out_pop_size; ++i) {
                cache[i].chromo.comp_genes = cache[i].ind.genes;
                out_pop[i] = cache[i].chromo;
            }
        }

        std::vector<double> fits(out_pop_size);
        for (int i = 0; i < out_pop_size; ++i) fits[i] = cache[i].ind.fitness;

        std::vector<int> indices(out_pop_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) { return fits[a] < fits[b]; });

        fitness_history.push_back(fits[indices[0]]);
        const auto& best_cache = cache[indices[0]];
        fitness_components_history.push_back({
            best_cache.ind.total_wire_length, best_cache.ind.total_cap, best_cache.ind.overlap_score,
            (double)best_cache.ind.wire_cross_count, best_cache.ind.wire_overlap_len,
            best_cache.ind.total_pin_dist, best_cache.ind.repulsion_penalty, best_cache.ind.fitness
        });

        double best_layout_score = std::numeric_limits<double>::max();
        for (int i = 0; i < out_pop_size; ++i) {
            const Individual& ind = cache[i].ind;
            bool routed = false;
            for (const auto& segs : ind.net_segments)
                if (!segs.empty()) { routed = true; break; }
            if (!routed) continue;
            best_layout_score = std::min(best_layout_score, compute_layout_score(ind));
        }
        if (best_layout_score < std::numeric_limits<double>::max())
            layout_score_history.emplace_back(gen, best_layout_score);

        if (gen % 10 == 0) {
            std::cout << "[Two-Stage GA] Gen " << gen << "/" << out_generations
                      << " (stage " << (stage2 ? 2 : 1) << ") | Best Fitness: " << fits[indices[0]] << std::endl;
        }

        if (!stage2) {
            int half = out_pop_size / 2;
            double mean = 0.0;
            for (int k = 0; k < half; ++k) mean += fits[indices[k]];
            mean /= half;
            double var = 0.0;
            for (int k = 0; k < half; ++k) {
                double diff = fits[indices[k]] - mean;
                var += diff * diff;
            }
            var /= half;
            double std_dev = std::sqrt(var);
            if (std_dev < diversity_std_threshold) {
                out_mutation_rate = std::min(1.0, out_mutation_rate + diversity_mutation_delta);
                out_crossover_rate = std::max(0.0, out_crossover_rate - diversity_crossover_delta);
            } else {
                out_mutation_rate += (original_out_mutation_rate - out_mutation_rate) * 0.1;
                out_crossover_rate += (original_out_crossover_rate - out_crossover_rate) * 0.1;
            }
        }

        std::vector<OutChromosome> new_pop_chromo;
        std::vector<IndCache> new_cache;
        std::vector<int> parent_idx(out_pop_size, indices[0]);
        for (int i = 0; i < elite_size; ++i) {
            int idx = indices[i];
            IndCache ic = cache[idx];
            if (stage2) {
                std::mt19937 elite_rng(static_cast<unsigned int>(out_rng() + idx * 100 + gen * 10000));
                if (periodic_full_reroute) {
                    ic.ind = evaluate_global(ic.chromo, gen + 1, elite_rng);
                } else {
                    ic.ind = evaluate_incremental(ic.ind, ic.chromo, gen, elite_rng);
                }
            } else {
                ic.ind = evaluate_global(ic.chromo, gen, out_rng);
            }
            new_pop_chromo.push_back(ic.chromo);
            new_cache.push_back(ic);
            parent_idx[i] = idx;
        }

        auto select_idx = [&](int size, std::mt19937& rnd) -> int {
            std::uniform_int_distribution<> dist(0, (int)out_pop.size() - 1);
            int best_idx = dist(rnd);
            for (int i = 1; i < size; ++i) {
                int cand = dist(rnd);
                if (fits[cand] < fits[best_idx]) best_idx = cand;
            }
            return best_idx;
        };

        auto crossover_group = [&](OutChromosome& a, OutChromosome& b, std::mt19937& rnd) {
            if (groups.size() < 2) return;
            int max_idx = (int)groups.size() - 1;
            int pt1 = std::uniform_int_distribution<>(1, max_idx)(rnd);
            int pt2 = std::uniform_int_distribution<>(pt1, max_idx)(rnd);
            for (int i = pt1; i <= pt2; ++i) {
                std::swap(a.group_transforms[i], b.group_transforms[i]);
                std::swap(a.group_solution_indices[i], b.group_solution_indices[i]); 
            }
        };

        auto crossover_comp = [&](OutChromosome& a, OutChromosome& b, std::mt19937& rnd) {
            std::uniform_real_distribution<> d01(0.0, 1.0);
            size_t n = std::min(a.comp_genes.size(), b.comp_genes.size());
            for (size_t i = 0; i < n; ++i)
                if (d01(rnd) < 0.5) std::swap(a.comp_genes[i], b.comp_genes[i]);
        };

        auto mutate_group = [&](OutChromosome& chromo, std::mt19937& rnd) {
            static int ang[] = {0,90,180,270};
            double T = temperature(gen);
            for (size_t g=0; g<chromo.group_transforms.size(); ++g) {
                if (std::uniform_real_distribution<>(0,1)(rnd) < out_mutation_rate) {
                    chromo.group_transforms[g].x += std::normal_distribution<>(0, global_side*0.05)(rnd);
                    chromo.group_transforms[g].y += std::normal_distribution<>(0, global_side*0.05)(rnd);
                    chromo.group_transforms[g].x = std::clamp(chromo.group_transforms[g].x,
                        board_cx - board_half_x, board_cx + board_half_x);
                    chromo.group_transforms[g].y = std::clamp(chromo.group_transforms[g].y,
                        board_cy - board_half_y, board_cy + board_half_y);
                }
                if (std::uniform_real_distribution<>(0,1)(rnd) < out_mutation_rate)
                    chromo.group_transforms[g].angle = ang[std::uniform_int_distribution<>(0,3)(rnd)];

                if (std::uniform_real_distribution<>(0,1)(rnd) < out_mutation_rate) {
                    const auto& scores = ema_scores[g];
                    double max_s = *std::max_element(scores.begin(), scores.end());
                    std::vector<double> probs(top_n);
                    double sum = 0.0;
                    for (int k = 0; k < top_n; ++k) {
                        probs[k] = std::exp((max_s - scores[k]) / T);
                        sum += probs[k];
                    }
                    for (int k = 0; k < top_n; ++k) probs[k] /= sum;
                    std::discrete_distribution<int> dist(probs.begin(), probs.end());
                    chromo.group_solution_indices[g] = dist(rnd);
                }
                // -------------------------------------------------
            }
        };

        auto mutate_comp = [&](OutChromosome& chromo, std::mt19937& rnd) {
            int n_comps = (int)chromo.comp_genes.size();
            if (n_comps <= 0) return;
            double progress = (double)(gen - out_combine_start_gen) / std::max(1, out_generations - out_combine_start_gen);
            progress = std::clamp(progress, 0.0, 1.0);
            double step = global_side * (stage2_step_max_factor + (stage2_step_min_factor - stage2_step_max_factor) * progress);
            int n_mut = std::min(std::uniform_int_distribution<>(1, 3)(rnd), n_comps);
            std::vector<int> pool(n_comps);
            std::iota(pool.begin(), pool.end(), 0);
            std::shuffle(pool.begin(), pool.end(), rnd);
            for (int m = 0; m < n_mut; ++m) {
                int c_idx = pool[m];
                chromo.comp_genes[c_idx].x += std::normal_distribution<>(0.0, step)(rnd);
                chromo.comp_genes[c_idx].y += std::normal_distribution<>(0.0, step)(rnd);
                chromo.comp_genes[c_idx].x = std::clamp(chromo.comp_genes[c_idx].x,
                    board_cx - board_half_x, board_cx + board_half_x);
                chromo.comp_genes[c_idx].y = std::clamp(chromo.comp_genes[c_idx].y,
                    board_cy - board_half_y, board_cy + board_half_y);
            }
        };

        while (new_pop_chromo.size() < (size_t)out_pop_size) {
            int i1 = select_idx(out_tournament_size, out_rng);
            int i2 = select_idx(out_tournament_size, out_rng);
            auto c1 = out_pop[i1], c2 = out_pop[i2];
            int child_pos = (int)new_pop_chromo.size();

            if (stage2) {
                if (std::uniform_real_distribution<>(0.0, 1.0)(out_rng) < out_crossover_rate)
                    crossover_comp(c1, c2, out_rng);
                if (std::uniform_real_distribution<>(0.0, 1.0)(out_rng) < out_mutation_rate)
                    mutate_comp(c1, out_rng);
                if (std::uniform_real_distribution<>(0.0, 1.0)(out_rng) < out_mutation_rate)
                    mutate_comp(c2, out_rng);
            } else {
                if (std::uniform_real_distribution<>(0.0, 1.0)(out_rng) < out_crossover_rate)
                    crossover_group(c1, c2, out_rng);
                mutate_group(c1, out_rng); mutate_group(c2, out_rng);
            }

            new_pop_chromo.push_back(c1);
            parent_idx[child_pos] = i1;
            if (new_pop_chromo.size() < (size_t)out_pop_size) {
                new_pop_chromo.push_back(c2);
                parent_idx[child_pos + 1] = i2;
            }
        }

        new_cache.resize(out_pop_size);
        #pragma omp parallel for
        for (int i = elite_size; i < out_pop_size; ++i) {
            std::mt19937 thread_rng(static_cast<unsigned int>(i * 7919u + gen * 104729u + 0x9E3779B9u));
            IndCache ic;
            ic.chromo = new_pop_chromo[i];
            if (stage2) {
                if (periodic_full_reroute) {
                    ic.ind = evaluate_global(ic.chromo, gen + 1, thread_rng, false, false);
                } else {
                    const Individual& ref_ind = cache[parent_idx[i]].ind;
                    ic.ind = evaluate_incremental(ref_ind, ic.chromo, gen + 1, thread_rng);
                }
            } else {
                ic.ind = evaluate_global(ic.chromo, gen, thread_rng, false, false);
            }
            #pragma omp critical
            new_cache[i] = ic;
        }

        if (!stage2) {
            for (int i = elite_size; i < out_pop_size; ++i) {
                const IndCache& icc = new_cache[i];
                for (size_t g = 0; g < groups.size(); ++g) {
                    int sidx = icc.chromo.group_solution_indices[g];
                    ema_scores[g][sidx] = (1 - ema_alpha) * ema_scores[g][sidx] + ema_alpha * icc.ind.fitness;
                }
            }
        }

        out_pop = new_pop_chromo;
        cache = new_cache;
    }


    {
        std::sort(cache.begin(), cache.end(), [](const IndCache& a, const IndCache& b) { return a.ind.fitness < b.ind.fitness; });
        int refresh_count = std::min(50, (int)cache.size());
        #pragma omp parallel for
        for (int i = 0; i < refresh_count; ++i) {
            OutChromosome chromo;
            chromo.comp_genes = cache[i].ind.genes;
            std::mt19937 r(static_cast<unsigned int>(i * 7919 + 2026));
            cache[i].ind = evaluate_global(chromo, out_generations - 1, r, true);
        }
    }

    std::sort(cache.begin(), cache.end(), [](const IndCache& a, const IndCache& b) { return a.ind.fitness < b.ind.fitness; });

    if (!layout_score_history.empty() && !cache.empty()) {
        double final_layout_score = compute_layout_score(cache[0].ind);
        if (layout_score_history.back().first == out_generations - 1)
            layout_score_history.back().second = final_layout_score;
        else
            layout_score_history.emplace_back(out_generations - 1, final_layout_score);
    }

    py::list py_best;
    int n_top = std::min(10, (int)cache.size());
    for (int i = 0; i < n_top; ++i) {
        const Individual& ind = cache[i].ind;
        py::dict py_ind;
        py::dict py_comps, py_pins;
        for (int j = 0; j < (int)global_ctx_base.comps.size(); ++j) {
            py::dict c;
            c["x"] = ind.genes[j].x; c["y"] = ind.genes[j].y; c["angle"] = ind.genes[j].angle;
            py_comps[py::str(global_ctx_base.comps[j].id)] = c;
            py::dict p_dict;
            for (const auto& kv : ind.abs_pins[j])
                p_dict[py::str(kv.first)] = py::make_tuple(kv.second.x, kv.second.y);
            py_pins[py::str(global_ctx_base.comps[j].id)] = p_dict;
        }
        py::dict py_nets;
        for (size_t n = 0; n < global_ctx_base.nets.size(); ++n) {
            py::list segs;
            for (const auto& seg : ind.net_segments[n])
                segs.append(py::make_tuple(py::make_tuple(seg.p1.x, seg.p1.y, seg.p1.l),
                                           py::make_tuple(seg.p2.x, seg.p2.y, seg.p2.l)));
            py_nets[py::str(global_ctx_base.nets[n].net_id)] = segs;
        }
        py_ind["comps"] = py_comps;
        py_ind["pins"] = py_pins;
        py_ind["net_segments"] = py_nets;
        py_ind["fitness"] = ind.fitness;
        py_ind["total_wire_length"] = ind.total_wire_length;
        py_ind["total_cap"] = ind.total_cap;
        py_ind["overlap_score"] = ind.overlap_score;
        py_ind["wire_cross_count"] = ind.wire_cross_count;
        py_best.append(py_ind);
    }

    py::list py_local_groups;
    for (size_t g = 0; g < groups.size(); ++g) {
        const auto& gsol = group_solutions_pool[g][0]; 
        py::dict py_ind;
        py::dict py_comps, py_pins;
        for (size_t i = 0; i < gsol.local_genes.size(); ++i) {
            int global_idx = gsol.comp_global_indices[i];
            const std::string& cid = global_ctx_base.comps[global_idx].id;
            py::dict c;
            c["x"] = gsol.local_genes[i].x; c["y"] = gsol.local_genes[i].y; c["angle"] = gsol.local_genes[i].angle;
            py_comps[py::str(cid)] = c;
            py::dict p_dict;
            if (i < gsol.local_abs_pins.size())
                for (const auto& kv : gsol.local_abs_pins[i])
                    p_dict[py::str(kv.first)] = py::make_tuple(kv.second.x, kv.second.y);
            py_pins[py::str(cid)] = p_dict;
        }
        py::dict py_nets;
        for (const auto& kv : gsol.local_net_segments) {
            const auto& net = global_ctx_base.nets[kv.first];
            py::list segs;
            for (const auto& seg : kv.second)
                segs.append(py::make_tuple(py::make_tuple(seg.p1.x, seg.p1.y, seg.p1.l),
                                           py::make_tuple(seg.p2.x, seg.p2.y, seg.p2.l)));
            py_nets[py::str(net.net_id)] = segs;
        }
        py_ind["comps"] = py_comps;
        py_ind["pins"] = py_pins;
        py_ind["net_segments"] = py_nets;
        py_ind["fitness"] = gsol.fitness;
        py_ind["total_wire_length"] = 0.0;
        py_ind["total_cap"] = 0.0;
        py_ind["overlap_score"] = 0.0;
        py_ind["wire_cross_count"] = 0;
        py_local_groups.append(py_ind);
    }

    py::dict final_result;
    final_result["best_individuals"] = py_best;
    final_result["local_groups_best_individuals"] = py_local_groups;
    final_result["fitness_history"] = py::cast(fitness_history);

    py::list py_comp_hist;
    for (const auto& row : fitness_components_history) {
        py::list r;
        for (double v : row) r.append(v);
        py_comp_hist.append(r);
    }
    final_result["fitness_components_history"] = py_comp_hist;

    py::list py_group_histories;
    for (const auto& hist : group_fitness_histories)
        py_group_histories.append(py::cast(hist));
    final_result["group_fitness_histories"] = py_group_histories;

    py::list py_ls_history;
    for (const auto& p : layout_score_history)
        py_ls_history.append(py::make_tuple(p.first, p.second));
    final_result["layout_score_history"] = py_ls_history;

    struct LayoutScoreEntry {
        std::string layout_id;
        double cap, wire_length, congestion, total_score;
    };
    std::vector<LayoutScoreEntry> score_entries;
    score_entries.reserve(n_top);
    for (int i = 0; i < n_top; ++i) {
        const Individual& ind = cache[i].ind;
        LayoutScoreEntry e;
        std::string lid = "best_";
        if (i < 10) lid += "0";
        lid += std::to_string(i);
        e.layout_id = lid;
        e.cap = ind.total_cap;
        e.wire_length = ind.total_wire_length;
        e.congestion = ind.wire_cross_count * 10.0 + ind.overlap_score;
        e.total_score = compute_layout_score(ind);
        score_entries.push_back(e);
    }
    std::sort(score_entries.begin(), score_entries.end(),
              [](const LayoutScoreEntry& a, const LayoutScoreEntry& b) { return a.total_score < b.total_score; });
    py::list py_layout_scores;
    for (const auto& e : score_entries) {
        py::dict d;
        d["layout_id"] = e.layout_id;
        d["parasitic_capacitance"] = e.cap;
        d["wire_length"] = e.wire_length;
        d["congestion"] = e.congestion;
        d["total_score"] = e.total_score;
        py_layout_scores.append(d);
    }
    final_result["layout_scores"] = py_layout_scores;

    return final_result;
}

PYBIND11_MODULE(pcb_engine, m) {
    m.doc() = "High performance C++ PCB Routing & Two-Stage GA Engine";
    m.def("run_ga_optimization", &run_ga_optimization, "Run Two-Stage GA for PCB placement and routing");
}
