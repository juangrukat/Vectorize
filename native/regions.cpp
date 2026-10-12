// Connected Ward agglomeration: the mathematical merge criterion is Ward's
// increase in squared reconstruction error. A deterministic priority queue is
// an independent scheduling choice, rather than the original raster schedule.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <vector>
namespace py = pybind11;
using Image = py::array_t<double, py::array::c_style | py::array::forcecast>;
struct Node {
    int parent;
    int count = 1;
    uint64_t version = 0;
    std::array<double, 4> sum;
    std::unordered_set<int> neighbors;
};
struct Edge {
    double cost;
    int a, b;
    uint64_t va, vb;
};
struct Later {
    bool operator()(const Edge &a, const Edge &b) const {
        if (a.cost != b.cost) return a.cost > b.cost;
        if (a.a != b.a) return a.a > b.a;
        return a.b > b.b;
    }
};
class Ward {
    std::vector<Node> nodes;
    std::priority_queue<Edge, std::vector<Edge>, Later> queue;
    int active;
    int minimum;
    bool small_only = false;

    int root(int id) {
        int r = id;
        while (nodes[r].parent != r) r = nodes[r].parent;
        while (nodes[id].parent != id) {
            int next = nodes[id].parent;
            nodes[id].parent = r;
            id = next;
        }
        return r;
    }
    double cost(int a, int b) const {
        const auto &x = nodes[a];
        const auto &y = nodes[b];
        double distance = 0;
        for (int k = 0; k < 4; ++k) {
            double d = x.sum[k] / x.count - y.sum[k] / y.count;
            distance += d * d;
        }
        return (static_cast<double>(x.count) * y.count / (x.count + y.count)) * distance;
    }
    void push(int a, int b) {
        if (a == b) return;
        if (small_only && nodes[a].count >= minimum && nodes[b].count >= minimum) return;
        if (a > b) std::swap(a, b);
        queue.push({cost(a, b), a, b, nodes[a].version, nodes[b].version});
    }
    bool pop(Edge &edge) {
        while (!queue.empty()) {
            edge = queue.top();
            queue.pop();
            if (nodes[edge.a].parent != edge.a || nodes[edge.b].parent != edge.b) continue;
            if (nodes[edge.a].version != edge.va || nodes[edge.b].version != edge.vb) continue;
            if (small_only && nodes[edge.a].count >= minimum && nodes[edge.b].count >= minimum) continue;
            return true;
        }
        return false;
    }
    void rebuild(bool small) {
        small_only = small;
        queue = decltype(queue){};
        for (int a = 0; a < static_cast<int>(nodes.size()); ++a) {
            if (nodes[a].parent != a) continue;
            for (int b : nodes[a].neighbors) if (a < b) push(a, b);
        }
    }
    void merge(int a, int b, bool update_graph) {
        // Union by size bounds disjoint-set paths. Equal sizes favor lower id.
        if (nodes[a].count < nodes[b].count ||
            (nodes[a].count == nodes[b].count && a > b)) std::swap(a, b);
        nodes[b].parent = a;
        nodes[a].count += nodes[b].count;
        for (int k = 0; k < 4; ++k) nodes[a].sum[k] += nodes[b].sum[k];
        ++nodes[a].version;
        --active;
        if (!update_graph) return;
        nodes[a].neighbors.erase(b);
        for (int neighbor : nodes[b].neighbors) {
            if (neighbor == a) continue;
            nodes[neighbor].neighbors.erase(b);
            nodes[neighbor].neighbors.insert(a);
            nodes[a].neighbors.insert(neighbor);
        }
        nodes[b].neighbors.clear();
        for (int neighbor : nodes[a].neighbors) push(a, neighbor);
    }
public:
    Ward(const double *image, int width, int height, int min_pixels)
        : nodes(static_cast<size_t>(width) * height), active(width * height), minimum(min_pixels) {
        for (int i = 0; i < active; ++i) {
            nodes[i].parent = i;
            for (int k = 0; k < 4; ++k) nodes[i].sum[k] = image[4 * i + k];
        }
        // Exact equal-color components can be joined before graph creation:
        // every such Ward merge has zero cost and leaves the mean unchanged.
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            auto equal = [&](int j) {
                for (int k = 0; k < 4; ++k) if (image[4 * i + k] != image[4 * j + k]) return false;
                return true;
            };
            if (x && equal(i - 1)) { int a = root(i), b = root(i - 1); if (a != b) merge(a, b, false); }
            if (y && equal(i - width)) { int a = root(i), b = root(i - width); if (a != b) merge(a, b, false); }
        }
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            int i = y * width + x, a = root(i);
            auto connect = [&](int j) {
                int b = root(j);
                if (a != b) { nodes[a].neighbors.insert(b); nodes[b].neighbors.insert(a); }
            };
            if (x + 1 < width) connect(i + 1);
            if (y + 1 < height) connect(i + width);
        }
        rebuild(false);
    }
    void run(double threshold, int max_regions) {
        Edge edge;
        while (pop(edge)) {
            if (edge.cost >= threshold) break;
            merge(edge.a, edge.b, true);
        }
        if (minimum > 1) {
            rebuild(true);
            while (pop(edge)) merge(edge.a, edge.b, true);
        }
        if (active > max_regions) {
            rebuild(false);
            while (active > max_regions && pop(edge)) merge(edge.a, edge.b, true);
        }
    }
    int count() const { return active; }
    void write(const double *image, int width, int height, int passes, int64_t *labels, double *palette) {
        std::vector<int> ids(nodes.size(), -1);
        int next = 0;
        std::vector<int> counts(active, 0);
        std::vector<std::array<double, 4>> sums(active);
        // First pixel order fixes compact output region ids deterministically.
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            int r = root(i);
            if (ids[r] < 0) {
                ids[r] = next++;
                counts[ids[r]] = nodes[r].count;
                sums[ids[r]] = nodes[r].sum;
                for (int k = 0; k < 4; ++k) palette[4 * ids[r] + k] = nodes[r].sum[k] / nodes[r].count;
            }
            labels[i] = ids[r];
        }
        // Original-inspired local boundary reassignment. The scan order and
        // simple-point check are independent choices. Immediate mean updates
        // make every accepted transfer strictly reduce total region SSE.
        auto simple_removal = [&](int x, int y, int region) {
            bool available[9] = {}, reached[9] = {};
            int first = -1;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                if (!dx && !dy) continue;
                int xx = x + dx, yy = y + dy, local = (dy + 1) * 3 + dx + 1;
                if (xx >= 0 && xx < width && yy >= 0 && yy < height && labels[yy * width + xx] == region)
                    available[local] = true;
            }
            const int axial[4] = {1, 3, 5, 7};
            for (int local : axial) if (available[local]) { first = local; break; }
            if (first < 0) return false;
            int work[9], used = 1, head = 0;
            work[0] = first; reached[first] = true;
            while (head < used) {
                int local = work[head++], xx = local % 3, yy = local / 3;
                auto visit = [&](int n) { if (available[n] && !reached[n]) { reached[n] = true; work[used++] = n; } };
                if (xx > 0) visit(local - 1);
                if (xx < 2) visit(local + 1);
                if (yy > 0) visit(local - 3);
                if (yy < 2) visit(local + 3);
            }
            for (int local : axial) if (available[local] && !reached[local]) return false;
            return true;
        };
        auto transfer = [&](int i) {
            int a = static_cast<int>(labels[i]);
            if (counts[a] <= 1) return;
            int x = i % width, y = i / width;
            int adjacent[4], size = 0;
            if (x > 0) adjacent[size++] = i - 1;
            if (x + 1 < width) adjacent[size++] = i + 1;
            if (y > 0) adjacent[size++] = i - width;
            if (y + 1 < height) adjacent[size++] = i + width;
            auto distance = [&](int r) {
                double d2 = 0;
                for (int k = 0; k < 4; ++k) {
                    double d = image[4 * i + k] - sums[r][k] / counts[r];
                    d2 += d * d;
                }
                return d2;
            };
            int best = a;
            double best_distance = distance(a);
            for (int j = 0; j < size; ++j) {
                int b = static_cast<int>(labels[adjacent[j]]);
                if (b == a) continue;
                double d2 = distance(b);
                if (d2 < best_distance) { best = b; best_distance = d2; }
            }
            if (best == a || !simple_removal(x, y, a)) return;
            --counts[a]; ++counts[best];
            for (int k = 0; k < 4; ++k) {
                sums[a][k] -= image[4 * i + k];
                sums[best][k] += image[4 * i + k];
            }
            labels[i] = best;
        };
        for (int pass = 0; pass < passes; ++pass) {
            for (int i = 0; i < width * height; ++i) transfer(i);
            for (int i = width * height - 1; i >= 0; --i) transfer(i);
        }
        for (int r = 0; r < active; ++r) for (int k = 0; k < 4; ++k)
            palette[4 * r + k] = sums[r][k] / counts[r];
    }
};
PYBIND11_MODULE(_regions, m) {
    m.doc() = "Connected Ward segmentation with independently estimated region colors.";
    m.def("ward", [](Image image, double threshold, int min_pixels, int max_regions, int refine_passes) {
        if (image.ndim() != 3 || image.shape(2) != 4 || image.shape(0) < 1 || image.shape(1) < 1 ||
            image.shape(0) * image.shape(1) > 1048576)
            throw std::invalid_argument("image must be H x W x 4 with at most 1048576 pixels");
        if (!std::isfinite(threshold) || threshold < 0 || min_pixels < 1 || min_pixels > 1048576 ||
            max_regions < 1 || max_regions > 1048576 || refine_passes < 0 || refine_passes > 4)
            throw std::invalid_argument("invalid Ward threshold or region bounds");
        for (ssize_t i = 0; i < image.size(); ++i)
            if (!std::isfinite(image.data()[i]) || image.data()[i] < 0 || image.data()[i] > 1)
                throw std::invalid_argument("image values must be finite within [0, 1]");
        int h = static_cast<int>(image.shape(0)), w = static_cast<int>(image.shape(1));
        py::array_t<int64_t> labels({h, w});
        // Allocate segmentation state while GIL is released; acquire for NumPy allocation.
        std::unique_ptr<Ward> engine;
        {
            py::gil_scoped_release release;
            engine = std::make_unique<Ward>(image.data(), w, h, min_pixels);
            engine->run(threshold, max_regions);
        }
        Image palette({engine->count(), 4});
        {
            py::gil_scoped_release release;
            engine->write(image.data(), w, h, refine_passes, labels.mutable_data(), palette.mutable_data());
            engine.reset();
        }
        return py::make_tuple(labels, palette);
    }, py::arg("image"), py::arg("threshold"), py::arg("min_pixels"), py::arg("max_regions"), py::arg("refine_passes") = 2);
}
