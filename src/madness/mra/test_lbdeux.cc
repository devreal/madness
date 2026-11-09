/// Unit tests for the space-filling-curve load balancer (LoadBalanceDeux::load_balance_sfc, SFCPmap).
///
/// The functions are Slater cusps at different places and of different widths, so their trees
/// are deep in a few places and the cost is far from uniform in space.

#include <madness/mra/mra.h>
#include <madness/mra/lbdeux.h>
#include <madness/world/test_utilities.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace madness;

static const std::size_t D = 3;

/// c * exp(-a |r-R|), cusped at R
class Slater : public FunctionFunctorInterface<double, D> {
    double a, c;
    Vector<double, D> R;
public:
    Slater(double a, double c, const Vector<double, D>& R) : a(a), c(c), R(R) {}
    double operator()(const Vector<double, D>& r) const override {
        double r2 = 0.0;
        for (std::size_t i = 0; i < D; ++i) r2 += (r[i]-R[i]) * (r[i]-R[i]);
        return c * std::exp(-a * std::sqrt(r2));
    }
    std::vector<Vector<double,D>> special_points() const override { return {R}; }
};

/// cost of a node as in the SCF: 1 for a leaf, 8 for an interior node
struct Cost {
    double operator()(const Key<D>&, const FunctionNode<double,D>& node) const {
        return node.has_children() ? 8.0 : 1.0;
    }
};

static std::vector<Function<double,D>> make_functions(World& world) {
    std::vector<Vector<double,D>> centres = {
        {{ 0.0,  0.0,  0.0}},
        {{ 1.3, -0.7,  0.4}},
        {{-2.9,  1.1, -1.2}},
        {{ 3.1,  2.3, -0.6}},
    };
    std::vector<double> alpha = {1.0, 1.7, 2.5, 0.8};
    std::vector<Function<double,D>> v;
    for (std::size_t i = 0; i < centres.size(); ++i)
        v.emplace_back(FunctionFactory<double,D>(world)
                       .functor(std::shared_ptr<FunctionFunctorInterface<double,D>>(
                                    new Slater(alpha[i], 1.0 + 0.1*double(i), centres[i]))));
    reconstruct(world, v);
    return v;
}

/// The cost of the nodes of the functions on this process, summed per part of the map
static std::vector<double> local_part_costs(const std::vector<Function<double,D>>& v, const SFCPmap<D>& map,
                                            int nproc) {
    std::vector<double> cost(nproc*map.num_devices(), 0.0);
    const Cost costfn;
    for (const auto& f : v) {
        for (const auto& [key, node] : f.get_impl()->get_coeffs()) {
            cost[map.owner(key)*map.num_devices() + map.device(key)] += costfn(key, node);
        }
    }
    return cost;
}

/// The range of a box contains the ranges of its children, which tile it in child order.
int test_curve(World& world) {
    test_output t("SFCCode: nested, contiguous ranges");
    using C = SFCCode<D>;
    bool ok = true;
    for (Level n = 0; n < 6; ++n) {
        const Translation m = Translation(1) << n;
        for (Translation x = 0; x < m; x += std::max<Translation>(1, m/5)) {
            for (Translation y = 0; y < m; y += std::max<Translation>(1, m/3)) {
                for (Translation z = 0; z < m; z += std::max<Translation>(1, m/4)) {
                    const Key<D> key(n, Vector<Translation,D>{x, y, z});
                    const uint64_t b = C::code(key), e = b + C::extent(key);
                    uint64_t next = b;
                    // the children in curve order: the Morton order of their translation bits
                    std::vector<Key<D>> kids;
                    for (KeyChildIterator<D> kit(key); kit; ++kit) kids.push_back(kit.key());
                    std::sort(kids.begin(), kids.end(), [](const Key<D>& a, const Key<D>& c) {
                        return C::code(a) < C::code(c);
                    });
                    for (const auto& kid : kids) {
                        ok = ok && C::code(kid) == next && C::extent(kid)*8 == C::extent(key);
                        next = C::code(kid) + C::extent(kid);
                    }
                    ok = ok && next == e;
                }
            }
        }
    }
    // deeper than the curve resolves: the ancestor's position, extent one
    const Key<D> deep(C::max_level + 2, Vector<Translation,D>{5, 6, 7});
    ok = ok && C::code(deep) == C::code(deep.parent(2)) && C::extent(deep) == 1;
    t.checkpoint(ok, "children tile their parent's range");
    return t.end();
}

/// A cut shared by a group of processes (SFCPmap::group_part): the group's members and their
/// devices take the parts in contiguous blocks; a group of all processes gets (owner, device).
/// On a map made up here of 6 processes with 2 devices each, 12 parts tiling the curve.
int test_group_part(World& world) {
    test_output t("SFCPmap::group_part");
    const int nproc = 6, ndevice = 2, nparts = nproc*ndevice;
    std::vector<std::pair<uint64_t,int>> pieces;
    const uint64_t extent = SFCCode<D>::extent(Key<D>(0));
    for (int q = 0; q < nparts; ++q) pieces.emplace_back(q*(extent/nparts), q);
    const SFCPmap<D> map(pieces, nproc, ndevice);

    bool identity = true, ranges = true, blocks = true;
    const Translation m = 16;
    for (Translation x = 0; x < m; ++x)
        for (Translation y = 0; y < m; ++y)
            for (Translation z = 0; z < m; ++z) {
                const Key<D> key(4, Vector<Translation,D>{x, y, z});
                const int q = map.part_of(key);
                const auto [p, d] = map.group_part(key, nproc);
                identity = identity && p == map.owner(key) && d == map.device(key);
                for (int size : {1, 2, 3, 4, 5}) {
                    const auto [member, dev] = map.group_part(key, size);
                    ranges = ranges && member >= 0 && member < size && dev >= 0 && dev < ndevice;
                    // a divisor of nproc: each slot of the group takes nproc/size consecutive parts
                    if (nproc % size == 0) blocks = blocks && member*ndevice + dev == q/(nproc/size);
                }
            }
    t.checkpoint(identity, "a group of all processes: owner and device");
    t.checkpoint(ranges, "members and devices in range");
    t.checkpoint(blocks, "contiguous blocks of parts per member and device");
    return t.end();
}

/// Every process builds the same map, every key maps to a valid process and device, and
/// the cost is balanced across processes and devices.
int test_map(World& world, const std::vector<Function<double,D>>& v, int ndevice, double fac) {
    char title[80];
    snprintf(title, sizeof(title), "load_balance_sfc: %d devices per process, fac %.0f", ndevice, fac);
    test_output t(title);
    const int nproc = world.size();

    LoadBalanceDeux<D> lb(world);
    for (const auto& f : v) lb.add_tree(f, Cost(), false);
    world.gop.fence();
    std::shared_ptr<SFCPmap<D>> map = lb.load_balance_sfc(ndevice, fac);

    // the same on every process: a checksum over all boxes at level 5
    double check = 0.0;
    bool valid = true;
    const Translation m = 32;
    for (Translation x = 0; x < m; ++x)
        for (Translation y = 0; y < m; ++y)
            for (Translation z = 0; z < m; ++z) {
                const Key<D> key(5, Vector<Translation,D>{x, y, z});
                const int p = map->owner(key), d = map->device(key);
                valid = valid && p >= 0 && p < nproc && d >= 0 && d < ndevice;
                check += double((p*ndevice + d + 1) * ((x*m + y)*m + z + 1));
            }
    double cmin = check, cmax = check;
    world.gop.min(cmin);
    world.gop.max(cmax);
    t.checkpoint(valid, "every key maps to a valid process and device");
    t.checkpoint(cmin == cmax, "the map is the same on every process");

    // balance, as predicted by the load balancer
    const std::vector<double>& cost = map->costs();
    const double total = std::accumulate(cost.begin(), cost.end(), 0.0);
    const double mean = total/cost.size();
    const double maxc = *std::max_element(cost.begin(), cost.end());
    if (world.rank() == 0)
        printf("  %zu pieces, part costs: mean %.0f max %.0f (%.2fx)\n", map->num_pieces(), mean, maxc, maxc/mean);
    t.checkpoint(maxc/mean < 1.5, "the parts cost at most 1.5x the mean");

    // the costs the functions have under the map are the predicted ones (up to the
    // ancestors at the start of a piece, which go with the piece)
    std::vector<double> actual = local_part_costs(v, *map, nproc);
    world.gop.sum(actual.data(), actual.size());
    double dev = 0.0;
    for (std::size_t i = 0; i < cost.size(); ++i) dev = std::max(dev, std::abs(actual[i] - cost[i]));
    if (world.rank() == 0) printf("  max |actual - predicted| part cost %.0f\n", dev);
    t.checkpoint(dev < 0.1*mean, "the parts cost what the load balancer predicted");
    return t.end();
}

/// Functions moved onto the map are the same functions, and their nodes are where the map says.
int test_redistribute(World& world, std::vector<Function<double,D>>& v) {
    test_output t("redistribute onto an SFCPmap");
    std::vector<double> before = norm2s(world, v);
    const double ip_before = inner(v[0], v[1]);

    LoadBalanceDeux<D> lb(world);
    for (const auto& f : v) lb.add_tree(f, Cost(), false);
    world.gop.fence();
    std::shared_ptr<SFCPmap<D>> map = lb.load_balance_sfc(2, 2.0);
    auto oldpmap = FunctionDefaults<D>::get_pmap();
    FunctionDefaults<D>::redistribute(world, map);

    long misplaced = 0;
    for (const auto& f : v) {
        for (const auto& [key, node] : f.get_impl()->get_coeffs()) misplaced += (map->owner(key) != world.rank());
    }
    world.gop.sum(misplaced);
    t.checkpoint(misplaced == 0, "every node is on the process the map assigns it to");

    std::vector<double> after = norm2s(world, v);
    double err = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) err = std::max(err, std::abs(after[i] - before[i]));
    t.checkpoint(err, 1.e-13, "norms unchanged");
    t.checkpoint(std::abs(inner(v[0], v[1]) - ip_before), 1.e-13, "inner product unchanged");

    // operations on the redistributed functions
    Function<double,D> s = v[0] + v[1];
    s.compress();
    s.reconstruct();
    const double ns = s.norm2(), ref = std::sqrt(before[0]*before[0] + before[1]*before[1] + 2.0*ip_before);
    t.checkpoint(std::abs(ns - ref), 1.e-6, "sum computed on the new map");

    FunctionDefaults<D>::redistribute(world, oldpmap);
    return t.end();
}

int main(int argc, char** argv) {
    World& world = initialize(argc, argv);
    startup(world, argc, argv, true);

    FunctionDefaults<D>::set_k(6);
    FunctionDefaults<D>::set_thresh(1.e-5);
    FunctionDefaults<D>::set_cubic_cell(-8.0, 8.0);
    FunctionDefaults<D>::set_refine(true);
    FunctionDefaults<D>::set_initial_level(2);

    int success = 0;
    {
        std::vector<Function<double,D>> v = make_functions(world);
        success += test_curve(world);
        success += test_group_part(world);
        success += test_map(world, v, 1, 4.0);
        success += test_map(world, v, 3, 4.0);
        success += test_redistribute(world, v);
    }

    world.gop.fence();
    finalize();
    return success;
}
