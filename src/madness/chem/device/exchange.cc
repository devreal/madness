/*
  This file is part of MADNESS.

  Copyright (C) 2007,2010 Oak Ridge National Laboratory

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

  For more information please contact:

  Robert J. Harrison
  Oak Ridge National Laboratory
  One Bethel Valley Road
  P.O. Box 2008, MS-6367

  email: harrisonrj@ornl.gov
  tel:   865-241-3937
  fax:   865-572-0680
*/

/// \file device/exchange.cc
/// \brief The Hartree-Fock exchange K psi_i = sum_j psi_j G(psi_j psi_i) on MRA-TTG.
///
/// As the symmetric host Exchange (exchangeoperator.h), every pair i <= j is formed once:
/// P_ij = psi_i psi_j (truncated), V_ij = G P_ij (truncated), and V_ij contributes psi_j V_ij to
/// K psi_i and, for i != j, psi_i V_ij to K psi_j. MRA-TTG multiplies two sets elementwise, so
/// the factors are laid out by pair with transforms whose matrices select functions (0/1):
///
///   A = psi_{i(p)}, B = psi_{j(p)}              (transform, N -> T pairs)
///   V = G (A B)                                 (multiply, truncate; one operator for all pairs)
///   C = psi_{src(d)}, W = V_{p(d)}              (transform to the D directed products)
///   K_i += sum_{d: target(d) = i} C_d W_d       (multiply, transform D -> N)
///
/// The pairs are processed in tiles of one graph each, which bounds the device memory of the
/// products; K accumulates on the device across the tiles. Unlike mul_sparse, the products are
/// not screened.
///
/// With the orbitals in several batches, the pairs are formed in blocks of two batches a <= b
/// (the pairs i of a, j of b), each computed by the process group of one of them, to which the
/// other one's orbitals are sent: the steps above run per block, the transforms within each
/// block. A block's contributions to K psi_i of batch a and to K psi_j of batch b are sent to
/// their batches and summed there. The blocks are scheduled in rounds (a round-robin
/// tournament of the batches) in which every batch is in one block. A tile is a number of
/// rounds or, if a round has more pairs than a tile, a part of one: a chunk of the pairs of
/// each of its blocks.

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>
#include "graph_impl.h"

namespace madness {

/// the device memory a pair holds in an exchange graph, in trees of the average orbital's size.
/// Calibrated on water5 HF at k=8 (one A100, 34 GB pool): with ~290 nodes per orbital the
/// exchange slows down beyond ~100-110 pairs per graph (4.3 s at 96, 6.9 s at 128, 38 s at all
/// 325), i.e. ~18 trees per pair in half the pool. Water10 HF at k=8, thresh 1e-6 (four A100s,
/// half of 159 GB) with 253 nodes per orbital: 425 pairs per graph evict (49.6 s), 255 do not
/// (2.0 s), i.e. 23-38 trees per pair. A graph that evicts is 25-60 times slower, while graphs
/// of half the size cost only ~10 % more, so this errs on the large side.
static constexpr double exchange_pair_trees = 40.0;
/// the exchange tiles per graph, overlapped (see device_exchange; MRA_TTG_EXCHANGE_OVERLAP overrides
/// it, 1: one graph per tile)
static constexpr std::size_t exchange_overlap_tiles = 4;
/// the share of each device memory pool given to the pairs of one exchange graph, by default:
/// half on one process (the calibration above); less on several, where data sent between the
/// processes takes device memory too (staging for remote sends, received copies) and a GPU
/// manager left without memory it can evict can stall. MRA_TTG_EXCHANGE_POOL_SHARE overrides it.
static constexpr double exchange_pool_share_single = 0.5;
static constexpr double exchange_pool_share_multi = 0.3;

static double exchange_pool_share(World& world) {
    if (const char* s = std::getenv("MRA_TTG_EXCHANGE_POOL_SHARE"); s != nullptr && *s != '\0') {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end != s && v > 0.0 && v <= 1.0) return v;
        if (world.rank() == 0) print("MRA_TTG_EXCHANGE_POOL_SHARE must be in (0, 1], ignoring", s);
    }
    return world.size() > 1 ? exchange_pool_share_multi : exchange_pool_share_single;
}

/// bytes of the device memory pools of this process. PaRSEC takes its pool at startup (a share of
/// each device's memory, PARSEC_MCA_device_cuda_memory_use), so the free memory cannot be queried.
static std::size_t local_device_pool_bytes() {
    std::size_t bytes = 0;
#if defined(MRA_ENABLE_CUDA) || defined(MRA_ENABLE_HIP)
    for (int d = 0; d < ttg::device::num_devices(); ++d) {
        const int parsec_id = ttg_parsec::detail::ttg_device_to_parsec_device(
            ttg::device::Device(d, ttg::device::available_execution_space));
        auto* gpu = reinterpret_cast<parsec_device_gpu_module_t*>(parsec_mca_device_get(parsec_id));
        if (gpu != nullptr && gpu->mem_nb_blocks > 0) bytes += gpu->mem_block_size * std::size_t(gpu->mem_nb_blocks);
    }
#endif
    return bytes;
}

/// the nonzero function-nodes (each holds (2k)^3 coefficients) of the orbitals on this process
static double local_orbital_nodes(DeviceFunctions& psi) {
    double local_nodes = 0;
    fv(psi).for_each_local_node([&](const auto&, const auto& node) {
        if (!node.empty()) local_nodes += double(node.coeffs().sparsity().count_nonzero());
    });
    return local_nodes;
}

double device_nodes_per_orbital(World& world, DeviceFunctions& psi) {
    double nodes = local_orbital_nodes(psi);
    world.gop.sum(nodes);
    return psi.size() > 0 ? nodes / double(psi.size()) : 0.0;
}

/// the items (pairs, orbitals) of local_bytes_per_item each on this process that fit into
/// local_budget bytes on every process: the process that fits the fewest decides (summing the pools
/// of all processes would assume a perfectly even distribution). -1: some process knows no device
/// pool. Collective.
static double items_that_fit(World& world, double local_bytes_per_item, double local_budget) {
    double fit = std::numeric_limits<double>::max();
    bool no_pool = false;
    if (local_bytes_per_item > 0) {
        if (local_budget <= 0) no_pool = true;
        else fit = std::floor(local_budget / local_bytes_per_item);
    }
    double any_no_pool = no_pool ? 1.0 : 0.0;
    world.gop.max(any_no_pool);
    world.gop.min(fit);
    return any_no_pool > 0 ? -1.0 : fit;
}

std::size_t device_exchange_tile(World& world, DeviceFunctions& psi, double min_nodes_per_orbital) {
    const double N = double(psi.size());
    const double npairs = N * (N + 1) / 2;
    // the size of the orbitals on this process
    double local_nodes = local_orbital_nodes(psi);
    const double local_pool = double(local_device_pool_bytes());
    double nodes = local_nodes;
    world.gop.sum(nodes);
    if (nodes == 0 || npairs == 0) return 0;
    // orbitals not yet refined for this threshold: size the pairs for the orbitals they will
    // become, keeping each process's share of the nodes
    const bool floored = min_nodes_per_orbital * N > nodes;
    if (floored) {
        const double scale = min_nodes_per_orbital * N / nodes;
        local_nodes *= scale;
        nodes *= scale;
    }

    // A pair holds, at the same time, its product, the convolution's intermediates and result, and
    // the directed products it contributes to K. Each is a tree about the size of an orbital, made
    // finer by the convolution (the potential extends further than the product). The trees of a
    // pair are distributed over the processes like the orbitals, so this process holds its share
    // of the nodes of every pair.
    const double k = FunctionDefaults<3>::get_k();
    const double node_bytes = 8.0 * 8 * k * k * k;   // (2k)^3 doubles
    const double local_bytes_per_pair = exchange_pair_trees * (local_nodes / N) * node_bytes;
    // part of the pool stays with what the graph does not hold per pair: the resident orbitals,
    // K, the operator tables and the buffers PaRSEC keeps for reuse
    const double share = exchange_pool_share(world);
    const double fit = items_that_fit(world, local_bytes_per_pair, share * local_pool);
    if (fit < 0) return 0;   // no device pool known: one graph
    // for the report: the smallest pool and the largest share of the nodes of a process
    double min_pool = local_pool, max_nodes = local_nodes;
    world.gop.min(min_pool);
    world.gop.max(max_nodes);
    // tiles of even size: as many as needed, each as full as the others (0: all in one graph)
    const double ntiles = std::ceil(npairs / std::max(fit, 1.0));
    const std::size_t tile = fit >= npairs ? 0 : std::size_t(std::ceil(npairs / ntiles));
    if (world.rank() == 0)
        print("device exchange:", std::size_t(npairs), "pairs in", std::size_t(ntiles), "tile(s) (",
              std::size_t(nodes / N), floored ? "nodes per orbital (expected; fewer now), up to" : "nodes per orbital, up to", std::lround(100.0 * max_nodes / nodes),
              "% of them on one process, about", std::lround(exchange_pair_trees * (nodes / N) * node_bytes / 1e6),
              "MB per pair; ", std::lround(100 * share), "% of the smallest device pool,",
              std::lround(min_pool / 1e9), "GB )");
    return tile;
}

/// the device memory an orbital holds in a BSH residual graph, in trees of the average orbital's
/// size: its Fock transform, the shifted V psi, its non-standard form, the convolution's result
/// (much finer than the orbital: the operator spreads every node over its neighbors) and the
/// residual. Calibrated on water20 HF at k=8, thresh 1e-6 (four A100s, half of 159 GB), in the
/// second iteration after k=6: 50 orbitals per graph evict (32.8 s, against 39 s for all 100 in one
/// graph), 25 do not (7.4 s). Sized together with the expected growth of the orbitals at the start
/// of a protocol (device_exchange_unrefined_growth, 697 instead of ~100 nodes per orbital for
/// water20), which picks 4 tiles of 25; 250 trees picked 8 tiles, ~2-4 s slower per iteration.
/// Without that growth, e.g., restarting from orbitals already converged for this threshold, the
/// actual node counts may pick tiles that evict.
static constexpr double bsh_orbital_trees = 120.0;

bool device_spread_tiles() {
    static const bool spread = [] {
        const char* v = std::getenv("MRA_TTG_SPREAD_TILES");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return spread;
}

std::size_t device_bsh_tile(World& world, DeviceFunctions& psi, double min_nodes_per_orbital) {
    const double N = double(psi.size());
    double local_nodes = local_orbital_nodes(psi);
    const double local_pool = double(local_device_pool_bytes());
    double nodes = local_nodes;
    world.gop.sum(nodes);
    if (nodes == 0 || N == 0) return 0;
    // orbitals not yet refined for this threshold: sized for the orbitals they will become
    const bool floored = min_nodes_per_orbital * N > nodes;
    if (floored) {
        const double scale = min_nodes_per_orbital * N / nodes;
        local_nodes *= scale;
        nodes *= scale;
    }
    const double k = FunctionDefaults<3>::get_k();
    const double node_bytes = 8.0 * 8 * k * k * k;   // (2k)^3 doubles
    // the trees of an orbital are distributed over the processes like the orbitals
    const double local_bytes_per_orbital = bsh_orbital_trees * (local_nodes / N) * node_bytes;
    const double share = exchange_pool_share(world);
    const double fit = items_that_fit(world, local_bytes_per_orbital, share * local_pool);
    if (fit < 0) return 0;   // no device pool known: one graph
    double min_pool = local_pool;
    world.gop.min(min_pool);
    const double ntiles = std::ceil(N / std::max(fit, 1.0));
    const std::size_t tile = fit >= N ? 0 : std::size_t(std::ceil(N / ntiles));
    if (world.rank() == 0)
        print("device BSH residual:", std::size_t(N), "orbitals in", std::size_t(ntiles), "tile(s) (",
              std::size_t(nodes / N), floored ? "nodes per orbital (expected; fewer now), about" : "nodes per orbital, about",
              std::lround(bsh_orbital_trees * (nodes / N) * node_bytes / 1e6), "MB per orbital; ",
              std::lround(100 * share), "% of the smallest device pool,", std::lround(min_pool / 1e9), "GB )");
    return tile;
}

namespace {

/// the blocks (a, b), a <= b, of the pairs of nb batches in rounds, every batch in one block of
/// each round: a round-robin tournament (circle method). With nb odd, the batch paired with the
/// dummy nb does its own block (a, a); with nb even, the blocks (a, a) form a last round.
std::vector<std::vector<std::pair<int, int>>> exchange_rounds(int nb) {
    const int n = nb + nb % 2;
    std::vector<int> ring(n);
    for (int x = 0; x < n; ++x) ring[x] = x;
    std::vector<std::vector<std::pair<int, int>>> rounds;
    for (int r = 0; r < n - 1; ++r) {
        std::vector<std::pair<int, int>> round;
        for (int k = 0; k < n / 2; ++k) {
            const int x = ring[k], y = ring[n - 1 - k];
            if (x == nb) round.emplace_back(y, y);
            else if (y == nb) round.emplace_back(x, x);
            else round.emplace_back(std::min(x, y), std::max(x, y));
        }
        rounds.push_back(std::move(round));
        std::rotate(ring.begin() + 1, ring.end() - 1, ring.end());   // all but the first
    }
    if (nb % 2 == 0) {
        std::vector<std::pair<int, int>> diagonal;
        for (int a = 0; a < nb; ++a) diagonal.emplace_back(a, a);
        rounds.push_back(std::move(diagonal));
    }
    return rounds;
}

/// pairs (i, j) of the orbitals (global indices) of the block of batches a <= b
struct PairChunk { int a, b; std::vector<std::pair<long, long>> pairs; };

/// one tile of the exchange with the orbitals in several batches: the pairs of `chunks`, their
/// contributions added to K (empty: K is their sum, and every batch must be in a chunk)
void exchange_tile_batched(World& world, DeviceFunctions& psi, DeviceOperatorCache& cache,
                           const std::vector<PairChunk>& chunks, DeviceFunctions& K) {
    const double thresh = FunctionDefaults<3>::get_thresh();
    const DeviceLayout L = layout_of(psi);
    const auto off = [&](int b) { return long(L->offset(mra::Batch(b))); };
    const auto size = [&](int b) { return long(L->batch_size(mra::Batch(b))); };

    // the blocks: the pairs (i, j) of a and b, computed by the group of `home`, which
    // alternates between a and b (as for matrix_inner)
    struct Block { int a, b, home; std::vector<std::pair<long, long>> pairs; };
    std::vector<Block> blocks;
    for (const PairChunk& c : chunks) blocks.push_back({c.a, c.b, ((c.a + c.b) % 2 == 1) ? c.b : c.a, c.pairs});
    // the directed products of each block by the batch of their targets: (target, source, pair
    // in the block). A block of one batch has one set of them, a block of two one per batch.
    struct Directed { int block, t, s; std::vector<std::array<long, 3>> d; };
    std::vector<Directed> dirs;
    for (int q = 0; q < int(blocks.size()); ++q) {
        const Block& B = blocks[q];
        Directed da{q, B.a, B.b, {}}, db{q, B.b, B.a, {}};
        for (long p = 0; p < long(B.pairs.size()); ++p) {
            const auto [i, j] = B.pairs[p];
            da.d.push_back({i, j, p});
            if (i != j) (B.a == B.b ? da : db).d.push_back({j, i, p});
        }
        dirs.push_back(std::move(da));
        if (B.a != B.b) dirs.push_back(std::move(db));
    }

    // the layouts of the sets per block and per directed set, where their block's home batch is
    const auto layout = [&](const std::vector<mra::size_type>& sizes, const std::vector<int>& homes) {
        return std::make_shared<const mra::BatchLayout>(
            sizes, L->groups().placed_as(std::vector<mra::Batch>(homes.begin(), homes.end())));
    };
    std::vector<mra::size_type> s_a, s_b, s_p, s_s, s_v, s_d, s_t;
    std::vector<int> h_q, h_r;
    for (const Block& B : blocks) {
        s_a.push_back(size(B.a));
        s_b.push_back(size(B.b));
        s_p.push_back(B.pairs.size());
        h_q.push_back(B.home);
    }
    for (const Directed& D : dirs) {
        s_s.push_back(size(D.s));
        s_v.push_back(blocks[D.block].pairs.size());
        s_d.push_back(D.d.size());
        s_t.push_back(size(D.t));
        h_r.push_back(blocks[D.block].home);
    }
    const DeviceLayout LA = layout(s_a, h_q), LB = layout(s_b, h_q), LP = layout(s_p, h_q);
    const DeviceLayout LS = layout(s_s, h_r), LV = layout(s_v, h_r), LD = layout(s_d, h_r), LT = layout(s_t, h_r);

    // the matrices selecting the factors, within each block or directed set
    std::vector<Tensor<double>> Ua, Ub, Uc, Uw, Us;
    for (const Block& B : blocks) {
        const long T = long(B.pairs.size());
        Ua.emplace_back(size(B.a), T);
        Ub.emplace_back(size(B.b), T);
        for (long p = 0; p < T; ++p) {
            Ua.back()(B.pairs[p].first - off(B.a), p) = 1.0;
            Ub.back()(B.pairs[p].second - off(B.b), p) = 1.0;
        }
    }
    for (const Directed& D : dirs) {
        const long nd = long(D.d.size());
        Uc.emplace_back(size(D.s), nd);
        Uw.emplace_back(long(blocks[D.block].pairs.size()), nd);
        Us.emplace_back(nd, size(D.t));
        for (long d = 0; d < nd; ++d) {
            const auto [target, source, p] = D.d[d];
            Uc.back()(source - off(D.s), d) = 1.0;
            Uw.back()(p, d) = 1.0;
            Us.back()(d, target - off(D.t)) = 1.0;
        }
    }
    // where the nodes of a batch go: the blocks and directed sets that use it
    std::vector<std::vector<int>> to_a(L->num_batches()), to_b(L->num_batches()), to_s(L->num_batches()),
        v_to(blocks.size());
    for (int q = 0; q < int(blocks.size()); ++q) {
        to_a[blocks[q].a].push_back(q);
        to_b[blocks[q].b].push_back(q);
    }
    for (int r = 0; r < int(dirs.size()); ++r) {
        to_s[dirs[r].s].push_back(r);
        v_to[dirs[r].block].push_back(r);
    }
    const auto targets = [](std::vector<std::vector<int>> to) {
        return [to = std::move(to)](int b) { return to[b]; };
    };

    DeviceGraph g(world, "exchange");
    auto f = g.feed(psi, true);
    // [ttg: rebatch x2 + transform x2 + multiply + compress/truncate], per block
    auto A = g.transform(g.rebatch(f, LA, targets(to_a), "x-to-A"), Ua, LP, 0.0, "x-A");
    auto Bf = g.transform(g.rebatch(f, LB, targets(to_b), "x-to-B"), Ub, LP, 0.0, "x-B");
    auto pair = g.multiply(A, Bf, thresh, "x-pair");
    // [ttg: nonstandard + convolution + compress/truncate], all pairs in one convolution
    auto v = g.convolution(cache.exchange_op, pair, thresh, &cache);
    // [ttg: rebatch x2 + transform x2 + multiply + compress + transform], per directed set
    auto C = g.transform(g.rebatch(f, LS, targets(to_s), "x-to-C"), Uc, LD, 0.0, "x-C");
    auto W = g.transform(g.rebatch(v, LV, targets(v_to), "x-to-W"), Uw, LD, 0.0, "x-W");
    auto kp = g.transform(g.multiply(C, W, 0.0, "x-dir"), Us, LT, 0.0, "x-sum");
    // the contributions to their batches, summed there (and to K so far)
    std::vector<std::pair<DeviceGraph::Functions, std::function<int(int)>>> parts;
    parts.emplace_back(kp, [targets = std::vector<int>([&] {
        std::vector<int> t;
        for (const Directed& D : dirs) t.push_back(D.t);
        return t;
    }())](int r) { return targets[r]; });
    if (K.size() != 0) parts.emplace_back(g.feed(K, false), [](int c) { return c; });
    g.store(g.sum_into(parts, L, "x-K"), K);
    g.execute();
}

} // namespace

DeviceFunctions device_exchange(World& world, DeviceFunctions& psi, double lo, DeviceOperatorCache& cache,
                                std::size_t tile, double min_nodes_per_orbital) {
    const long N = long(psi.size());
    const double thresh = FunctionDefaults<3>::get_thresh();
    if (!cache.exchange_op) cache.exchange_op = poperatorT(CoulombOperatorPtr(world, lo, thresh));
    if (tile == device_exchange_auto_tile) tile = device_exchange_tile(world, psi, min_nodes_per_orbital);

    const DeviceLayout L = layout_of(psi);
    if (L->num_batches() > 1) {
        // The rounds, each cut into parts of at most `tile` pairs: every block of the round
        // contributes a chunk of its pairs to each part (the first chunk is never empty, so the
        // first part, which starts K, reaches every batch). Consecutive parts form the tiles.
        const auto off = [&](int b) { return long(L->offset(mra::Batch(b))); };
        const auto size = [&](int b) { return long(L->batch_size(mra::Batch(b))); };
        std::vector<std::vector<PairChunk>> parts;
        for (const auto& round : exchange_rounds(L->num_batches())) {
            std::vector<PairChunk> blocks;
            std::size_t npairs = 0;
            for (const auto& [a, b] : round) {
                PairChunk q{a, b, {}};
                for (long i = off(a); i < off(a) + size(a); ++i)
                    for (long j = (a == b ? i : off(b)); j < off(b) + size(b); ++j) q.pairs.emplace_back(i, j);
                npairs += q.pairs.size();
                blocks.push_back(std::move(q));
            }
            // part k holds chunk k of every block, chunks of ceil(T/m) of a block's T pairs; m
            // grows until each part fits a tile, or the chunks are single pairs
            std::size_t max_block = 0;
            for (const PairChunk& q : blocks) max_block = std::max(max_block, q.pairs.size());
            const auto part_pairs = [&](std::size_t m, std::size_t k) {
                std::size_t n = 0;
                for (const PairChunk& q : blocks) {
                    const std::size_t c = (q.pairs.size() + m - 1) / m;
                    n += std::min(q.pairs.size(), (k + 1) * c) - std::min(q.pairs.size(), k * c);
                }
                return n;
            };
            std::size_t m = (tile == 0) ? 1 : std::max<std::size_t>(1, (npairs + tile - 1) / tile);
            while (tile != 0 && m < max_block && part_pairs(m, 0) > tile) ++m;
            for (std::size_t k = 0; k < m; ++k) {
                std::vector<PairChunk> part;
                for (const PairChunk& q : blocks) {
                    const std::size_t c = (q.pairs.size() + m - 1) / m;   // chunk size
                    const std::size_t p0 = std::min(k * c, q.pairs.size()), p1 = std::min(p0 + c, q.pairs.size());
                    if (p0 < p1) part.push_back({q.a, q.b, {q.pairs.begin() + p0, q.pairs.begin() + p1}});
                }
                if (!part.empty()) parts.push_back(std::move(part));
            }
        }
        const auto npairs = [](const std::vector<PairChunk>& part) {
            std::size_t n = 0;
            for (const PairChunk& q : part) n += q.pairs.size();
            return n;
        };
        DeviceFunctions K;
        for (std::size_t r0 = 0; r0 < parts.size();) {
            std::vector<PairChunk> chunks = parts[r0];
            std::size_t r1 = r0 + 1, count = npairs(parts[r0]);
            while (r1 < parts.size() && (tile == 0 || count + npairs(parts[r1]) <= tile)) {
                count += npairs(parts[r1]);
                chunks.insert(chunks.end(), parts[r1].begin(), parts[r1].end());
                ++r1;
            }
            exchange_tile_batched(world, psi, cache, chunks, K);
            r0 = r1;
        }
        return K;
    }

    std::vector<std::pair<long, long>> pairs;
    for (long i = 0; i < N; ++i)
        for (long j = i; j < N; ++j) pairs.emplace_back(i, j);
    if (tile == 0) tile = pairs.size();

    // Tiles per graph (MRA_TTG_EXCHANGE_OVERLAP, default exchange_overlap_tiles): within a graph,
    // a tile starts once the convolution of the previous one is complete (the norm reduction of
    // its V finishes only after every node of V), so its products and convolution run while the
    // previous tile forms its directed products and contributions to K, on GPUs that would
    // otherwise idle until that tile's last device finishes. Only one convolution, the bulk of a
    // tile's memory, is in flight at a time. The tiles' contributions are summed in the graph; a
    // graph ends with a fence, as a tile used to.
    const std::size_t ntiles = (pairs.size() + tile - 1) / tile;
    const std::size_t per_graph = std::max<std::size_t>(1, [] {
        const char* v = std::getenv("MRA_TTG_EXCHANGE_OVERLAP");
        return v ? std::size_t(std::max(1L, std::atol(v))) : exchange_overlap_tiles;
    }());
    // The tiles are contiguous rows of pairs, not spread over all orbitals as the BSH residual's
    // (device_spread_tiles): a tile's functions are stored together, node by node, so a tile of
    // pairs of every orbital spans the whole volume and its transforms and products work on many
    // more nodes. Dealing the pairs to the tiles (water20 at k=8, 4 A100s) balanced the devices
    // slightly better but took 23% more kernel time and twice the transfers between devices, and
    // evicted: the exchange went from 131 to 210 s over the protocol.

    DeviceFunctions K;
    for (std::size_t t0 = 0; t0 < ntiles; t0 += per_graph) {
        DeviceGraph g(world, "exchange");
        auto f = g.feed(psi, true);
        std::optional<DeviceGraph::Functions> acc;   // the contributions of this graph's tiles so far
        std::optional<DeviceGraph::Result> prev_v;   // the norms of the latest tile's V
        for (std::size_t t = t0; t < std::min(ntiles, t0 + per_graph); ++t) {
            const std::size_t p0 = t * tile;
            const long T = long(std::min(tile, pairs.size() - p0));
            // the directed products of this tile: (target, source, pair)
            struct Directed { long target, source, pair; };
            std::vector<Directed> dirs;
            for (long p = 0; p < T; ++p) {
                const auto [i, j] = pairs[p0 + p];
                dirs.push_back({i, j, p});
                if (i != j) dirs.push_back({j, i, p});
            }
            const long D = long(dirs.size());
            Tensor<double> Ua(N, T), Ub(N, T), Uc(N, D), Uw(T, D), Us(D, N);
            for (long p = 0; p < T; ++p) {
                Ua(pairs[p0 + p].first, p) = 1.0;
                Ub(pairs[p0 + p].second, p) = 1.0;
            }
            for (long d = 0; d < D; ++d) {
                Uc(dirs[d].source, d) = 1.0;
                Uw(dirs[d].pair, d) = 1.0;
                Us(d, dirs[d].target) = 1.0;
            }
            // a selection from the orbitals: right away, or once the V whose norms are `after` is
            // complete (the orbitals' nodes wait in the transform until then)
            const auto select = [&](const Tensor<double>& U, std::optional<DeviceGraph::Result> after,
                                    const char* name) {
                if (!after) return g.transform(f, U, 0.0, name);
                auto m = g.decide(*after, [U](const Tensor<double>&) { return U; });
                return g.transform(f, m, std::size_t(U.dim(1)), 0.0);
            };
            // [ttg: transform x2 + multiply + compress/truncate], after the previous tile's V
            auto pair = g.multiply(select(Ua, prev_v, "x-A"), select(Ub, prev_v, "x-B"), thresh, "x-pair");
            // [ttg: nonstandard + convolution + compress/truncate], all pairs in one convolution
            auto v = g.convolution(cache.exchange_op, pair, thresh, &cache);
            prev_v = g.norms(v);
            // [ttg: transform x2 + multiply + compress + transform (sum over the sources)]; the
            // sources (2T functions) are selected once V is complete, not held through the convolution
            auto kp = g.transform(g.multiply(select(Uc, prev_v, "x-C"), g.transform(v, Uw, 0.0, "x-W"), 0.0, "x-dir"),
                                  Us, 0.0, "x-sum");
            acc = acc ? g.gaxpy(1.0, *acc, 1.0, kp, 0.0) : kp;
        }
        if (K.size() == 0) {
            g.store(*acc, K);
        } else {
            g.store(g.gaxpy(1.0, g.feed(K, false), 1.0, *acc, 0.0), K);
        }
        g.execute();
    }
    return K;
}

} // namespace madness
