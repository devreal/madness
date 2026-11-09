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
/// tournament of the batches) in which every batch is in one block, and so receives one
/// contribution; a tile is a number of rounds.

#include <algorithm>
#include "graph_impl.h"

namespace madness {

/// the device memory a pair holds in an exchange graph, in trees of the average orbital's size.
/// Calibrated on water5 HF at k=8 (one A100, 34 GB pool): with ~290 nodes per orbital the
/// exchange slows down beyond ~100-110 pairs per graph (4.3 s at 96, 6.9 s at 128, 38 s at all
/// 325), i.e. ~18 trees per pair in half the pool; 20 leaves some margin.
static constexpr double exchange_pair_trees = 20.0;
/// the share of the device memory pools given to the pairs of one exchange graph
static constexpr double exchange_pool_share = 0.5;

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

std::size_t device_exchange_tile(World& world, DeviceFunctions& psi) {
    const double N = double(psi.size());
    const double npairs = N * (N + 1) / 2;
    // the size of the orbitals: their nonzero function-nodes (each holds (2k)^3 coefficients)
    double nodes = 0;
    fv(psi).for_each_local_node([&](const auto&, const auto& node) {
        if (!node.empty()) nodes += double(node.coeffs().sparsity().count_nonzero());
    });
    double pool = double(local_device_pool_bytes());
    world.gop.sum(nodes);
    world.gop.sum(pool);
    if (pool == 0 || nodes == 0 || npairs == 0) return 0;   // no device pool known: one graph

    // A pair holds, at the same time, its product, the convolution's intermediates and result, and
    // the directed products it contributes to K. Each is a tree about the size of an orbital, made
    // finer by the convolution (the potential extends further than the product).
    const double k = FunctionDefaults<3>::get_k();
    const double node_bytes = 8.0 * 8 * k * k * k;   // (2k)^3 doubles
    const double bytes_per_pair = exchange_pair_trees * (nodes / N) * node_bytes;
    // part of the pool stays with what the graph does not hold per pair: the resident orbitals,
    // K, the operator tables and the buffers PaRSEC keeps for reuse
    const double budget = exchange_pool_share * pool;
    const double fit = std::floor(budget / bytes_per_pair);
    // tiles of even size: as many as needed, each as full as the others (0: all in one graph)
    const double ntiles = std::ceil(npairs / std::max(fit, 1.0));
    const std::size_t tile = fit >= npairs ? 0 : std::size_t(std::ceil(npairs / ntiles));
    if (world.rank() == 0)
        print("device exchange:", std::size_t(npairs), "pairs in", std::size_t(ntiles), "tile(s) (",
              std::size_t(nodes / N), "nodes per orbital, about", std::lround(bytes_per_pair / 1e6), "MB per pair,",
              std::lround(pool / 1e9), "GB of device memory )");
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

/// one tile of the exchange with the orbitals in several batches: the blocks of `rounds`, their
/// contributions added to K (empty: K is their sum)
void exchange_tile_batched(World& world, DeviceFunctions& psi, DeviceOperatorCache& cache,
                           const std::vector<std::vector<std::pair<int, int>>>& rounds, DeviceFunctions& K) {
    const double thresh = FunctionDefaults<3>::get_thresh();
    const DeviceLayout L = layout_of(psi);
    const auto off = [&](int b) { return long(L->offset(mra::Batch(b))); };
    const auto size = [&](int b) { return long(L->batch_size(mra::Batch(b))); };

    // the blocks: the pairs (i, j) of a and b (global indices), computed by the group of `home`,
    // which alternates between a and b (as for matrix_inner)
    struct Block { int a, b, home, round; std::vector<std::pair<long, long>> pairs; };
    std::vector<Block> blocks;
    for (std::size_t k = 0; k < rounds.size(); ++k) {
        for (const auto& [a, b] : rounds[k]) {
            Block q{a, b, ((a + b) % 2 == 1) ? b : a, int(k), {}};
            for (long i = off(a); i < off(a) + size(a); ++i)
                for (long j = (a == b ? i : off(b)); j < off(b) + size(b); ++j) q.pairs.emplace_back(i, j);
            blocks.push_back(std::move(q));
        }
    }
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

    // the layouts of the sets per block and per directed set, on the group of their block's home
    const auto layout = [&](const std::vector<mra::size_type>& sizes, const std::vector<int>& homes) {
        std::vector<std::pair<int, int>> ranges;
        for (int h : homes) ranges.push_back(L->groups().batch_pes(mra::Batch(h)));
        return std::make_shared<const mra::BatchLayout>(sizes, mra::BatchGroups(ranges, L->num_pes()));
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
    std::vector<std::vector<std::vector<int>>> to_t(rounds.size(), std::vector<std::vector<int>>(dirs.size()));
    for (int q = 0; q < int(blocks.size()); ++q) {
        to_a[blocks[q].a].push_back(q);
        to_b[blocks[q].b].push_back(q);
    }
    for (int r = 0; r < int(dirs.size()); ++r) {
        to_s[dirs[r].s].push_back(r);
        v_to[dirs[r].block].push_back(r);
        to_t[blocks[dirs[r].block].round][r].push_back(dirs[r].t);
    }
    const auto targets = [](std::vector<std::vector<int>> to) {
        return [to = std::move(to)](int b) { return to[b]; };
    };

    DeviceGraph g(world);
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
    // the contributions of each round, one per batch, to their batches, summed there
    std::optional<DeviceGraph::Functions> sum;
    if (K.size() != 0) sum = g.feed(K, false);
    for (std::size_t k = 0; k < rounds.size(); ++k) {
        auto part = g.rebatch(kp, L, targets(to_t[k]), "x-to-K-" + std::to_string(k));
        sum = sum ? g.gaxpy(1.0, *sum, 1.0, part, 0.0) : part;
    }
    g.store(*sum, K);
    g.execute();
}

} // namespace

DeviceFunctions device_exchange(World& world, DeviceFunctions& psi, double lo, DeviceOperatorCache& cache,
                                std::size_t tile) {
    const long N = long(psi.size());
    const double thresh = FunctionDefaults<3>::get_thresh();
    if (!cache.exchange_op) cache.exchange_op = poperatorT(CoulombOperatorPtr(world, lo, thresh));
    if (tile == device_exchange_auto_tile) tile = device_exchange_tile(world, psi);

    const DeviceLayout L = layout_of(psi);
    if (L->num_batches() > 1) {
        // tiles of whole rounds, of at most `tile` pairs unless a round has more
        const auto rounds = exchange_rounds(L->num_batches());
        const auto npairs = [&](const std::vector<std::pair<int, int>>& round) {
            std::size_t n = 0;
            for (const auto& [a, b] : round) {
                const std::size_t na = L->batch_size(mra::Batch(a)), nb = L->batch_size(mra::Batch(b));
                n += (a == b) ? na * (na + 1) / 2 : na * nb;
            }
            return n;
        };
        DeviceFunctions K;
        for (std::size_t r0 = 0; r0 < rounds.size();) {
            std::size_t r1 = r0 + 1, count = npairs(rounds[r0]);
            while (r1 < rounds.size() && (tile == 0 || count + npairs(rounds[r1]) <= tile)) count += npairs(rounds[r1++]);
            exchange_tile_batched(world, psi, cache, {rounds.begin() + r0, rounds.begin() + r1}, K);
            r0 = r1;
        }
        return K;
    }

    std::vector<std::pair<long, long>> pairs;
    for (long i = 0; i < N; ++i)
        for (long j = i; j < N; ++j) pairs.emplace_back(i, j);
    if (tile == 0) tile = pairs.size();

    DeviceFunctions K;
    for (std::size_t p0 = 0; p0 < pairs.size(); p0 += tile) {
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

        DeviceGraph g(world);
        auto f = g.feed(psi, true);
        // [ttg: transform x2 + multiply + compress/truncate]
        auto pair = g.multiply(g.transform(f, Ua, 0.0, "x-A"), g.transform(f, Ub, 0.0, "x-B"), thresh, "x-pair");
        // [ttg: nonstandard + convolution + compress/truncate], all pairs in one convolution
        auto v = g.convolution(cache.exchange_op, pair, thresh, &cache);
        // [ttg: transform x2 + multiply + compress + transform (sum over the sources)]
        auto kp = g.transform(g.multiply(g.transform(f, Uc, 0.0, "x-C"), g.transform(v, Uw, 0.0, "x-W"), 0.0, "x-dir"),
                              Us, 0.0, "x-sum");
        if (K.size() == 0) {
            g.store(kp, K);
        } else {
            g.store(g.gaxpy(1.0, g.feed(K, false), 1.0, kp, 0.0), K);
        }
        g.execute();
    }
    return K;
}

} // namespace madness
