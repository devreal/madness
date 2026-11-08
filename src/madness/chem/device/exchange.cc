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

DeviceFunctions device_exchange(World& world, DeviceFunctions& psi, double lo, DeviceOperatorCache& cache,
                                std::size_t tile) {
    const long N = long(psi.size());
    const double thresh = FunctionDefaults<3>::get_thresh();
    if (!cache.exchange_op) cache.exchange_op = poperatorT(CoulombOperatorPtr(world, lo, thresh));

    std::vector<std::pair<long, long>> pairs;
    for (long i = 0; i < N; ++i)
        for (long j = i; j < N; ++j) pairs.emplace_back(i, j);
    if (tile == device_exchange_auto_tile) tile = device_exchange_tile(world, psi);
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
