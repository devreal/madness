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

/// \file device/bsh_resident.cc
/// \brief The device-resident BSH residual (see bsh.cc for SCF::apply_bsh_mrattg).

#include <algorithm>
#include "common.h"

namespace madness {

/// BSH residual on the device: with fockp the Fock matrix minus eps on the diagonal,
///   rneg = G(-2 (Vpsi - psi fockp)) - psi,   rnorm_i = ||rneg_i||
/// i.e., the negative of the residual r = psi - new psi. psi is kept, Vpsi is consumed.
///
/// The orbitals are processed in tiles, one graph each, which bounds the device memory of the
/// convolution. A tile is a part of every batch of the orbitals (as many parts of each batch as
/// there are tiles), so that it is in the batches of the orbitals, smaller: the transforms select
/// its parts within each batch, without communication. A part is every ntiles-th orbital of the
/// batch (device_spread_tiles: a tile of neighboring localized orbitals would lie on a few devices)
/// or a contiguous range. A tile computes
///   Vpsi'_t = -2 Vpsi_t + 2 psi fockp(:, t),   rneg_t = G_t(Vpsi'_t) - psi_t
/// with Vpsi_t, psi_t selected by 0/1 matrices, and places rneg_t into rneg (a transform to the N
/// functions, the others empty, summed with the tiles so far).
DeviceFunctions device_bsh_residual(World& world, DeviceFunctions& psi, DeviceFunctions& Vpsi,
                                    const Tensor<double>& fockp, const tensorT& eps,
                                    const CalculationParameters& param, std::vector<double>& rnorm,
                                    DeviceOperatorCache* cache, std::size_t tile) {
    const long N = long(psi.size());
    // in the orbitals' batches: each batch is computed by its process group; the Fock matrix must
    // be block-diagonal in the batches (canonical orbitals: diagonal)
    const DeviceLayout L = layout_of(psi);
    MADNESS_CHECK_THROW(same_batches(*L, *layout_of(Vpsi)), "device_bsh_residual: psi and Vpsi in different batches");
    const mra::Batch nb = L->num_batches();
    const auto off = [&](mra::Batch b) { return long(L->offset(b)); };
    const auto size = [&](mra::Batch b) { return long(L->batch_size(b)); };
    for (mra::Batch b = 0; b < nb; ++b)
        for (long i = 0; i < N; ++i) {
            if (i >= off(b) && i < off(b) + size(b)) continue;
            for (long j = off(b); j < off(b) + size(b); ++j)
                device_check(fockp(i, j) == 0.0, "device_bsh_residual: the Fock matrix couples orbitals of different batches");
        }
    const double thresh = FunctionDefaults<3>::get_thresh();
    std::vector<poperatorT> ops = SCF::make_bsh_operators(world, eps, param);

    if (tile == device_bsh_auto_tile) tile = device_bsh_tile(world, Vpsi);
    // tiles: every batch in as many parts, so none is smaller than the smallest batch
    long min_batch = N;
    for (mra::Batch b = 0; b < nb; ++b) min_batch = std::min(min_batch, size(b));
    const long ntiles = (tile == 0) ? 1 : std::clamp<long>((N + long(tile) - 1) / long(tile), 1, std::max<long>(min_batch, 1));

    DeviceOperatorCache local_cache;
    if (!cache) cache = &local_cache;
    if (ntiles > 1) {
        // the term tables of all orbitals' operators, which every tile's operator then finds its
        // terms in (made here, as each tile's operators alone would replace them with their own)
        mra::GaussianConvolutionOperator<double, 3> all(ops, cache->bsh_tables);
        cache->bsh_tables = all.term_tables();
    }

    const bool spread = device_spread_tiles();
    DeviceFunctions r;
    rnorm.assign(N, 0.0);
    for (long t = 0; t < ntiles; ++t) {
        const bool last = (t == ntiles - 1);
        // the orbitals of the tile in each batch b, ascending
        std::vector<std::vector<long>> orb(nb);
        std::vector<mra::size_type> sizes(nb);
        for (mra::Batch b = 0; b < nb; ++b) {
            if (spread)
                for (long o = off(b) + t; o < off(b) + size(b); o += ntiles) orb[b].push_back(o);
            else
                for (long o = off(b) + size(b) * t / ntiles; o < off(b) + size(b) * (t + 1) / ntiles; ++o)
                    orb[b].push_back(o);
            sizes[b] = mra::size_type(orb[b].size());
        }
        // the tile in the batches of the orbitals: on their process groups
        const DeviceLayout LT = (ntiles == 1) ? L : std::make_shared<const mra::BatchLayout>(sizes, L->groups());
        std::vector<Tensor<double>> F(nb), Sel(nb), Place(nb);
        std::vector<poperatorT> tops;
        for (mra::Batch b = 0; b < nb; ++b) {
            const long T = long(orb[b].size());
            F[b] = Tensor<double>(size(b), T);
            Sel[b] = Tensor<double>(size(b), T);
            Place[b] = Tensor<double>(T, size(b));
            for (long i = 0; i < size(b); ++i)
                for (long j = 0; j < T; ++j) F[b](i, j) = 2.0 * fockp(off(b) + i, orb[b][j]);
            for (long j = 0; j < T; ++j) {
                Sel[b](orb[b][j] - off(b), j) = 1.0;
                Place[b](j, orb[b][j] - off(b)) = 1.0;
                tops.push_back(ops[orb[b][j]]);   // in the order of the tile's functions
            }
        }

        DeviceGraph g(world, "bsh_residual");
        auto p = g.feed(psi, true);
        auto v = g.feed(Vpsi, !last);   // consumed by the last tile
        // [ttg: transform (+ transform) + gaxpy] Vpsi'_t = -2 Vpsi_t + 2 psi fockp(:, t)
        auto fp = g.transform(p, F, LT, 0.0, "fock_psi");
        auto vt = (ntiles == 1) ? v : g.transform(v, Sel, LT, 0.0, "Vpsi_tile");
        auto vp = g.gaxpy(-2.0, vt, 1.0, fp, 0.0);
        // [ttg: nonstandard + convolution + compress/truncate] one BSH operator per orbital
        auto conv = g.convolution(tops, vp, thresh, cache);
        // -r = new psi - psi. Not psi - new psi: gaxpy forwards an operand with scalar 1 unchanged
        // where the other one is empty, which would hand a kept psi node to the store of r (see
        // mra::FunctionVector)
        auto pt = (ntiles == 1) ? p : g.transform(p, Sel, LT, 0.0, "psi_tile");
        auto rt = g.gaxpy(1.0, conv, -1.0, pt, 0.0);
        auto norms = g.norms(rt);
        if (ntiles == 1) {
            g.store(rt, r);
        } else {
            // [ttg: transform + sum] placed into the N functions, added to the tiles so far
            std::vector<std::pair<DeviceGraph::Functions, std::function<int(int)>>> parts;
            parts.emplace_back(g.transform(rt, Place, L, 0.0, "place_r"), [](int b) { return b; });
            if (r.size() != 0) parts.emplace_back(g.feed(r, false), [](int b) { return b; });
            g.store(g.sum_into(parts, L, "r"), r);
        }
        g.execute();
        const Tensor<double> n = g.get(norms);   // per batch, at its offset in the tile
        for (mra::Batch b = 0; b < nb; ++b)
            for (long j = 0; j < long(orb[b].size()); ++j) rnorm[orb[b][j]] = n(long(LT->offset(b)) + j);
    }
    check_device_tree(r, "device_bsh_residual");
    return r;
}

} // namespace madness
