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

/// \file device/potential.cc
/// \brief Local potential times the orbitals, and the density, on MRA-TTG.

#include "graph_impl.h"
#include <mutex>

namespace madness {

/// Vpsi_i = v psi_i for one (host) function v, compressed and truncated with tol unless tol == 0;
/// psi is kept, Vpsi receives the result
void device_apply_local_potential(World& world, const functionT& v, DeviceFunctions& psi,
                                  DeviceFunctions& Vpsi, double tol) {
    DeviceGraph g(world);
    g.store(g.apply_local_potential(v, g.feed(psi, true), tol), Vpsi);
    g.execute();
}

/// sum_i occ_i psi_i^2 (cf. SCF::make_density), compressed: one function, on all processes.
/// With the orbitals in several batches, each batch's group forms its partial density
/// sum_{i in b} occ_i psi_i^2, and the partial densities are summed into one function (one
/// function's worth of nodes moving from each group).
DeviceGraph::Functions DeviceGraph::density(Functions psi, const tensorT& occ) {
    using T = double;
    auto& g = pimpl->g;
    const mra::size_type N = psi.size;
    MADNESS_CHECK_THROW(occ.size() >= long(N), "DeviceGraph::density: an occupation for every function");
    const DeviceLayout layout = pimpl->get(psi).layout;
    const mra::Batch nb = layout->num_batches();
    auto fns = make_mrattg_functionset(layout);
    NodeEdge sq_c_e, rho_e;
    RecEdge sq_e;
    // squared straight from the compressed orbitals, without reconstructing them
    g.add(mra::make_square_compressed<T, 3>(pimpl->domain(), fns, g.functiondata, g.K, pimpl->get(psi).compressed,
                                            sq_e, "psi^2", device_procmap(fns), device_devicemap(fns)));
    g.add(mra::make_compress(fns, g.K, false, g.functiondata, sq_e, sq_c_e, "compress_psi^2",
                             device_procmap(fns), device_devicemap(fns)));
    // the occupations as one column per batch: block-diagonal in the batches, one function each
    Tensor<double> w(static_cast<long>(N), static_cast<long>(nb));
    for (mra::Batch b = 0; b < nb; ++b)
        for (mra::size_type l = 0; l < layout->batch_size(b); ++l) w(layout->global_index(b, l), b) = occ[layout->global_index(b, l)];
    const DeviceLayout parts = std::make_shared<const mra::BatchLayout>(
                                 std::vector<mra::size_type>(nb, 1), layout->groups());
    auto fns_parts = make_mrattg_functionset(parts);
    g.add(mra::make_transform<T, 3>(fns, fns_parts, g.K, device_matrix(w), sq_c_e, rho_e, "sum_occ",
                                    device_procmap(fns), device_devicemap(fns)));
    if (nb == 1) return pimpl->add_functions(rho_e, one_batch(1));
    // the partial densities summed into one function on all processes
    const DeviceLayout one = one_batch(1);
    NodeEdge sum = pimpl->add_sum_into({{rho_e, parts, [](mra::Batch) { return 0; }}}, one, "rho-sum");
    return pimpl->add_functions(sum, one);
}

/// sum_i occ_i psi_i^2 (cf. SCF::make_density), compressed, with the process map of `like`; psi is kept
functionT device_density(World& world, DeviceFunctions& psi, const tensorT& occ, const functionT& like) {
    DeviceGraph g(world);
    functionT rho;
    g.store(g.density(g.feed(psi, true), occ), rho, like);
    g.execute();
    return rho;
}

} // namespace madness
