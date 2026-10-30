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

#include "common.h"
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

/// sum_i occ_i psi_i^2 (cf. SCF::make_density), compressed, with the process map of `like`; psi is kept
functionT device_density(World& world, DeviceFunctions& psi, const tensorT& occ, const functionT& like) {
    using T = double;
    using RecEdge = ttg::Edge<mra::Key<3>, mra::FunctionsReconstructedNode<T, 3>>;
    MraGraph g;
    const mra::size_type N = psi.size();
    auto fns = make_mrattg_functionset(N);
    auto db = device_domain();
    Tensor<double> w(static_cast<long>(N), 1L);
    for (long i = 0; i < long(N); ++i) w(i, 0) = occ[i];
    NodeEdge psi_e, sq_c_e, rho_e;
    RecEdge sq_e;
    g.add(fv(psi).make_feed(g.control, psi_e, true, "feed_psi"));
    // squared straight from the compressed orbitals, without reconstructing them
    g.add(mra::make_square_compressed<T, 3>(db, fns, g.functiondata, g.K, psi_e, sq_e, "psi^2"));
    g.add(mra::make_compress(fns, g.K, false, g.functiondata, sq_e, sq_c_e, "compress_psi^2"));
    g.add(mra::make_transform<T, 3>(fns, make_mrattg_functionset(1), g.K, device_matrix(w), sq_c_e, rho_e, "sum_occ"));
    vecfuncT rho(1);
    rho[0].set_impl(like, false);
    g.add(mra::vmra::make_vmra_store(rho, rho_e, true /* add_leaves */));
    g.execute();
    check_device_tree(psi, "device_density feed(keep)");
    return rho[0];
}

} // namespace madness
