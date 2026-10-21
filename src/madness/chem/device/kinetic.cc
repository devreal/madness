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

/// \file device/kinetic.cc
/// \brief Kinetic-energy matrix on MRA-TTG.

#include "common.h"
#include <mutex>

namespace madness {

/// 1/2 sum_axis <d psi_i|d psi_j> (cf. SCF::kinetic_energy_matrix with the abgv derivative); psi is kept
Tensor<double> device_kinetic_matrix(World& world, DeviceFunctions& psi) {
    using T = double;
    using RecEdge = ttg::Edge<mra::Key<3>, mra::FunctionsReconstructedNode<T, 3>>;
    MraGraph g;
    const mra::size_type N = psi.size();
    auto fns = make_mrattg_functionset(N);
    auto db = device_domain();
    NodeEdge psi_e;
    RecEdge psi_rec_e;
    std::array<RecEdge, 3> d_e;
    std::array<NodeEdge, 3> dc_e;
    std::array<ttg::Edge<mra::Key<3>, mra::DenseTensor<T, 2>>, 3> m_e;
    Tensor<double> result(static_cast<long>(N), static_cast<long>(N));
    std::mutex m;
    g.add(fv(psi).make_feed(g.control, psi_e, true, "feed_psi"));
    g.add(mra::make_reconstruct(fns, g.K, false, g.functiondata, psi_e, psi_rec_e, "reconstruct_psi"));
    for (mra::Dimension axis = 0; axis < 3; ++axis) {
        const std::string name = "d" + std::to_string(axis);
        g.add(mra::make_derivative<T, 3>(db, fns, g.functiondata, g.K, axis, psi_rec_e, d_e[axis], name));
        g.add(mra::make_compress(fns, g.K, false, g.functiondata, d_e[axis], dc_e[axis], "compress_" + name));
        g.add(mra::make_matrix_inner<T, 3>(fns, fns, g.K, dc_e[axis], dc_e[axis], m_e[axis], "inner_" + name));
        g.add(ttg::make_tt([&result, &m, N](const mra::Key<3>& key, const mra::DenseTensor<T, 2>& r) {
            const T* data = r.current_view().data();
            std::lock_guard<std::mutex> lock(m);
            for (mra::size_type i = 0; i < N; ++i)
                for (mra::size_type j = 0; j < N; ++j) result(i, j) += 0.5*data[i*N + j];
        }, ttg::edges(m_e[axis]), ttg::edges(), "sink_" + name));
    }
    g.execute();
    world.gop.sum(result.ptr(), result.size());   // only the root's owner received it
    return result;
}

} // namespace madness
