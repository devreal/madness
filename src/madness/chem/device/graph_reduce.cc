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

/// \file device/graph_reduce.cc
/// \brief DeviceGraph reductions to host results: matrix_inner, norms, kinetic matrix.

#include "graph_impl.h"

namespace madness {

void DeviceGraph::Impl::add_matrix_sink(MatrixEdge e, Res& r, double fac, const std::string& name) {
    g.add(ttg::make_tt([&r, fac](const mra::Key<3>& key, const mra::DenseTensor<double, 2>& m) {
        const double* data = m.current_view().data();
        std::lock_guard<std::mutex> lock(r.m);
        for (long i = 0; i < r.value.size(); ++i) r.value.ptr()[i] += fac * data[i];
    }, ttg::edges(e), ttg::edges(), name));
}

DeviceGraph::Result DeviceGraph::matrix_inner(Functions a, Functions b) {
    require_one_batch(*pimpl->get(a).layout, "DeviceGraph::matrix_inner");
    require_one_batch(*pimpl->get(b).layout, "DeviceGraph::matrix_inner");
    auto& g = pimpl->g;
    Impl::Res& r = pimpl->add_result(long(a.size), long(b.size));
    r.matrix.emplace();
    g.add(mra::make_matrix_inner<double, 3>(make_mrattg_functionset(a.size), make_mrattg_functionset(b.size), g.K,
                                            pimpl->get(a).compressed, pimpl->get(b).compressed, *r.matrix,
                                            "matrix_inner", device_procmap(), device_devicemap(), pimpl->inner_group));
    pimpl->add_matrix_sink(*r.matrix, r, 1.0, "matrix_inner_sink");
    return Result{int(pimpl->results.size()) - 1};
}

DeviceGraph::Result DeviceGraph::inner(Functions a, Functions b) {
    MADNESS_CHECK_THROW(a.size == b.size, "DeviceGraph::inner: a and b must have the same size");
    const DeviceLayout layout = pimpl->get(a).layout;
    MADNESS_CHECK_THROW(same_batches(*layout, *pimpl->get(b).layout), "DeviceGraph::inner: a and b in different batches");
    auto& g = pimpl->g;
    Impl::Res& r = pimpl->add_result(long(a.size), 0);
    r.vector.emplace();
    r.layout = layout;
    g.add(mra::make_inner<double, 3>(make_mrattg_functionset(layout), g.K, pimpl->get(a).compressed,
                                     pimpl->get(b).compressed, *r.vector, "inner", device_procmap(),
                                     device_devicemap(), pimpl->inner_group));
    g.add(make_batch_vector_sink(*r.vector, r.value, layout, "inner_sink"));
    return Result{int(pimpl->results.size()) - 1};
}

DeviceGraph::Result DeviceGraph::norms(Functions a) {
    auto& g = pimpl->g;
    const DeviceLayout layout = pimpl->get(a).layout;
    Impl::Res& r = pimpl->add_result(long(a.size), 0);
    r.vector.emplace();
    r.layout = layout;
    g.add(mra::make_norm(make_mrattg_functionset(layout), g.K, pimpl->get(a).compressed, *r.vector, "norm",
                         device_procmap(), device_devicemap()));
    g.add(make_batch_vector_sink(*r.vector, r.value, layout, "norm_sink"));
    return Result{int(pimpl->results.size()) - 1};
}

DeviceGraph::Result DeviceGraph::kinetic_matrix(Functions psi) {
    using T = double;
    auto& g = pimpl->g;
    const mra::size_type N = psi.size;
    require_one_batch(*pimpl->get(psi).layout, "DeviceGraph::kinetic_matrix");
    auto fns = make_mrattg_functionset(N);
    Impl::Res& r = pimpl->add_result(long(N), long(N));
    mra::BatchGroup compress_group;   // the three axes batch together
    // the three derivatives in one descent over the compressed orbitals, which are not reconstructed
    std::array<RecEdge, 3> d_e;
    g.add(mra::make_gradient_compressed<T, 3>(pimpl->domain(), fns, g.functiondata, g.K, pimpl->get(psi).compressed,
                                              d_e, "gradient", device_procmap(), device_devicemap()));
    for (mra::Dimension axis = 0; axis < 3; ++axis) {
        const std::string name = "d" + std::to_string(axis);
        NodeEdge dc_e;
        MatrixEdge m_e;
        g.add(mra::make_compress(fns, g.K, false, g.functiondata, d_e[axis], dc_e, "compress_" + name,
                                 device_procmap(), device_devicemap(), false, T{}, 0, T{}, compress_group));
        g.add(mra::make_matrix_inner<T, 3>(fns, fns, g.K, dc_e, dc_e, m_e, "inner_" + name,
                                           device_procmap(), device_devicemap(), pimpl->inner_group));
        pimpl->add_matrix_sink(m_e, r, 0.5, "sink_" + name);
    }
    return Result{int(pimpl->results.size()) - 1};
}

} // namespace madness
