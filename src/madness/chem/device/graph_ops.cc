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

/// \file device/graph_ops.cc
/// \brief DeviceGraph operations producing functions: local potential, transform, concat.

#include "graph_impl.h"

namespace madness {

DeviceGraph::Functions DeviceGraph::load(const functionT& f) {
    auto& g = pimpl->g;
    pimpl->keepalive.push_back({f});   // the loader refers to it until the graph has run
    NodeEdge e;
    load_compressed_mrattg(pimpl->keepalive.back(), make_mrattg_functionset(1), g.K, g.functiondata, g.control, e,
                           g.tts, "host");
    return pimpl->add_functions(e, 1);
}

void DeviceGraph::store(Functions f, functionT& out, const functionT& like) {
    MADNESS_CHECK_THROW(f.size == 1, "DeviceGraph::store: one function can be stored to the host");
    pimpl->host_out.emplace_back(1);
    vecfuncT& v = pimpl->host_out.back();
    v[0].set_impl(like, false);
    pimpl->g.add(mra::vmra::make_vmra_store(v, pimpl->get(f).compressed, true /* add_leaves */));
    pimpl->host_targets.push_back(&out);
}

DeviceGraph::Functions DeviceGraph::gaxpy(double a, Functions x, double b, Functions y, double tol) {
    MADNESS_CHECK_THROW(x.size == y.size, "DeviceGraph::gaxpy: x and y must have the same size");
    auto& g = pimpl->g;
    auto fns = make_mrattg_functionset(x.size);
    NodeEdge out_e, trunc_e;
    g.add(mra::make_gaxpy(a, b, fns, g.K, pimpl->get(x).compressed, pimpl->get(y).compressed, out_e, "gaxpy"));
    if (tol <= 0.0) return pimpl->add_functions(out_e, x.size);
    g.add(mra::make_truncate(fns, g.K, tol, FunctionDefaults<3>::get_truncate_mode(),
                             FunctionDefaults<3>::get_cell_min_width(), out_e, trunc_e, "gaxpy-truncate"));
    return pimpl->add_functions(trunc_e, x.size);
}

DeviceGraph::Functions DeviceGraph::apply_local_potential(const functionT& v, Functions psi, double tol) {
    return apply_local_potential(load(v), psi, tol);
}

DeviceGraph::Functions DeviceGraph::apply_local_potential(Functions v, Functions psi, double tol) {
    MADNESS_CHECK_THROW(v.size == 1, "DeviceGraph::apply_local_potential: one potential applies to all functions");
    return multiply(v, psi, tol);
}

DeviceGraph::Functions DeviceGraph::multiply(Functions a, Functions b, double tol, const std::string& name) {
    using T = double;
    MADNESS_CHECK_THROW(a.size == b.size || a.size == 1, "DeviceGraph::multiply: a must have the size of b, or 1");
    auto& g = pimpl->g;
    const mra::size_type N = b.size;
    auto fns = make_mrattg_functionset(N);
    RecEdge prod_e;
    NodeEdge comp_e;
    // multiplied in compressed form: the factors are not reconstructed for this
    g.add(mra::make_multiply_compressed<T, 3>(pimpl->domain(), make_mrattg_functionset(a.size), fns, g.functiondata,
                                              g.K, pimpl->get(a).compressed, pimpl->get(b).compressed, prod_e,
                                              name));
    add_compress(g.tts, fns, g.K, g.functiondata, prod_e, comp_e, tol, name + "-compress");
    return pimpl->add_functions(comp_e, N);
}

namespace {
    /// out_j = sum_i in_i U(i,j) with U fixed or from an edge, truncated with tol unless tol == 0
    template <typename UT>
    NodeEdge add_transform(MraGraph& g, NodeEdge in, std::size_t N, std::size_t M, UT&& U, double tol,
                           const std::string& name = "transform") {
        auto fns_out = make_mrattg_functionset(M);
        NodeEdge out_e, trunc_e;
        g.add(mra::make_transform<double, 3>(make_mrattg_functionset(N), fns_out, g.K, std::forward<UT>(U), in, out_e,
                                             name));
        if (tol <= 0.0) return out_e;
        g.add(mra::make_truncate(fns_out, g.K, tol, FunctionDefaults<3>::get_truncate_mode(),
                                 FunctionDefaults<3>::get_cell_min_width(), out_e, trunc_e, "truncate"));
        return trunc_e;
    }
}

DeviceGraph::Functions DeviceGraph::transform(Functions in, const Tensor<double>& U, double tol, const std::string& name) {
    MADNESS_CHECK_THROW(U.dim(0) == long(in.size), "DeviceGraph::transform: U must have in.size rows");
    const std::size_t M = U.dim(1);
    return pimpl->add_functions(add_transform(pimpl->g, pimpl->get(in).compressed, in.size, M, device_matrix(U), tol, name), M);
}

DeviceGraph::Functions DeviceGraph::transform(Functions in, Matrix U, std::size_t M, double tol) {
    return pimpl->add_functions(add_transform(pimpl->g, pimpl->get(in).compressed, in.size, M, pimpl->matrix(U), tol), M);
}

DeviceGraph::Functions DeviceGraph::concat(Functions a, Functions b) {
    auto& g = pimpl->g;
    NodeEdge out_e;
    g.add(mra::make_concat<double, 3>(make_mrattg_functionset(a.size), make_mrattg_functionset(b.size), g.K,
                                      pimpl->get(a).compressed, pimpl->get(b).compressed, out_e, "concat",
                                      ttg::Void{}, ttg::Void{}, pimpl->concat_group));
    return pimpl->add_functions(out_e, a.size + b.size);
}

} // namespace madness
