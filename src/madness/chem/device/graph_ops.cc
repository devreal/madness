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
    pimpl->g.add(mra::vmra::make_vmra_store(v, pimpl->get(f).compressed, true /* add_leaves */,
                                            madness::TreeState::compressed, "store_host", pimpl->get(f).layout));
    pimpl->host_targets.push_back(&out);
}

DeviceGraph::Functions DeviceGraph::gaxpy(double a, Functions x, double b, Functions y, double tol) {
    MADNESS_CHECK_THROW(x.size == y.size, "DeviceGraph::gaxpy: x and y must have the same size");
    const DeviceLayout layout = pimpl->get(x).layout;
    MADNESS_CHECK_THROW(same_batches(*layout, *pimpl->get(y).layout), "DeviceGraph::gaxpy: x and y in different batches");
    auto& g = pimpl->g;
    auto fns = make_mrattg_functionset(layout);
    NodeEdge out_e, trunc_e;
    g.add(mra::make_gaxpy(a, b, fns, g.K, pimpl->get(x).compressed, pimpl->get(y).compressed, out_e, "gaxpy",
                          device_procmap(), device_devicemap()));
    if (tol <= 0.0) return pimpl->add_functions(out_e, layout);
    g.add(mra::make_truncate(fns, g.K, tol, FunctionDefaults<3>::get_truncate_mode(),
                             FunctionDefaults<3>::get_cell_min_width(), out_e, trunc_e, "gaxpy-truncate", device_procmap(), device_devicemap()));
    return pimpl->add_functions(trunc_e, layout);
}

DeviceGraph::Functions DeviceGraph::combine(const std::vector<double>& c, const std::vector<Functions>& f) {
    MADNESS_CHECK_THROW(f.size() >= 2 && c.size() == f.size(), "DeviceGraph::combine: two or more sets, a coefficient each");
    auto& g = pimpl->g;
    const std::size_t N = f.front().size;
    const DeviceLayout layout = pimpl->get(f.front()).layout;
    for (const auto& x : f) {
        MADNESS_CHECK_THROW(x.size == N, "DeviceGraph::combine: the sets must have the same size");
        MADNESS_CHECK_THROW(same_batches(*layout, *pimpl->get(x).layout), "DeviceGraph::combine: the sets in different batches");
    }
    auto fns = make_mrattg_functionset(layout);
    // a chain of gaxpys, none forwarding an input node: c0 f0 + c1 f1, then + c2 f2, ...
    NodeEdge acc = pimpl->get(f.front()).compressed;
    double ca = c.front();
    for (std::size_t k = 1; k < f.size(); ++k) {
        NodeEdge out_e;
        g.add(mra::make_gaxpy(ca, c[k], fns, g.K, acc, pimpl->get(f[k]).compressed, out_e, "combine",
                              device_procmap(), device_devicemap(), /*forward=*/false));
        acc = out_e;
        ca = 1.0;
    }
    return pimpl->add_functions(acc, layout);
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
    const DeviceLayout layout = pimpl->get(b).layout;
    auto& g = pimpl->g;
    NodeEdge a_e = pimpl->get(a).compressed;
    DeviceLayout a_layout = pimpl->get(a).layout;
    if (a.size == b.size) {
        MADNESS_CHECK_THROW(same_batches(*a_layout, *layout), "DeviceGraph::multiply: a and b in different batches");
    } else if (layout->num_batches() > 1) {
        // the single factor, held in batch 0, in every batch of b: one copy per process group
        MADNESS_CHECK_THROW(a_layout->num_batches() == 1, "DeviceGraph::multiply: a single factor must be in one batch");
        NodeEdge rep_e;
        g.add(mra::make_replicate<double, 3>(a_e, rep_e, layout->num_batches(), name + "-replicate",
                                             device_procmap(), device_devicemap()));
        a_e = rep_e;
        a_layout = std::make_shared<const mra::BatchLayout>(
                     std::vector<mra::size_type>(layout->num_batches(), 1), layout->groups());
    }
    const mra::size_type N = b.size;
    auto fns = make_mrattg_functionset(layout);
    RecEdge prod_e;
    NodeEdge comp_e;
    // multiplied in compressed form: the factors are not reconstructed for this
    g.add(mra::make_multiply_compressed<T, 3>(pimpl->domain(), make_mrattg_functionset(a_layout), fns, g.functiondata,
                                              g.K, a_e, pimpl->get(b).compressed, prod_e,
                                              name, device_procmap(), device_devicemap()));
    add_compress(g.tts, fns, g.K, g.functiondata, prod_e, comp_e, tol, name + "-compress");
    return pimpl->add_functions(comp_e, layout);
}

namespace {
    /// out_j = sum_i in_i U(i,j) with U fixed or from an edge (at the root of each batch, cut to
    /// it), truncated with tol unless tol == 0
    template <typename UT>
    NodeEdge add_transform(MraGraph& g, NodeEdge in, const DeviceLayout& in_layout, const DeviceLayout& out_layout,
                           UT&& U, double tol, const std::string& name) {
        auto fns_out = make_mrattg_functionset(out_layout);
        NodeEdge out_e, trunc_e;
        g.add(mra::make_transform<double, 3>(make_mrattg_functionset(in_layout), fns_out, g.K, std::forward<UT>(U), in,
                                             out_e, name, device_procmap(), device_devicemap()));
        if (tol <= 0.0) return out_e;
        g.add(mra::make_truncate(fns_out, g.K, tol, FunctionDefaults<3>::get_truncate_mode(),
                                 FunctionDefaults<3>::get_cell_min_width(), out_e, trunc_e, name + "-truncate",
                                 device_procmap(), device_devicemap()));
        return trunc_e;
    }

    /// The layout of the result of a transform of a set in `in` by an N x M matrix: with
    /// several batches, each transforms its own functions (U block-diagonal, M == N); a single
    /// batch of M otherwise.
    DeviceLayout transform_layout(const DeviceLayout& in, std::size_t M, const std::string& name) {
        if (in->num_batches() == 1) return one_batch(M);
        MADNESS_CHECK_THROW(M == in->num_functions(),
                            (name + ": with several batches, U must be square (block-diagonal in the batches)").c_str());
        return in;
    }
}

DeviceGraph::Functions DeviceGraph::transform(Functions in, const Tensor<double>& U, double tol, const std::string& name) {
    MADNESS_CHECK_THROW(U.dim(0) == long(in.size), "DeviceGraph::transform: U must have in.size rows");
    const DeviceLayout in_layout = pimpl->get(in).layout;
    const DeviceLayout out_layout = transform_layout(in_layout, U.dim(1), "DeviceGraph::transform");
    // MRA-TTG cuts U to each batch's block and refuses a U that couples batches
    return pimpl->add_functions(add_transform(pimpl->g, pimpl->get(in).compressed, in_layout, out_layout,
                                              device_matrix(U), tol, name), out_layout);
}

DeviceGraph::Functions DeviceGraph::transform(Functions in, Matrix U, std::size_t M, double tol) {
    const DeviceLayout in_layout = pimpl->get(in).layout;
    const DeviceLayout out_layout = transform_layout(in_layout, M, "DeviceGraph::transform");
    MatrixEdge U_e = pimpl->matrix(U);
    if (in_layout->num_batches() > 1) {
        // U arrives once, at the root of batch 0: send each batch's root its block
        MatrixEdge blocks;
        pimpl->g.add(ttg::make_tt([in_layout](const mra::Key<3>& key, const mra::DenseTensor<double, 2>& u) {
            const mra::size_type N = u.dim(0);
            const double* p = u.current_view().data();
            for (mra::Batch b = 0; b < in_layout->num_batches(); ++b) {
                const mra::size_type o = in_layout->offset(b), n = in_layout->batch_size(b);
                for (mra::size_type i = 0; i < N; ++i) {
                    if (i >= o && i < o + n) continue;
                    for (mra::size_type j = o; j < o + n; ++j)
                        MADNESS_CHECK_THROW(p[i*N + j] == 0.0, "DeviceGraph::transform: U couples the functions of different batches");
                }
                mra::DenseTensor<double, 2> blk(std::array<mra::size_type, 2>{n, n});
                double* q = blk.buffer().host_ptr();
                for (mra::size_type i = 0; i < n; ++i)
                    for (mra::size_type j = 0; j < n; ++j) q[i*n + j] = p[(o + i)*N + o + j];
                ttg::send<0>(mra::Key<3>(b, mra::Level(0)), std::move(blk));
            }
        }, ttg::edges(U_e), ttg::edges(blocks), "transform-distribute"));
        U_e = blocks;
    }
    return pimpl->add_functions(add_transform(pimpl->g, pimpl->get(in).compressed, in_layout, out_layout, U_e, tol,
                                              "transform"), out_layout);
}

DeviceGraph::Functions DeviceGraph::concat(Functions a, Functions b) {
    require_one_batch(*pimpl->get(a).layout, "DeviceGraph::concat");
    require_one_batch(*pimpl->get(b).layout, "DeviceGraph::concat");
    auto& g = pimpl->g;
    NodeEdge out_e;
    g.add(mra::make_concat<double, 3>(make_mrattg_functionset(a.size), make_mrattg_functionset(b.size), g.K,
                                      pimpl->get(a).compressed, pimpl->get(b).compressed, out_e, "concat",
                                      device_procmap(), device_devicemap(), pimpl->concat_group));
    return pimpl->add_functions(out_e, a.size + b.size);
}

} // namespace madness
