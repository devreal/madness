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
        std::lock_guard<std::mutex> lock(r.m);
        r.place(key, m.current_view().data(), r.value, fac);
    }, ttg::edges(e), ttg::edges(), name));
}

MatrixEdge DeviceGraph::Impl::add_matrix_inner(NodeEdge a, const DeviceLayout& la, NodeEdge b, const DeviceLayout& lb,
                                                bool same, Res& r, double fac, const std::string& name) {
    MatrixEdge m_e;
    if (la->num_batches() == 1 && lb->num_batches() == 1) {
        g.add(mra::make_matrix_inner<double, 3>(make_mrattg_functionset(la), make_mrattg_functionset(lb), g.K, a,
                                                same ? a : b, m_e, name, device_procmap(la), device_devicemap(la),
                                                inner_group));
        r.arrivals += 1;
        r.place = [](const mra::Key<3>&, const double* d, Tensor<double>& into, double f) {
            for (long i = 0; i < into.size(); ++i) into.ptr()[i] += f * d[i];
        };
        add_matrix_sink(m_e, r, fac, name + "-sink");
        return m_e;
    }
    // The pairs of batches (ba, bb), for a set with itself ba <= bb (the other half mirrored). A
    // pair is computed under its own key (batch p) by the group of its home batch, one of ba and
    // bb: the nodes of the home batch stay on their processes, those of the other are sent there.
    // For a set with itself the home alternates, so that each batch is home to about half its pairs.
    std::vector<std::pair<mra::Batch, mra::Batch>> pairs;
    std::vector<std::pair<int, int>> ranges, slots;   // the home batch's group and slot
    std::vector<mra::size_type> sizes_a, sizes_b;
    auto ta = std::make_shared<std::vector<std::vector<mra::Batch>>>(la->num_batches());
    auto tb = std::make_shared<std::vector<std::vector<mra::Batch>>>(lb->num_batches());
    for (mra::Batch ba = 0; ba < la->num_batches(); ++ba) {
        for (mra::Batch bb = same ? ba : 0; bb < lb->num_batches(); ++bb) {
            const mra::Batch p = mra::Batch(pairs.size());
            MADNESS_CHECK_THROW(pairs.size() < 32767, "DeviceGraph: too many pairs of batches for a matrix");
            pairs.emplace_back(ba, bb);
            const bool home_b = same && ((ba + bb) % 2 == 1);
            const mra::BatchGroups& hg = home_b ? lb->groups() : la->groups();
            ranges.push_back(hg.batch_pes(home_b ? bb : ba));
            slots.push_back(hg.batch_slot(home_b ? bb : ba));
            sizes_a.push_back(la->batch_size(ba));
            sizes_b.push_back(lb->batch_size(bb));
            (*ta)[ba].push_back(p);
            (*tb)[bb].push_back(p);
        }
    }
    const mra::BatchGroups pg(ranges, la->num_pes(), slots);
    const DeviceLayout pa = std::make_shared<const mra::BatchLayout>(sizes_a, pg);
    const DeviceLayout pb = std::make_shared<const mra::BatchLayout>(sizes_b, pg);
    NodeEdge a_p, b_p;
    g.add(mra::make_rebatch<double, 3>(a, a_p, [ta](mra::Batch x) { return (*ta)[x]; }, name + "-pairs-a",
                                       device_procmap(la), device_devicemap(la)));
    g.add(mra::make_rebatch<double, 3>(b, b_p, [tb](mra::Batch x) { return (*tb)[x]; }, name + "-pairs-b",
                                       device_procmap(lb), device_devicemap(lb)));
    g.add(mra::make_matrix_inner<double, 3>(make_mrattg_functionset(pa), make_mrattg_functionset(pb), g.K, a_p, b_p,
                                            m_e, name, device_procmap(pa), device_devicemap(pa), inner_group));
    r.arrivals += int(pairs.size());
    r.place = [pairs, la, lb, same](const mra::Key<3>& key, const double* d, Tensor<double>& into, double f) {
        const auto [ba, bb] = pairs[key.batch()];
        const long oa = la->offset(ba), na = la->batch_size(ba), ob = lb->offset(bb), nb = lb->batch_size(bb);
        for (long i = 0; i < na; ++i) {
            for (long j = 0; j < nb; ++j) {
                into(oa + i, ob + j) += f * d[i*nb + j];
                if (same && ba != bb) into(ob + j, oa + i) += f * d[i*nb + j];
            }
        }
    };
    add_matrix_sink(m_e, r, fac, name + "-sink");
    return m_e;
}

DeviceGraph::Result DeviceGraph::matrix_inner(Functions a, Functions b) {
    Impl::Res& r = pimpl->add_result(long(a.size), long(b.size));
    r.matrix = pimpl->add_matrix_inner(pimpl->get(a).compressed, pimpl->get(a).layout, pimpl->get(b).compressed,
                                       pimpl->get(b).layout, a.id == b.id, r, 1.0, "matrix_inner");
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
                                     pimpl->get(b).compressed, *r.vector, "inner", device_procmap(layout),
                                     device_devicemap(layout), pimpl->inner_group));
    g.add(make_batch_vector_sink(*r.vector, r.value, layout, "inner_sink"));
    return Result{int(pimpl->results.size()) - 1};
}

DeviceGraph::Result DeviceGraph::norms(Functions a) {
    auto& g = pimpl->g;
    const DeviceLayout layout = pimpl->get(a).layout;
    Impl::Res& r = pimpl->add_result(long(a.size), 0);
    r.vector.emplace();
    r.layout = layout;
    g.add(device_ops::make_norm(make_mrattg_functionset(layout), g.K, pimpl->get(a).compressed, *r.vector, "norm"));
    g.add(make_batch_vector_sink(*r.vector, r.value, layout, "norm_sink"));
    return Result{int(pimpl->results.size()) - 1};
}

DeviceGraph::Result DeviceGraph::kinetic_matrix(Functions psi) {
    using T = double;
    auto& g = pimpl->g;
    const mra::size_type N = psi.size;
    // per batch: the derivatives and their compression; then the matrix across the batches
    const DeviceLayout layout = pimpl->get(psi).layout;
    auto fns = make_mrattg_functionset(layout);
    Impl::Res& r = pimpl->add_result(long(N), long(N));
    mra::BatchGroup compress_group;   // the three axes batch together
    // the three derivatives in one descent over the compressed orbitals, which are not reconstructed
    std::array<RecEdge, 3> d_e;
    g.add(mra::make_gradient_compressed<T, 3>(pimpl->domain(), fns, g.functiondata, g.K, pimpl->get(psi).compressed,
                                              d_e, "gradient", device_procmap(layout), device_devicemap(layout)));
    for (mra::Dimension axis = 0; axis < 3; ++axis) {
        const std::string name = "d" + std::to_string(axis);
        NodeEdge dc_e;
        g.add(device_ops::make_compress(fns, g.K, false, g.functiondata, d_e[axis], dc_e, "compress_" + name,
                                        false, T{}, 0, T{}, compress_group));
        pimpl->add_matrix_inner(dc_e, layout, dc_e, layout, true, r, 0.5, "inner_" + name);
    }
    return Result{int(pimpl->results.size()) - 1};
}

} // namespace madness
