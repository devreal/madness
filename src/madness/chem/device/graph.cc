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

/// \file device/graph.cc
/// \brief DeviceGraph: several MRA-TTG operations in one graph, connected by edges.
///
/// The operations are split over graph.cc (graph plumbing), graph_ops.cc (operations producing
/// functions) and graph_reduce.cc (reductions); see graph_impl.h.

#include "graph_impl.h"

namespace madness {

DeviceGraph::DeviceGraph(World& world) : pimpl(std::make_unique<Impl>(world)) {}
DeviceGraph::~DeviceGraph() = default;

DeviceGraph::Functions DeviceGraph::feed(DeviceFunctions& v, bool keep) {
    NodeEdge e;
    const DeviceLayout layout = layout_of(v);   // read before the feed takes the nodes out
    pimpl->g.add(fv(v).make_feed(pimpl->g.control, e, keep, "feed", device_procmap(), device_devicemap()));
    if (keep) pimpl->kept.push_back(&v);
    return pimpl->add_functions(e, layout);
}

void DeviceGraph::store(Functions f, DeviceFunctions& v) {
    pimpl->g.add(fv(v).make_store(pimpl->get(f).compressed, f.size, "store", device_procmap(), device_devicemap(),
                                  pimpl->get(f).layout));
    pimpl->stored.push_back(&v);
}

DeviceGraph::Matrix DeviceGraph::decide(Result r, Decision f) {
    MADNESS_CHECK_THROW(r.id >= 0 && r.id < int(pimpl->results.size()), "DeviceGraph: invalid result handle");
    Impl::Res& res = *pimpl->results[r.id];
    if (res.matrix) return pimpl->add_decision(*res.matrix, res.value.dim(0), res.value.dim(1), std::move(f));
    if (res.vector && res.layout && res.layout->num_batches() > 1) {
        // the batches' parts, gathered at the root of batch 0: each widened to the whole vector,
        // the widened vectors summed by the input reducer of the gather task
        const DeviceLayout layout = res.layout;
        const long n = res.value.dim(0);
        ttg::Edge<mra::Key<3>, std::vector<double>> wide;
        VectorEdge gathered;
        pimpl->g.add(ttg::make_tt([layout, n](const mra::Key<3>& key, const mra::DenseTensor<double, 1>& part) {
            std::vector<double> w(n, 0.0);
            const double* p = part.current_view().data();
            std::copy(p, p + layout->batch_size(key.batch()), w.begin() + layout->offset(key.batch()));
            ttg::send<0>(mra::Key<3>(mra::Batch(0), mra::Level(0)), std::move(w));
        }, ttg::edges(*res.vector), ttg::edges(wide), "gather-widen"));
        auto gather = ttg::make_tt([n](const mra::Key<3>& key, std::vector<double>&& w) {
            mra::DenseTensor<double, 1> v{mra::size_type(n)};
            std::copy(w.begin(), w.end(), v.buffer().host_ptr());
            ttg::send<0>(key, std::move(v));
        }, ttg::edges(wide), ttg::edges(gathered), "gather");
        gather->template set_input_reducer<0>([](std::vector<double>& a, const std::vector<double>& b) {
            for (std::size_t i = 0; i < a.size(); ++i) a[i] += b[i];
        }, layout->num_batches());
        pimpl->g.add(std::move(gather));
        return pimpl->add_decision(gathered, n, 0, std::move(f));
    }
    if (res.vector) return pimpl->add_decision(*res.vector, res.value.dim(0), 0, std::move(f));
    MADNESS_EXCEPTION("DeviceGraph::decide: this result is not a single reduction", r.id);
}

DeviceGraph::Matrix DeviceGraph::decide(Matrix m, Decision f) {
    // the size of the matrix is only known when it arrives
    MatrixEdge in = pimpl->matrix(m);
    pimpl->matrices.emplace_back();
    MatrixEdge out = pimpl->matrices.back();
    pimpl->g.add(ttg::make_tt([f](const mra::Key<3>& key, const mra::DenseTensor<double, 2>& x) {
        Tensor<double> xh(long(x.dim(0)), long(x.dim(1)));
        std::copy(x.current_view().data(), x.current_view().data() + xh.size(), xh.ptr());
        Tensor<double> uh = copy(f(xh));
        mra::DenseTensor<double, 2> u(std::array<mra::size_type, 2>{mra::size_type(uh.dim(0)), mra::size_type(uh.dim(1))});
        std::copy(uh.ptr(), uh.ptr() + uh.size(), u.buffer().host_ptr());
        ttg::send<0>(key, std::move(u));
    }, ttg::edges(in), ttg::edges(out), "decide"));
    return Matrix{int(pimpl->matrices.size()) - 1};
}

void DeviceGraph::execute() {
    pimpl->g.execute();
    for (auto* v : pimpl->stored) check_device_tree(*v, "DeviceGraph::store");
    for (auto* v : pimpl->kept) check_device_tree(*v, "DeviceGraph::feed(keep)");
    for (auto& r : pimpl->results) {   // only the process owning the root received them
        pimpl->world.gop.sum(r->value.ptr(), r->value.size());
    }
    for (std::size_t i = 0; i < pimpl->host_targets.size(); ++i) *pimpl->host_targets[i] = pimpl->host_out[i][0];
}

Tensor<double> DeviceGraph::get(Result r) const {
    MADNESS_CHECK_THROW(r.id >= 0 && r.id < int(pimpl->results.size()), "DeviceGraph: invalid result handle");
    return copy(pimpl->results[r.id]->value);
}

/// 1/2 sum_axis <d psi_i|d psi_j> (cf. SCF::kinetic_energy_matrix with the abgv derivative); psi is kept
Tensor<double> device_kinetic_matrix(World& world, DeviceFunctions& psi) {
    DeviceGraph g(world);
    auto r = g.kinetic_matrix(g.feed(psi, true));
    g.execute();
    return g.get(r);
}

} // namespace madness
