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
/// \file device/ops_io.cc
/// \brief The operators moving functions into and out of the SCF's MRA-TTG graphs (see
/// device_ops in common.h): device vectors, and MADNESS functions on the host.

#include "common.h"

namespace madness {
namespace device_ops {

TTPtr make_feed(DeviceFunctions& v, ttg::Edge<mra::Key<3>, void>& control, NodeEdge& out, bool keep,
                const std::string& name, const DeviceLayout& layout) {
    const DeviceLayout l = layout ? layout : layout_of(v);   // read before the feed takes the nodes out
    return fv(v).make_feed(control, out, keep, name, device_procmap(l), device_devicemap(l));
}

TTPtr make_store(DeviceFunctions& v, NodeEdge& in, mra::size_type N, const std::string& name,
                 const DeviceLayout& layout) {
    const DeviceLayout l = layout ? layout : one_batch(N);
    return fv(v).make_store(in, N, name, device_procmap(l), device_devicemap(l), layout);
}

TTPtr make_vmra_load(const vecfuncT& f, ttg::Edge<mra::Key<3>, void>& control, RecEdge& out, const std::string& name,
                     const DeviceLayout& layout) {
    return mra::vmra::make_vmra_load(f, control, out, name, layout);
}

TTPtr make_vmra_load(const vecfuncT& f, ttg::Edge<mra::Key<3>, void>& control, NodeEdge& out, const std::string& name,
                     const DeviceLayout& layout) {
    return mra::vmra::make_vmra_load(f, control, out, name, layout);
}

TTPtr make_vmra_store(vecfuncT& f, NodeEdge& in, const std::string& name, const DeviceLayout& layout) {
#if defined(MRA_ENABLE_HOST)
    return mra::vmra::make_vmra_store(f, in, true /* add_leaves */, madness::TreeState::compressed, name, layout);
#else
    /* The store is a host task, which reads the host copy of the coefficients without checking
     * that it is current: TTG pushes data out to the host for a host consumer only in some cases
     * (e.g., not when all GPUs can access their peers). A device task first copies each node to
     * the host (wait), then hands it on. */
    using node_type = mra::FunctionsCompressedNode<double, 3>;
    const DeviceLayout l = layout ? layout : one_batch(f.size());
    NodeEdge host_e(name + "-host");
    auto to_host_tt = ttg::make_tt<::Space>([](const mra::Key<3>& key, const node_type& node) -> TASKTYPE {
        if (!node.empty() && !node.coeffs().buffer().empty()) {
            co_await ttg::device::select(node.coeffs().buffer());
            co_await ttg::device::wait(node.coeffs().buffer());
        }
        co_await ttg::device::send<0>(key, node);
    }, ttg::edges(in), ttg::edges(host_e), name + "-to_host", {"input"}, {"output"});
    to_host_tt->set_keymap(device_procmap(l));
    to_host_tt->set_devicemap(device_devicemap(l));
    auto store_tt = mra::vmra::make_vmra_store(f, host_e, true /* add_leaves */, madness::TreeState::compressed,
                                               name, layout);
    auto ins = std::make_tuple(to_host_tt->template in<0>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops;
    ops.push_back(std::move(to_host_tt));
    ops.push_back(std::move(store_tt));
    return ttg::make_ttg(std::move(ops), ins, std::make_tuple(), name);
#endif
}

} // namespace device_ops
} // namespace madness
