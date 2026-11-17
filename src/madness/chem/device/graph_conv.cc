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

/// \file device/graph_conv.cc
/// \brief DeviceGraph convolution (e.g., the Coulomb operator), in its own TU as the
/// convolution pipeline is the most expensive operator to compile.

#include "graph_impl.h"

namespace madness {

DeviceOperatorCache& SCF::device_operator_cache() {
    if (!device_operators) device_operators = std::make_shared<DeviceOperatorCache>();
    return *device_operators;
}

namespace {

/// dop(in), compressed and truncated with tol unless tol == 0
DeviceGraph::Functions add_convolution(DeviceGraph::Impl& impl, std::shared_ptr<mra::GaussianConvolutionOperator<double, 3>> dop,
                                       DeviceGraph::Functions in, double tol) {
    using T = double;
    auto& g = impl.g;
    const DeviceLayout layout = impl.get(in).layout;
    // the convolution tasks refer to the function set's shared_ptr object itself and to the
    // operator (captured by reference): both are held here, at stable addresses, until the graph has run
    auto fns_holder = std::make_shared<decltype(make_mrattg_functionset(layout))>(make_mrattg_functionset(layout));
    const auto& fns = *fns_holder;
    impl.operators.push_back(fns_holder);
    impl.operators.push_back(dop);
    NodeEdge ns_e, conv_e, out_e;
    // non-standard form for the convolution, compressed (and truncated) result, each in a single pass
    g.add(device_ops::make_nonstandard(fns, g.K, g.functiondata, impl.get(in).compressed, ns_e, "nonstandard"));
    g.add(device_ops::make_convolution(fns, g.K, ns_e, conv_e, *dop, T(FunctionDefaults<3>::get_thresh()),
                                       FunctionDefaults<3>::get_truncate_mode(), FunctionDefaults<3>::get_cell_min_width(),
                                       "convolution"));
    add_compress_from_ns(g.tts, fns, g.K, g.functiondata, conv_e, out_e, tol, "recompress_convolution");
    return impl.add_functions(out_e, layout);
}

} // namespace

DeviceGraph::Functions DeviceGraph::convolution(const poperatorT& op, Functions in, double tol,
                                                DeviceOperatorCache* cache) {
    std::shared_ptr<mra::GaussianConvolutionOperator<double, 3>> dop;
    if (cache) {
        for (auto& [hop, d] : cache->ops) if (hop == op) dop = d;
    }
    if (!dop) {
        // one operator for all N functions (per batch: every function of the set)
        dop = std::make_shared<mra::GaussianConvolutionOperator<double, 3>>(op);
        if (cache) cache->ops.emplace_back(op, dop);
    }
    return add_convolution(*pimpl, std::move(dop), in, tol);
}

DeviceGraph::Functions DeviceGraph::convolution(const std::vector<poperatorT>& ops, Functions in, double tol,
                                                DeviceOperatorCache* cache) {
    device_check(ops.size() == in.size, "DeviceGraph::convolution: one operator per function");
    // the matrices of the 1D terms earlier operators generated are reused (see DeviceOperatorCache)
    auto dop = std::make_shared<mra::GaussianConvolutionOperator<double, 3>>(ops, cache ? cache->bsh_tables : nullptr);
    if (cache) cache->bsh_tables = dop->term_tables();
    return add_convolution(*pimpl, std::move(dop), in, tol);
}

} // namespace madness
