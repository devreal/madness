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

DeviceGraph::Functions DeviceGraph::convolution(const poperatorT& op, Functions in, double tol,
                                                DeviceOperatorCache* cache) {
    using T = double;
    auto& g = pimpl->g;
    const DeviceLayout layout = pimpl->get(in).layout;   // per batch: one operator for every function
    // the convolution tasks refer to the function set's shared_ptr object itself and to the
    // operator (captured by reference): both are held here, at stable addresses, until the graph has run
    auto fns_holder = std::make_shared<decltype(make_mrattg_functionset(layout))>(make_mrattg_functionset(layout));
    const auto& fns = *fns_holder;
    std::shared_ptr<mra::GaussianConvolutionOperator<T, 3>> dop;
    if (cache) {
        for (auto& [hop, d] : cache->ops) if (hop == op) dop = d;
    }
    if (!dop) {
        // one operator for all N functions
        dop = std::make_shared<mra::GaussianConvolutionOperator<T, 3>>(op);
        if (cache) cache->ops.emplace_back(op, dop);
    }
    pimpl->operators.push_back(fns_holder);
    pimpl->operators.push_back(dop);
    NodeEdge ns_e, conv_e, out_e;
    // non-standard form for the convolution, compressed (and truncated) result, each in a single pass
    g.add(mra::make_nonstandard(fns, g.K, g.functiondata, pimpl->get(in).compressed, ns_e, "nonstandard",
                                device_procmap(), device_devicemap()));
    g.add(mra::make_convolution(fns, g.K, ns_e, conv_e, *dop, T(FunctionDefaults<3>::get_thresh()),
                                FunctionDefaults<3>::get_truncate_mode(), FunctionDefaults<3>::get_cell_min_width(),
                                "convolution", device_procmap(), device_devicemap()));
    add_compress_from_ns(g.tts, fns, g.K, g.functiondata, conv_e, out_e, tol, "recompress_convolution");
    return pimpl->add_functions(out_e, layout);
}

} // namespace madness
