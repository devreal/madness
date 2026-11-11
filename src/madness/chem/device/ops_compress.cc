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
/// \file device/ops_compress.cc
/// \brief The compressions of the SCF's MRA-TTG graphs (see device_ops in common.h).

#include "common.h"

namespace madness {
namespace device_ops {

TTPtr make_compress(const DeviceFunctionSet& fns, mra::size_type K, bool is_ns,
                    const mra::FunctionData<double, 3>& functiondata, RecEdge& in, NodeEdge& out,
                    const std::string& name, bool enable_truncate, double thresh,
                    int truncate_mode, double cell_min_width, const mra::BatchGroup& batch_group) {
    return mra::make_compress(fns, K, is_ns, functiondata, in, out, name, device_procmap(fns), device_devicemap(fns),
                              enable_truncate, thresh, truncate_mode, cell_min_width, batch_group);
}

TTPtr make_compress_from_ns(const DeviceFunctionSet& fns, mra::size_type K,
                            const mra::FunctionData<double, 3>& functiondata, NodeEdge& in, NodeEdge& out,
                            const std::string& name, bool enable_truncate, double thresh,
                            int truncate_mode, double cell_min_width) {
    return mra::make_compress_from_ns(fns, K, functiondata, in, out, name, device_procmap(fns), device_devicemap(fns),
                                      enable_truncate, thresh, truncate_mode, cell_min_width);
}

TTPtr make_nonstandard(const DeviceFunctionSet& fns, mra::size_type K,
                       const mra::FunctionData<double, 3>& functiondata, NodeEdge in, NodeEdge out,
                       const std::string& name) {
    return mra::make_nonstandard(fns, K, functiondata, std::move(in), std::move(out), name, device_procmap(fns),
                                 device_devicemap(fns));
}

} // namespace device_ops

void add_compress(std::vector<TTPtr>& tts, const DeviceFunctionSet& fns, mra::size_type K,
                  const mra::FunctionData<double, 3>& functiondata, RecEdge& in, NodeEdge& out,
                  double tol, const std::string& name) {
    static const bool separate = std::getenv("MRA_TTG_SEPARATE_TRUNCATE") != nullptr;
    const int mode = FunctionDefaults<3>::get_truncate_mode();
    const double min_width = FunctionDefaults<3>::get_cell_min_width();
    if (tol <= 0.0 || !separate) {
        tts.push_back(device_ops::make_compress(fns, K, false, functiondata, in, out, name,
                                                /*enable_truncate=*/tol > 0.0, tol, mode, min_width));
    } else {
        NodeEdge compressed;
        tts.push_back(device_ops::make_compress(fns, K, false, functiondata, in, compressed, name));
        tts.push_back(device_ops::make_truncate(fns, K, tol, mode, min_width, compressed, out, name + "-truncate"));
    }
}

void add_compress_from_ns(std::vector<TTPtr>& tts, const DeviceFunctionSet& fns, mra::size_type K,
                          const mra::FunctionData<double, 3>& functiondata, NodeEdge& in, NodeEdge& out,
                          double tol, const std::string& name) {
    static const bool separate = std::getenv("MRA_TTG_SEPARATE_TRUNCATE") != nullptr;
    const int mode = FunctionDefaults<3>::get_truncate_mode();
    const double min_width = FunctionDefaults<3>::get_cell_min_width();
    if (tol <= 0.0 || !separate) {
        tts.push_back(device_ops::make_compress_from_ns(fns, K, functiondata, in, out, name,
                                                        /*enable_truncate=*/tol > 0.0, tol, mode, min_width));
    } else {
        NodeEdge compressed;
        tts.push_back(device_ops::make_compress_from_ns(fns, K, functiondata, in, compressed, name));
        tts.push_back(device_ops::make_truncate(fns, K, tol, mode, min_width, compressed, out, name + "-truncate"));
    }
}

void load_compressed_mrattg(const vecfuncT& f, const DeviceFunctionSet& fns, mra::size_type K,
                            const mra::FunctionData<double, 3>& functiondata,
                            ttg::Edge<mra::Key<3>, void>& control, NodeEdge& out,
                            std::vector<TTPtr>& tts, const std::string& name) {
    const auto state = f.front().get_impl()->get_tree_state();
    const DeviceLayout layout(fns->batchman());
    if (state == madness::TreeState::reconstructed) {
        RecEdge reconstructed;
        tts.push_back(device_ops::make_vmra_load(f, control, reconstructed, "load_" + name, layout));
        tts.push_back(device_ops::make_compress(fns, K, false, functiondata, reconstructed, out, "compress_" + name));
    } else if (state == madness::TreeState::compressed) {
        tts.push_back(device_ops::make_vmra_load(f, control, out, "load_" + name, layout));
    } else {
        throw std::runtime_error(name + " must be in reconstructed or compressed form");
    }
}

} // namespace madness
