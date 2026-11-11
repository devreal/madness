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
/// \file device/ops_elementwise.cc
/// \brief The node-wise operators of the SCF's MRA-TTG graphs (see device_ops in common.h):
/// gaxpy, norm, truncate, transform.

#include "common.h"

namespace madness {
namespace device_ops {

TTPtr make_truncate(const DeviceFunctionSet& fns, mra::size_type K, double thresh, int truncate_mode,
                    double cell_min_width, NodeEdge& in, NodeEdge& out, const std::string& name) {
    return mra::make_truncate(fns, K, thresh, truncate_mode, cell_min_width, in, out, name, device_procmap(fns),
                              device_devicemap(fns));
}

TTPtr make_gaxpy(double scalarA, double scalarB, const DeviceFunctionSet& fns, mra::size_type K,
                 NodeEdge in1, NodeEdge in2, NodeEdge out, const char* name, bool forward) {
    return mra::make_gaxpy(scalarA, scalarB, fns, K, in1, in2, out, name, device_procmap(fns), device_devicemap(fns),
                           forward);
}

TTPtr make_norm(const DeviceFunctionSet& fns, mra::size_type K, NodeEdge in, VectorEdge result, const char* name) {
    return mra::make_norm(fns, K, in, result, name, device_procmap(fns), device_devicemap(fns));
}

TTPtr make_transform(const DeviceFunctionSet& fns_in, const DeviceFunctionSet& fns_out, mra::size_type K,
                     std::shared_ptr<mra::DenseTensor<double, 2>> U, NodeEdge in, NodeEdge out,
                     const std::string& name) {
    return mra::make_transform<double, 3>(fns_in, fns_out, K, std::move(U), in, out, name, device_procmap(fns_out),
                                          device_devicemap(fns_out));
}

TTPtr make_transform(const DeviceFunctionSet& fns_in, const DeviceFunctionSet& fns_out, mra::size_type K,
                     MatrixEdge U, NodeEdge in, NodeEdge out, const std::string& name) {
    return mra::make_transform<double, 3>(fns_in, fns_out, K, U, in, out, name, device_procmap(fns_out),
                                          device_devicemap(fns_out));
}

} // namespace device_ops
} // namespace madness
