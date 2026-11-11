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
/// \file device/ops_convolution.cc
/// \brief The convolution of the SCF's MRA-TTG graphs (see device_ops in common.h), in its own TU
/// as it is the most expensive operator to compile.

#include "common.h"

namespace madness {
namespace device_ops {

TTPtr make_convolution(const DeviceFunctionSet& fns, mra::size_type K, NodeEdge in, NodeEdge out,
                       const mra::GaussianConvolutionOperator<double, 3>& op, double thresh, int truncate_mode,
                       double cell_min_width, const std::string& name) {
    return mra::make_convolution(fns, K, in, out, op, thresh, truncate_mode, cell_min_width, name,
                                 device_procmap(fns), device_devicemap(fns));
}

} // namespace device_ops
} // namespace madness
