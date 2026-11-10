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

/// \file device/bsh_resident.cc
/// \brief The device-resident BSH residual (see bsh.cc for SCF::apply_bsh_mrattg).

#include "common.h"

namespace madness {

/// BSH residual on the device: with fockp the Fock matrix minus eps on the diagonal,
///   rneg = G(-2 (Vpsi - psi fockp)) - psi,   rnorm_i = ||rneg_i||
/// i.e., the negative of the residual r = psi - new psi. psi is kept, Vpsi is consumed.
DeviceFunctions device_bsh_residual(World& world, DeviceFunctions& psi, DeviceFunctions& Vpsi,
                                           const Tensor<double>& fockp, const tensorT& eps,
                                           const CalculationParameters& param, std::vector<double>& rnorm,
                                           DeviceOperatorCache* cache) {
    using T = double;
    MraGraph g;
    g.name = "bsh_residual";
    const mra::size_type N = psi.size();
    // in the orbitals' batches: each batch is computed by its process group; the Fock matrix must
    // be block-diagonal in the batches (canonical orbitals: diagonal), as MRA-TTG checks
    const DeviceLayout layout = layout_of(psi);
    MADNESS_CHECK_THROW(same_batches(*layout, *layout_of(Vpsi)), "device_bsh_residual: psi and Vpsi in different batches");
    const T precision = FunctionDefaults<3>::get_thresh();
    const int truncate_mode = FunctionDefaults<3>::get_truncate_mode();
    const T cell_min_width = FunctionDefaults<3>::get_cell_min_width();
    auto fns = make_mrattg_functionset(layout);
    std::vector<poperatorT> ops = SCF::make_bsh_operators(world, eps, param);
    // the matrices of the 1D terms the previous BSH operators generated are reused
    mra::GaussianConvolutionOperator<T, 3> op(ops, cache ? cache->bsh_tables : nullptr);
    if (cache) cache->bsh_tables = op.term_tables();

    NodeEdge psi_e, V_e, fpsi_e, Vp_e, conv_in_e, conv_e, trunc_e, r_e;
    ttg::Edge<mra::Key<3>, mra::DenseTensor<T, 1>> norm_e;
    DeviceFunctions r;

    g.add(fv(psi).make_feed(g.control, psi_e, true, "feed_psi", device_procmap(layout), device_devicemap(layout)));
    g.add(fv(Vpsi).make_feed(g.control, V_e, false, "feed_Vpsi", device_procmap(layout), device_devicemap(layout)));
    // Vpsi' = -2 Vpsi + 2 psi fockp
    g.add(mra::make_transform<T, 3>(fns, fns, g.K, device_matrix(fockp), psi_e, fpsi_e, "fock_psi", device_procmap(layout), device_devicemap(layout)));
    g.add(mra::make_gaxpy(T(-2.0), T(2.0), fns, g.K, V_e, fpsi_e, Vp_e, "Vpsi_shift", device_procmap(layout), device_devicemap(layout)));
    // BSH apply: non-standard form for the convolution, compressed and truncated result,
    // each conversion in a single pass
    g.add(mra::make_nonstandard(fns, g.K, g.functiondata, Vp_e, conv_in_e, "nonstandard", device_procmap(layout), device_devicemap(layout)));
    g.add(mra::make_convolution(fns, g.K, conv_in_e, conv_e, op, precision, truncate_mode, cell_min_width, "convolution",
                                device_procmap(layout), device_devicemap(layout)));
    // truncation folded into the recompression (see add_compress_from_ns)
    add_compress_from_ns(g.tts, fns, g.K, g.functiondata, conv_e, trunc_e, precision, "recompress_convolution");
    // -r = new psi - psi, and its norms. Not psi - new psi: gaxpy forwards an operand with
    // scalar 1 unchanged where the other one is empty, which would hand a kept psi node to
    // the store of r (see mra::FunctionVector)
    g.add(mra::make_gaxpy(T(1.0), T(-1.0), fns, g.K, trunc_e, psi_e, r_e, "residual", device_procmap(layout), device_devicemap(layout)));
    g.add(mra::make_norm(fns, g.K, r_e, norm_e, "residual_norm", device_procmap(layout), device_devicemap(layout)));
    Tensor<double> norms{long(N)};   // per batch, at its offset
    g.add(make_batch_vector_sink(norm_e, norms, layout, "residual_norm_sink"));
    g.add(fv(r).make_store(r_e, N, "store_r", device_procmap(layout), device_devicemap(layout), layout));
    g.execute();
    check_device_tree(r, "device_bsh_residual");
    world.gop.sum(norms.ptr(), norms.size());
    rnorm.assign(norms.ptr(), norms.ptr() + norms.size());
    return r;
}

} // namespace madness
