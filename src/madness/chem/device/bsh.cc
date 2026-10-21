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

/// \file device/bsh.cc
/// \brief BSH apply on MRA-TTG: SCF::apply_bsh_mrattg and the device-resident BSH residual.

#include "common.h"
#include "compare.h"

namespace madness {

vecfuncT SCF::apply_bsh_mrattg(World& world, vecfuncT& Vpsi, const tensorT& eps,
                               const CalculationParameters& param,
                               const vecfuncT* psi, std::vector<double>* rnorm) {

    scf_start_timer(world);

    const bool residual = (psi != nullptr);
    MADNESS_CHECK_THROW(!residual || psi->size() == Vpsi.size(), "apply_bsh_mrattg: psi and Vpsi differ in size");
    std::vector<poperatorT> ops = make_bsh_operators(world, eps, param);
    set_thresh(world, Vpsi, FunctionDefaults<3>::get_thresh());


    // define N Gaussians, don't instantiate
    // TODO: use the Vpsi process map
    using T = double;
    constexpr mra::Dimension NDIM = 3;
    const mra::size_type N = Vpsi.size();
    const mra::size_type K = FunctionDefaults<3>::get_k();
    const T precision = FunctionDefaults<3>::get_thresh();
    mra::FunctionData<T, NDIM> functiondata(K);
    mra::GaussianConvolutionOperator<T, NDIM> op(ops);
    auto pmap = mra::make_procmap<NDIM>(N, 1 /* batch */);
    auto dmap = mra::make_devicemap<NDIM>(pmap);
    auto gaussians = mra::make_functionset<mra::Gaussian<T, NDIM>>(pmap.batch_manager());

    /* create the result function vector (new psi, or the residual); share impl for process-map ownership */
    vecfuncT result(N);
    for (mra::size_type i = 0; i < N; ++i) {
        result[i].set_impl(Vpsi[i], false);
    }

    /**
     * Quick check: do all functions have the same truncate mode?
     */
    int truncate_mode = Vpsi.front().get_impl()->get_truncate_mode();
    for (auto& f : Vpsi) {
        if (f.get_impl()->get_truncate_mode() != truncate_mode) {
            throw std::runtime_error("SCF: all functions must have the same truncate mode");
        }
    }

    /**
     * TODO: have the make* functions return their output edge so we don't have to define it up front.
     */
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> reconstruct_conv_result, compress_input;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> conv_input, compress_conv_result,
                                                                    convolution_result, recompress_result;
    ttg::Edge<mra::Key<NDIM>, void> load_control;
    std::vector<std::unique_ptr<ttg::TTBase>> tts; // crutch because we need to keep TTs alive
    auto start            = mra::make_start(gaussians, load_control);
    if (Vpsi.front().get_impl()->get_tree_state() == madness::TreeState::reconstructed) {
        // Reconstructed nodes: load and compress into nonstandard form
        auto load = mra::vmra::make_vmra_load(Vpsi, load_control, compress_input, "load_vmra");
        tts.push_back(std::move(load));
    } else if (Vpsi.front().get_impl()->get_tree_state() == madness::TreeState::compressed) {
        // Compressed nodes: load, reconstruct, and compress into nonstandard form
        ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> load_vmra;
        auto load = mra::vmra::make_vmra_load(Vpsi, load_control, load_vmra, "load_vmra");
        auto reconstruct = mra::make_reconstruct(gaussians, K, false, functiondata, load_vmra, compress_input, "reconstruct");
        tts.push_back(std::move(load));
        tts.push_back(std::move(reconstruct));
    } else {
        throw std::runtime_error("Vpsi must be in reconstructed or compressed form");
    }
    auto compress         = mra::make_compress(gaussians, K, true, functiondata, compress_input, conv_input, "compress");
    auto convolve         = mra::make_convolution(gaussians, K, conv_input, convolution_result, op, precision, truncate_mode,
                                                    FunctionDefaults<NDIM>::get_cell_min_width(), "convolution");
    auto reconstruct_conv = mra::make_reconstruct(gaussians, K, true, functiondata, convolution_result, reconstruct_conv_result, "reconstruct_convolution");
    // truncation folded directly into recompress (screens out negligible
    // wavelet coefficients as they are produced, instead of a second
    // make_truncate pass over the fully compressed tree)
    auto recompress_conv  = mra::make_compress(gaussians, K, false, functiondata, reconstruct_conv_result, recompress_result, "recompress_convolution",
                                                ttg::Void{}, ttg::Void{}, /*enable_truncate=*/true, precision,
                                                madness::FunctionDefaults<NDIM>::get_truncate_mode(),
                                                madness::FunctionDefaults<NDIM>::get_cell_min_width());

    // residual r = psi - BSH(Vpsi) and its norms, on the device
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> psi_compressed, residual_result;
    ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 1>> norm_result;
    std::vector<double> norms(N, 0.0);
    if (residual) {
        load_compressed_mrattg(*psi, gaussians, K, functiondata, load_control, psi_compressed, tts, "psi");
        tts.push_back(mra::make_gaxpy(T(1.0), T(-1.0), gaussians, K, psi_compressed, recompress_result,
                                      residual_result, "residual"));
        tts.push_back(mra::make_norm(gaussians, K, residual_result, norm_result, "residual_norm"));
        // the norms of all N functions arrive at the root of the (single) batch
        tts.push_back(ttg::make_tt([&norms, N](const mra::Key<NDIM>& key, const mra::DenseTensor<T, 1>& n) {
            MADNESS_CHECK_THROW(key.level() == 0 && n.size() == N, "residual norms: expected one batch at the root");
            auto view = n.current_view();
            for (mra::size_type i = 0; i < N; ++i) norms[i] = view[i];
        }, ttg::edges(norm_result), ttg::edges(), "residual_norm_sink"));
    }
    // add_leaves: MRA-TTG does not forward nodes truncated away for all functions,
    // MADNESS needs them as empty leaves to keep the compressed tree well-formed
    auto store_tt         = mra::vmra::make_vmra_store(result, residual ? residual_result : recompress_result,
                                                       true /* add_leaves */);

    execute_mra_ttg(start, mrattg_tt_list(tts, {compress.get(), convolve.get(), reconstruct_conv.get(),
                                                recompress_conv.get(), store_tt.get()}));

    scf_end_timer(world, residual ? "Apply BSH & residual" : "Apply BSH & truncate");

    if (residual) {
        // only the rank owning the root key received the norms
        world.gop.sum(norms.data(), norms.size());
        if (rnorm) *rnorm = norms;
    }

    if (validate_mra_ttg) {
        auto mad_new_psi = apply(world, ops, Vpsi);
        madness::compress(world, mad_new_psi);
        truncate(world, mad_new_psi);
        if (residual) {
            madness::compress(world, *psi);
            vecfuncT mad_r = sub(world, *psi, mad_new_psi);
            compare_mra_madness(mad_r, result, "BSH-residual", 1e-8);
            std::vector<double> mad_rnorm = norm2s(world, mad_r);
            std::vector<double> mad_rnorm_ttg = norm2s(world, result);
            double maxdiff = 0.0, maxdiff_stored = 0.0;
            for (mra::size_type i = 0; i < N; ++i) {
                maxdiff = std::max(maxdiff, std::abs(norms[i] - mad_rnorm[i]));
                maxdiff_stored = std::max(maxdiff_stored, std::abs(mad_rnorm_ttg[i] - mad_rnorm[i]));
            }
            if (world.rank() == 0) {
                print("BSH-residual norms (madness)   ", mad_rnorm);
                print("BSH-residual norms (mra-ttg)   ", norms);
                print("BSH-residual norms: max |mra-ttg - madness|", maxdiff,
                      " max |norm2(stored r) - madness|", maxdiff_stored);
            }
        } else {
            compare_mra_madness(mad_new_psi, result, "BSH-conv-result", 1e-8);
        }
    }

    ops.clear();
    // TODO: move Vpsi into load-vmra so we can free nodes once they are loaded
    Vpsi.clear();

    return result;
}

/// BSH residual on the device: with fockp the Fock matrix minus eps on the diagonal,
///   rneg = G(-2 (Vpsi - psi fockp)) - psi,   rnorm_i = ||rneg_i||
/// i.e., the negative of the residual r = psi - new psi. psi is kept, Vpsi is consumed.
DeviceFunctions device_bsh_residual(World& world, DeviceFunctions& psi, DeviceFunctions& Vpsi,
                                           const Tensor<double>& fockp, const tensorT& eps,
                                           const CalculationParameters& param, std::vector<double>& rnorm) {
    using T = double;
    MraGraph g;
    const mra::size_type N = psi.size();
    const T precision = FunctionDefaults<3>::get_thresh();
    const int truncate_mode = FunctionDefaults<3>::get_truncate_mode();
    const T cell_min_width = FunctionDefaults<3>::get_cell_min_width();
    auto fns = make_mrattg_functionset(N);
    std::vector<poperatorT> ops = SCF::make_bsh_operators(world, eps, param);
    mra::GaussianConvolutionOperator<T, 3> op(ops);

    using RecEdge = ttg::Edge<mra::Key<3>, mra::FunctionsReconstructedNode<T, 3>>;
    NodeEdge psi_e, V_e, fpsi_e, Vp_e, conv_in_e, conv_e, recomp_e, trunc_e, r_e;
    RecEdge rec_e, rec_conv_e;
    ttg::Edge<mra::Key<3>, mra::DenseTensor<T, 1>> norm_e;
    DeviceFunctions r;

    g.add(fv(psi).make_feed(g.control, psi_e, true, "feed_psi"));
    g.add(fv(Vpsi).make_feed(g.control, V_e, false, "feed_Vpsi"));
    // Vpsi' = -2 Vpsi + 2 psi fockp
    g.add(mra::make_transform<T, 3>(fns, fns, g.K, device_matrix(fockp), psi_e, fpsi_e, "fock_psi"));
    g.add(mra::make_gaxpy(T(-2.0), T(2.0), fns, g.K, V_e, fpsi_e, Vp_e, "Vpsi_shift"));
    // BSH apply: non-standard form for the convolution, compressed and truncated result
    g.add(mra::make_reconstruct(fns, g.K, false, g.functiondata, Vp_e, rec_e, "reconstruct"));
    g.add(mra::make_compress(fns, g.K, true, g.functiondata, rec_e, conv_in_e, "compress"));
    g.add(mra::make_convolution(fns, g.K, conv_in_e, conv_e, op, precision, truncate_mode, cell_min_width, "convolution"));
    g.add(mra::make_reconstruct(fns, g.K, true, g.functiondata, conv_e, rec_conv_e, "reconstruct_convolution"));
    g.add(mra::make_compress(fns, g.K, false, g.functiondata, rec_conv_e, recomp_e, "recompress_convolution"));
    g.add(mra::make_truncate(fns, g.K, precision, truncate_mode, cell_min_width, recomp_e, trunc_e, "truncate"));
    // -r = new psi - psi, and its norms. Not psi - new psi: gaxpy forwards an operand with
    // scalar 1 unchanged where the other one is empty, which would hand a kept psi node to
    // the store of r (see mra::FunctionVector)
    g.add(mra::make_gaxpy(T(1.0), T(-1.0), fns, g.K, trunc_e, psi_e, r_e, "residual"));
    g.add(mra::make_norm(fns, g.K, r_e, norm_e, "residual_norm"));
    rnorm.assign(N, 0.0);
    g.add(ttg::make_tt([&rnorm, N](const mra::Key<3>& key, const mra::DenseTensor<T, 1>& n) {
        const T* data = n.current_view().data();
        for (mra::size_type i = 0; i < N; ++i) rnorm[i] = data[i];
    }, ttg::edges(norm_e), ttg::edges(), "residual_norm_sink"));
    g.add(fv(r).make_store(r_e, N, "store_r"));
    g.execute();
    check_device_tree(r, "device_bsh_residual");
    world.gop.sum(rnorm.data(), rnorm.size());
    return r;
}

} // namespace madness
