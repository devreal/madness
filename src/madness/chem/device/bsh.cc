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
/// \brief BSH apply on MRA-TTG: SCF::apply_bsh_mrattg. The device-resident BSH residual is in
/// bsh_resident.cc; each builds a full convolution pipeline, too much for one translation unit.

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
    // the matrices of the 1D terms the previous BSH operators generated are reused
    DeviceOperatorCache& cache = device_operator_cache();
    mra::GaussianConvolutionOperator<T, NDIM> op(ops, cache.bsh_tables);
    cache.bsh_tables = op.term_tables();
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
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> conv_input, convolution_result, recompress_result;
    ttg::Edge<mra::Key<NDIM>, void> load_control;
    std::vector<std::unique_ptr<ttg::TTBase>> tts; // crutch because we need to keep TTs alive
    auto start            = mra::make_start(gaussians, load_control);
    if (Vpsi.front().get_impl()->get_tree_state() == madness::TreeState::reconstructed) {
        // Reconstructed nodes: load and compress into nonstandard form
        ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> load_vmra;
        tts.push_back(mra::vmra::make_vmra_load(Vpsi, load_control, load_vmra, "load_vmra"));
        tts.push_back(mra::make_compress(gaussians, K, true, functiondata, load_vmra, conv_input, "compress"));
    } else if (Vpsi.front().get_impl()->get_tree_state() == madness::TreeState::compressed) {
        // Compressed nodes: load and convert into nonstandard form in a single top-down pass
        ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> load_vmra;
        tts.push_back(mra::vmra::make_vmra_load(Vpsi, load_control, load_vmra, "load_vmra"));
        tts.push_back(mra::make_nonstandard(gaussians, K, functiondata, load_vmra, conv_input, "nonstandard"));
    } else {
        throw std::runtime_error("Vpsi must be in reconstructed or compressed form");
    }
    auto convolve         = mra::make_convolution(gaussians, K, conv_input, convolution_result, op, precision, truncate_mode,
                                                    FunctionDefaults<NDIM>::get_cell_min_width(), "convolution");
    // back to standard form in a single bottom-up pass, truncation folded in (see add_compress_from_ns)
    add_compress_from_ns(tts, gaussians, K, functiondata, convolution_result, recompress_result, precision,
                         "recompress_convolution");

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

    execute_mra_ttg(start, mrattg_tt_list(tts, {convolve.get(), store_tt.get()}));

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

} // namespace madness
