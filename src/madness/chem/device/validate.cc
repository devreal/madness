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

/// \file device/validate.cc
/// \brief Validation of the MRA-TTG inner products and transform against MADNESS.

#include "common.h"
#include "compare.h"

namespace madness {

/// <f_i|g_i> computed with MRA-TTG; f and g must have the same size
static Tensor<double> mrattg_inner(World& world, const vecfuncT& f, const vecfuncT& g) {
    using T = double;
    constexpr mra::Dimension NDIM = 3;
    MADNESS_CHECK_THROW(f.size() == g.size(), "mrattg_inner: f and g differ in size");
    const mra::size_type N = f.size();
    const mra::size_type K = FunctionDefaults<3>::get_k();
    mra::FunctionData<T, NDIM> functiondata(K);
    auto fns = make_mrattg_functionset(N);

    ttg::Edge<mra::Key<NDIM>, void> control;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> fe, ge;
    ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 1>> result_e;
    std::vector<std::unique_ptr<ttg::TTBase>> tts;
    auto start = mra::make_start(fns, control);
    load_compressed_mrattg(f, fns, K, functiondata, control, fe, tts, "f");
    load_compressed_mrattg(g, fns, K, functiondata, control, ge, tts, "g");
    tts.push_back(mra::make_inner<T, NDIM>(fns, K, fe, ge, result_e));
    Tensor<double> result(N);
    tts.push_back(ttg::make_tt([&result, N](const mra::Key<NDIM>& key, const mra::DenseTensor<T, 1>& r) {
        const T* data = r.current_view().data();
        for (mra::size_type i = 0; i < N; ++i) result(i) = data[i];
    }, ttg::edges(result_e), ttg::edges(), "inner_sink"));
    execute_mra_ttg(start, mrattg_tt_list(tts));
    world.gop.sum(result.ptr(), result.size());   // only the root's owner received it
    return result;
}

/// <f_i|g_j> computed with MRA-TTG; passing the same vector twice feeds one edge to both sides
static Tensor<double> mrattg_matrix_inner(World& world, const vecfuncT& f, const vecfuncT& g) {
    using T = double;
    constexpr mra::Dimension NDIM = 3;
    const bool same = (&f == &g);
    const mra::size_type N = f.size(), M = g.size();
    const mra::size_type K = FunctionDefaults<3>::get_k();
    mra::FunctionData<T, NDIM> functiondata(K);
    auto fnsF = make_mrattg_functionset(N);
    auto fnsG = same ? fnsF : make_mrattg_functionset(M);

    ttg::Edge<mra::Key<NDIM>, void> control;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> fe, ge;
    ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 2>> result_e;
    std::vector<std::unique_ptr<ttg::TTBase>> tts;
    auto start = mra::make_start(fnsF, control);
    load_compressed_mrattg(f, fnsF, K, functiondata, control, fe, tts, "f");
    if (!same) load_compressed_mrattg(g, fnsG, K, functiondata, control, ge, tts, "g");
    tts.push_back(mra::make_matrix_inner<T, NDIM>(fnsF, fnsG, K, fe, same ? fe : ge, result_e));
    Tensor<double> result(N, M);
    tts.push_back(ttg::make_tt([&result, N, M](const mra::Key<NDIM>& key, const mra::DenseTensor<T, 2>& r) {
        const T* data = r.current_view().data();
        for (mra::size_type i = 0; i < N; ++i)
            for (mra::size_type j = 0; j < M; ++j) result(i, j) = data[i*M + j];
    }, ttg::edges(result_e), ttg::edges(), "matrix_inner_sink"));
    execute_mra_ttg(start, mrattg_tt_list(tts));
    world.gop.sum(result.ptr(), result.size());
    return result;
}

/// out_j = sum_i v_i U(i,j) computed with MRA-TTG; the result is compressed
static vecfuncT mrattg_transform(World& world, const vecfuncT& v, const Tensor<double>& U) {
    using T = double;
    constexpr mra::Dimension NDIM = 3;
    const mra::size_type N = v.size(), M = U.dim(1);
    MADNESS_CHECK_THROW(U.dim(0) == long(N), "mrattg_transform: U must have v.size() rows");
    const mra::size_type K = FunctionDefaults<3>::get_k();
    mra::FunctionData<T, NDIM> functiondata(K);
    auto fns_in = make_mrattg_functionset(N);
    auto fns_out = make_mrattg_functionset(M);
    auto Ud = std::make_shared<mra::DenseTensor<T, 2>>(std::array<mra::size_type, 2>{N, M});
    T* Uh = Ud->buffer().host_ptr();
    for (mra::size_type i = 0; i < N; ++i)
        for (mra::size_type j = 0; j < M; ++j) Uh[i*M + j] = U(i, j);

    vecfuncT out(M);
    for (auto& f : out) f.set_impl(v.front(), false);   // same process map, empty tree

    ttg::Edge<mra::Key<NDIM>, void> control;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in_e, out_e;
    std::vector<std::unique_ptr<ttg::TTBase>> tts;
    auto start = mra::make_start(fns_in, control);
    load_compressed_mrattg(v, fns_in, K, functiondata, control, in_e, tts, "v");
    tts.push_back(mra::make_transform<T, NDIM>(fns_in, fns_out, K, Ud, in_e, out_e));
    tts.push_back(mra::vmra::make_vmra_store(out, out_e, true /* add_leaves */));
    execute_mra_ttg(start, mrattg_tt_list(tts));
    return out;
}

/// compare the MRA-TTG inner products and transform against MADNESS on f and g (same size)
void validate_mrattg_linalg(World& world, const vecfuncT& f, const vecfuncT& g) {
    const long N = f.size();
    auto report = [&world](const std::string& what, const Tensor<double>& ttg, const Tensor<double>& mad) {
        const double diff = (ttg - mad).absmax();
        if (world.rank() == 0)
            print("mra-ttg", what, ": max |mra-ttg - madness|", diff, " max |madness|", mad.absmax());
    };
    report("inner(f,g)", mrattg_inner(world, f, g), inner(world, f, g));
    report("matrix_inner(f,g)", mrattg_matrix_inner(world, f, g), matrix_inner(world, f, g));
    report("matrix_inner(f,f)", mrattg_matrix_inner(world, f, f), matrix_inner(world, f, f));

    // dense square U, and a sparse rectangular one (identity plus a column mixing two functions)
    Tensor<double> Udense(N, N), Usparse(N, N + 1);
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) Udense(i, j) = 1.0 / (1.0 + std::abs(i - j));
        Usparse(i, i) = 1.0;
    }
    Usparse(0, N) = 0.5;
    Usparse(N - 1, N) = -0.5;
    for (const auto& [name, U] : {std::pair<std::string, Tensor<double>>{"dense", Udense},
                                  std::pair<std::string, Tensor<double>>{"sparse", Usparse}}) {
        vecfuncT ttg_res = mrattg_transform(world, f, U);
        vecfuncT mad_res = transform(world, f, U, 0.0, true);
        compress(world, mad_res);
        compare_mra_madness(mad_res, ttg_res, "transform-" + name, 1e-10);
        std::vector<double> dnorm = norm2s(world, sub(world, mad_res, ttg_res));
        std::vector<double> mnorm = norm2s(world, mad_res);
        if (world.rank() == 0)
            print("mra-ttg transform", name, ": max ||mra-ttg - madness||",
                  *std::max_element(dnorm.begin(), dnorm.end()),
                  " max ||madness||", *std::max_element(mnorm.begin(), mnorm.end()));
    }
}

} // namespace madness
