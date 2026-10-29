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

/// \file device/functions.cc
/// \brief Device vectors of functions: upload, download, inner products, transform, update.

#include "common.h"

#if defined(MRA_ENABLE_CUDA)
#include <nvtx3/nvToolsExt.h>
#endif

namespace madness {

#if defined(MRA_ENABLE_CUDA)
static nvtxDomainHandle_t scf_nvtx_domain() {
    static nvtxDomainHandle_t domain = nvtxDomainCreateA("MRA-TTG SCF");
    return domain;
}
#endif

DeviceRange::DeviceRange(const char* name) {
#if defined(MRA_ENABLE_CUDA)
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = 0xff76b900;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxDomainRangePushEx(scf_nvtx_domain(), &attr);
#else
    (void)name;
#endif
}

DeviceRange::~DeviceRange() {
#if defined(MRA_ENABLE_CUDA)
    nvtxDomainRangePop(scf_nvtx_domain());
#endif
}

DeviceFunctions::DeviceFunctions() : pimpl(std::make_unique<Impl>()) {}
DeviceFunctions::~DeviceFunctions() = default;
DeviceFunctions::DeviceFunctions(DeviceFunctions&&) noexcept = default;
DeviceFunctions& DeviceFunctions::operator=(DeviceFunctions&&) noexcept = default;

std::size_t DeviceFunctions::size() const { return pimpl ? pimpl->v.size() : 0; }

DeviceFunctions::Impl& DeviceFunctions::impl() {
    if (!pimpl) pimpl = std::make_unique<Impl>();   // moved from
    return *pimpl;
}

/// copy MADNESS functions into a device vector
DeviceFunctions device_upload(World& world, const vecfuncT& f) {
    MraGraph g;
    NodeEdge e;
    DeviceFunctions d;
    load_compressed_mrattg(f, make_mrattg_functionset(f.size()), g.K, g.functiondata, g.control, e, g.tts, "upload");
    g.add(fv(d).make_store(e, f.size(), "upload_store"));
    g.execute();
    check_device_tree(d, "device_upload");
    return d;
}

/// copy a device vector into (compressed) MADNESS functions with the process map of `like`
vecfuncT device_download(World& world, DeviceFunctions& d, const functionT& like) {
    MraGraph g;
    NodeEdge e;
    vecfuncT out(d.size());
    for (auto& f : out) f.set_impl(like, false);
    g.add(fv(d).make_feed(g.control, e, true /* keep */, "download"));
    g.add(mra::vmra::make_vmra_store(out, e, true /* add_leaves */));
    g.execute();
    return out;
}

/// <a_i|b_j>; a and b may be the same vector
Tensor<double> device_matrix_inner(World& world, DeviceFunctions& a, DeviceFunctions& b) {
    DeviceGraph g(world);
    auto fa = g.feed(a, true);
    auto fb = (&a == &b) ? fa : g.feed(b, true);
    auto r = g.matrix_inner(fa, fb);
    g.execute();
    return g.get(r);
}

/// ||a_i||
std::vector<double> device_norms(World& world, DeviceFunctions& a) {
    DeviceGraph g(world);
    auto r = g.norms(g.feed(a, true));
    g.execute();
    Tensor<double> n = g.get(r);
    return std::vector<double>(n.ptr(), n.ptr() + n.size());
}

/// v_j <- sum_i v_i U(i,j), truncated with tol unless tol == 0 (cf. madness::transform)
void device_transform(World& world, DeviceFunctions& v, const Tensor<double>& U, double tol) {
    MADNESS_CHECK_THROW(U.dim(0) == long(v.size()), "device_transform: U must have v.size() rows");
    DeviceGraph g(world);
    g.store(g.transform(g.feed(v, false), U, tol), v);
    g.execute();
}

/// [a, b]: the functions of a followed by those of b, in a new vector; a and b are kept or consumed
DeviceFunctions device_concat(World& world, DeviceFunctions& a, bool keep_a, DeviceFunctions& b, bool keep_b) {
    DeviceGraph g(world);
    DeviceFunctions out;
    auto fa = g.feed(a, keep_a);
    g.store(g.concat(fa, g.feed(b, keep_b)), out);
    g.execute();
    return out;
}

/// out_j = sum_i v_i U(i,j) in a new vector, truncated with tol unless tol == 0; v is kept
DeviceFunctions device_transformed(World& world, DeviceFunctions& v, const Tensor<double>& U, double tol) {
    MraGraph g;
    const mra::size_type N = v.size(), M = U.dim(1);
    MADNESS_CHECK_THROW(U.dim(0) == long(N), "device_transformed: U must have v.size() rows");
    auto fns_out = make_mrattg_functionset(M);
    NodeEdge in_e, out_e, trunc_e;
    DeviceFunctions out;
    g.add(fv(v).make_feed(g.control, in_e, true, "feed"));
    g.add(mra::make_transform<double, 3>(make_mrattg_functionset(N), fns_out, g.K, device_matrix(U), in_e, out_e));
    if (tol > 0.0) {
        g.add(mra::make_truncate(fns_out, g.K, tol, FunctionDefaults<3>::get_truncate_mode(),
                                 FunctionDefaults<3>::get_cell_min_width(), out_e, trunc_e, "truncate"));
    }
    g.add(fv(out).make_store(tol > 0.0 ? trunc_e : out_e, M, "store"));
    g.execute();
    check_device_tree(out, "device_transformed");
    return out;
}

/// ||sum_i v_i U(i,j)|| for every column j of U; v is kept
std::vector<double> device_transform_norms(World& world, DeviceFunctions& v, const Tensor<double>& U) {
    using T = double;
    MraGraph g;
    const mra::size_type N = v.size(), M = U.dim(1);
    MADNESS_CHECK_THROW(U.dim(0) == long(N), "device_transform_norms: U must have v.size() rows");
    auto fns_out = make_mrattg_functionset(M);
    NodeEdge in_e, out_e;
    ttg::Edge<mra::Key<3>, mra::DenseTensor<T, 1>> norm_e;
    g.add(fv(v).make_feed(g.control, in_e, true, "feed"));
    g.add(mra::make_transform<T, 3>(make_mrattg_functionset(N), fns_out, g.K, device_matrix(U), in_e, out_e));
    g.add(mra::make_norm(fns_out, g.K, out_e, norm_e));
    std::vector<double> norms(M, 0.0);
    g.add(ttg::make_tt([&norms, M](const mra::Key<3>& key, const mra::DenseTensor<T, 1>& n) {
        const T* data = n.current_view().data();
        for (mra::size_type i = 0; i < M; ++i) norms[i] = data[i];
    }, ttg::edges(norm_e), ttg::edges(), "norm_sink"));
    g.execute();
    world.gop.sum(norms.data(), norms.size());
    return norms;
}

/// v_i <- c_i v_i
void device_scale(World& world, DeviceFunctions& v, const std::vector<double>& c) {
    Tensor<double> D(static_cast<long>(c.size()), static_cast<long>(c.size()));
    for (size_t i = 0; i < c.size(); ++i) D(i, i) = c[i];
    device_transform(world, v, D, 0.0);
}

/// v_i <- v_i / ||v_i||, in one graph: the scaling is computed from the norms while v waits in it
void device_normalize(World& world, DeviceFunctions& v) {
    DeviceGraph g(world);
    auto f = g.feed(v, false);
    auto scale = g.decide(g.norms(f), [](const Tensor<double>& n) {
        Tensor<double> D(n.dim(0), n.dim(0));
        for (long i = 0; i < n.dim(0); ++i) D(i, i) = 1.0 / n(i);
        return D;
    });
    g.store(g.transform(f, scale, v.size(), 0.0), v);
    g.execute();
}

/// a_i <- a_i - c_i r_i; consumes r
void device_update(World& world, DeviceFunctions& a, DeviceFunctions& r, const std::vector<double>& c) {
    MraGraph g;
    const mra::size_type N = a.size();
    auto fns = make_mrattg_functionset(N);
    Tensor<double> D(static_cast<long>(N), static_cast<long>(N));
    for (mra::size_type i = 0; i < N; ++i) D(i, i) = c[i];
    NodeEdge a_e, r_e, cr_e, out_e;
    g.add(fv(r).make_feed(g.control, r_e, false, "feed_r"));
    g.add(fv(a).make_feed(g.control, a_e, false, "feed_a"));
    g.add(mra::make_transform<double, 3>(fns, fns, g.K, device_matrix(D), r_e, cr_e, "scale_r"));
    g.add(mra::make_gaxpy(1.0, -1.0, fns, g.K, a_e, cr_e, out_e, "update"));
    g.add(fv(a).make_store(out_e, N, "store"));
    g.execute();
    check_device_tree(a, "device_update");
}

} // namespace madness
