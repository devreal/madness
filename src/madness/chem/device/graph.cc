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

/// \file device/graph.cc
/// \brief DeviceGraph: several MRA-TTG operations in one graph, connected by edges.

#include <mutex>
#include <optional>
#include "common.h"

namespace madness {

using RecEdge = ttg::Edge<mra::Key<3>, mra::FunctionsReconstructedNode<double, 3>>;
using MatrixEdge = ttg::Edge<mra::Key<3>, mra::DenseTensor<double, 2>>;
using VectorEdge = ttg::Edge<mra::Key<3>, mra::DenseTensor<double, 1>>;

struct DeviceGraph::Impl {
    World& world;
    MraGraph g;
    std::optional<ttg::Buffer<mra::Domain<3>>> db;
    /// the instances of an operation in this graph batch their device tasks together
    mra::BatchGroup concat_group, inner_group;

    /// a set of functions: compressed nodes, and reconstructed ones once needed
    struct Fn {
        NodeEdge compressed;
        std::optional<RecEdge> reconstructed;
        std::size_t N;
    };
    std::vector<Fn> fns;

    /// a reduction: its value on the host, and the edge it arrives on (for decide)
    struct Res {
        Tensor<double> value;
        std::mutex m;
        std::optional<MatrixEdge> matrix;
        std::optional<VectorEdge> vector;
    };
    std::vector<std::unique_ptr<Res>> results;

    std::vector<MatrixEdge> matrices;
    std::vector<vecfuncT> keepalive;   // host functions loaded into the graph
    std::vector<DeviceFunctions*> stored;
    std::vector<DeviceFunctions*> kept;

    explicit Impl(World& world) : world(world) {}

    const ttg::Buffer<mra::Domain<3>>& domain() {
        if (!db) db.emplace(device_domain());
        return *db;
    }

    Functions add_functions(NodeEdge e, std::size_t N) {
        fns.push_back(Fn{e, std::nullopt, N});
        return Functions{int(fns.size()) - 1, N};
    }

    Fn& get(Functions f) {
        MADNESS_CHECK_THROW(f.id >= 0 && f.id < int(fns.size()), "DeviceGraph: invalid functions handle");
        return fns[f.id];
    }

    RecEdge& reconstructed(Functions f) {
        Fn& fn = get(f);
        if (!fn.reconstructed) {
            fn.reconstructed.emplace();
            g.add(mra::make_reconstruct(make_mrattg_functionset(fn.N), g.K, false, g.functiondata,
                                        fn.compressed, *fn.reconstructed, "reconstruct"));
        }
        return *fn.reconstructed;
    }

    Res& add_result(long rows, long cols) {
        results.push_back(std::make_unique<Res>());
        Res& r = *results.back();
        r.value = cols > 0 ? Tensor<double>(rows, cols) : Tensor<double>(rows);
        return r;
    }

    /// sink a matrix arriving at the root into r.value, scaled by fac and accumulated
    void add_matrix_sink(MatrixEdge e, Res& r, double fac, const std::string& name) {
        g.add(ttg::make_tt([&r, fac](const mra::Key<3>& key, const mra::DenseTensor<double, 2>& m) {
            const double* data = m.current_view().data();
            std::lock_guard<std::mutex> lock(r.m);
            for (long i = 0; i < r.value.size(); ++i) r.value.ptr()[i] += fac * data[i];
        }, ttg::edges(e), ttg::edges(), name));
    }

    /// a task at the root computing a matrix on the host from the reduction arriving on `in`
    template <typename InT>
    Matrix add_decision(ttg::Edge<mra::Key<3>, InT> in, long rows, long cols, Decision f) {
        matrices.emplace_back();
        MatrixEdge out = matrices.back();
        g.add(ttg::make_tt([rows, cols, f](const mra::Key<3>& key, const InT& x) {
            Tensor<double> xh = cols > 0 ? Tensor<double>(rows, cols) : Tensor<double>(rows);
            std::copy(x.current_view().data(), x.current_view().data() + xh.size(), xh.ptr());
            Tensor<double> uh = f(xh);
            MADNESS_CHECK_THROW(uh.ndim() == 2, "DeviceGraph::decide: the decision must return a matrix");
            mra::DenseTensor<double, 2> u(std::array<mra::size_type, 2>{mra::size_type(uh.dim(0)),
                                                                        mra::size_type(uh.dim(1))});
            Tensor<double> uc = copy(uh);   // contiguous, row-major
            std::copy(uc.ptr(), uc.ptr() + uc.size(), u.buffer().host_ptr());
            ttg::send<0>(key, std::move(u));
        }, ttg::edges(in), ttg::edges(out), "decide"));
        return Matrix{int(matrices.size()) - 1};
    }

    MatrixEdge& matrix(Matrix m) {
        MADNESS_CHECK_THROW(m.id >= 0 && m.id < int(matrices.size()), "DeviceGraph: invalid matrix handle");
        return matrices[m.id];
    }
};

DeviceGraph::DeviceGraph(World& world) : pimpl(std::make_unique<Impl>(world)) {}
DeviceGraph::~DeviceGraph() = default;

DeviceGraph::Functions DeviceGraph::feed(DeviceFunctions& v, bool keep) {
    NodeEdge e;
    pimpl->g.add(fv(v).make_feed(pimpl->g.control, e, keep, "feed"));
    if (keep) pimpl->kept.push_back(&v);
    return pimpl->add_functions(e, v.size());
}

void DeviceGraph::store(Functions f, DeviceFunctions& v) {
    pimpl->g.add(fv(v).make_store(pimpl->get(f).compressed, f.size, "store"));
    pimpl->stored.push_back(&v);
}

DeviceGraph::Functions DeviceGraph::apply_local_potential(const functionT& v, Functions psi, double tol) {
    using T = double;
    auto& g = pimpl->g;
    const mra::size_type N = psi.size;
    auto fns = make_mrattg_functionset(N);
    pimpl->keepalive.push_back({copy(v)});   // the loader needs it reconstructed
    vecfuncT& vv = pimpl->keepalive.back();
    vv[0].reconstruct();
    RecEdge prod_e;
    NodeEdge v_e, comp_e;
    load_compressed_mrattg(vv, make_mrattg_functionset(1), g.K, g.functiondata, g.control, v_e, g.tts, "v");
    // multiplied in compressed form: psi is not reconstructed for this
    g.add(mra::make_multiply_compressed<T, 3>(pimpl->domain(), make_mrattg_functionset(1), fns, g.functiondata, g.K,
                                              v_e, pimpl->get(psi).compressed, prod_e, "V*psi"));
    add_compress(g.tts, fns, g.K, g.functiondata, prod_e, comp_e, tol, "compress_Vpsi");
    return pimpl->add_functions(comp_e, N);
}

namespace {
    /// out_j = sum_i in_i U(i,j) with U fixed or from an edge, truncated with tol unless tol == 0
    template <typename UT>
    NodeEdge add_transform(MraGraph& g, NodeEdge in, std::size_t N, std::size_t M, UT&& U, double tol) {
        auto fns_out = make_mrattg_functionset(M);
        NodeEdge out_e, trunc_e;
        g.add(mra::make_transform<double, 3>(make_mrattg_functionset(N), fns_out, g.K, std::forward<UT>(U), in, out_e));
        if (tol <= 0.0) return out_e;
        g.add(mra::make_truncate(fns_out, g.K, tol, FunctionDefaults<3>::get_truncate_mode(),
                                 FunctionDefaults<3>::get_cell_min_width(), out_e, trunc_e, "truncate"));
        return trunc_e;
    }
}

DeviceGraph::Functions DeviceGraph::transform(Functions in, const Tensor<double>& U, double tol) {
    MADNESS_CHECK_THROW(U.dim(0) == long(in.size), "DeviceGraph::transform: U must have in.size rows");
    const std::size_t M = U.dim(1);
    return pimpl->add_functions(add_transform(pimpl->g, pimpl->get(in).compressed, in.size, M, device_matrix(U), tol), M);
}

DeviceGraph::Functions DeviceGraph::transform(Functions in, Matrix U, std::size_t M, double tol) {
    return pimpl->add_functions(add_transform(pimpl->g, pimpl->get(in).compressed, in.size, M, pimpl->matrix(U), tol), M);
}

DeviceGraph::Functions DeviceGraph::concat(Functions a, Functions b) {
    auto& g = pimpl->g;
    NodeEdge out_e;
    g.add(mra::make_concat<double, 3>(make_mrattg_functionset(a.size), make_mrattg_functionset(b.size), g.K,
                                      pimpl->get(a).compressed, pimpl->get(b).compressed, out_e, "concat",
                                      ttg::Void{}, ttg::Void{}, pimpl->concat_group));
    return pimpl->add_functions(out_e, a.size + b.size);
}

DeviceGraph::Result DeviceGraph::matrix_inner(Functions a, Functions b) {
    auto& g = pimpl->g;
    Impl::Res& r = pimpl->add_result(long(a.size), long(b.size));
    r.matrix.emplace();
    g.add(mra::make_matrix_inner<double, 3>(make_mrattg_functionset(a.size), make_mrattg_functionset(b.size), g.K,
                                            pimpl->get(a).compressed, pimpl->get(b).compressed, *r.matrix,
                                            "matrix_inner", ttg::Void{}, ttg::Void{}, pimpl->inner_group));
    pimpl->add_matrix_sink(*r.matrix, r, 1.0, "matrix_inner_sink");
    return Result{int(pimpl->results.size()) - 1};
}

DeviceGraph::Result DeviceGraph::norms(Functions a) {
    auto& g = pimpl->g;
    Impl::Res& r = pimpl->add_result(long(a.size), 0);
    r.vector.emplace();
    g.add(mra::make_norm(make_mrattg_functionset(a.size), g.K, pimpl->get(a).compressed, *r.vector));
    g.add(ttg::make_tt([&r](const mra::Key<3>& key, const mra::DenseTensor<double, 1>& n) {
        std::copy(n.current_view().data(), n.current_view().data() + r.value.size(), r.value.ptr());
    }, ttg::edges(*r.vector), ttg::edges(), "norm_sink"));
    return Result{int(pimpl->results.size()) - 1};
}

DeviceGraph::Result DeviceGraph::kinetic_matrix(Functions psi) {
    using T = double;
    auto& g = pimpl->g;
    const mra::size_type N = psi.size;
    auto fns = make_mrattg_functionset(N);
    Impl::Res& r = pimpl->add_result(long(N), long(N));
    mra::BatchGroup compress_group;   // the three axes batch together
    // the three derivatives in one descent over the compressed orbitals, which are not reconstructed
    std::array<RecEdge, 3> d_e;
    g.add(mra::make_gradient_compressed<T, 3>(pimpl->domain(), fns, g.functiondata, g.K, pimpl->get(psi).compressed,
                                              d_e, "gradient"));
    for (mra::Dimension axis = 0; axis < 3; ++axis) {
        const std::string name = "d" + std::to_string(axis);
        NodeEdge dc_e;
        MatrixEdge m_e;
        g.add(mra::make_compress(fns, g.K, false, g.functiondata, d_e[axis], dc_e, "compress_" + name,
                                 ttg::Void{}, ttg::Void{}, false, T{}, 0, T{}, compress_group));
        g.add(mra::make_matrix_inner<T, 3>(fns, fns, g.K, dc_e, dc_e, m_e, "inner_" + name,
                                           ttg::Void{}, ttg::Void{}, pimpl->inner_group));
        pimpl->add_matrix_sink(m_e, r, 0.5, "sink_" + name);
    }
    return Result{int(pimpl->results.size()) - 1};
}

DeviceGraph::Matrix DeviceGraph::decide(Result r, Decision f) {
    MADNESS_CHECK_THROW(r.id >= 0 && r.id < int(pimpl->results.size()), "DeviceGraph: invalid result handle");
    Impl::Res& res = *pimpl->results[r.id];
    if (res.matrix) return pimpl->add_decision(*res.matrix, res.value.dim(0), res.value.dim(1), std::move(f));
    if (res.vector) return pimpl->add_decision(*res.vector, res.value.dim(0), 0, std::move(f));
    MADNESS_EXCEPTION("DeviceGraph::decide: this result is not a single reduction", r.id);
}

DeviceGraph::Matrix DeviceGraph::decide(Matrix m, Decision f) {
    // the size of the matrix is only known when it arrives
    MatrixEdge in = pimpl->matrix(m);
    pimpl->matrices.emplace_back();
    MatrixEdge out = pimpl->matrices.back();
    pimpl->g.add(ttg::make_tt([f](const mra::Key<3>& key, const mra::DenseTensor<double, 2>& x) {
        Tensor<double> xh(long(x.dim(0)), long(x.dim(1)));
        std::copy(x.current_view().data(), x.current_view().data() + xh.size(), xh.ptr());
        Tensor<double> uh = copy(f(xh));
        mra::DenseTensor<double, 2> u(std::array<mra::size_type, 2>{mra::size_type(uh.dim(0)), mra::size_type(uh.dim(1))});
        std::copy(uh.ptr(), uh.ptr() + uh.size(), u.buffer().host_ptr());
        ttg::send<0>(key, std::move(u));
    }, ttg::edges(in), ttg::edges(out), "decide"));
    return Matrix{int(pimpl->matrices.size()) - 1};
}

void DeviceGraph::execute() {
    pimpl->g.execute();
    for (auto* v : pimpl->stored) check_device_tree(*v, "DeviceGraph::store");
    for (auto* v : pimpl->kept) check_device_tree(*v, "DeviceGraph::feed(keep)");
    for (auto& r : pimpl->results) {   // only the process owning the root received them
        pimpl->world.gop.sum(r->value.ptr(), r->value.size());
    }
}

Tensor<double> DeviceGraph::get(Result r) const {
    MADNESS_CHECK_THROW(r.id >= 0 && r.id < int(pimpl->results.size()), "DeviceGraph: invalid result handle");
    return copy(pimpl->results[r.id]->value);
}

/// 1/2 sum_axis <d psi_i|d psi_j> (cf. SCF::kinetic_energy_matrix with the abgv derivative); psi is kept
Tensor<double> device_kinetic_matrix(World& world, DeviceFunctions& psi) {
    DeviceGraph g(world);
    auto r = g.kinetic_matrix(g.feed(psi, true));
    g.execute();
    return g.get(r);
}

} // namespace madness
