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

/// \file device/graph_impl.h
/// \brief DeviceGraph::Impl, shared by the translation units that implement DeviceGraph.
///
/// The DeviceGraph members are spread over graph.cc, graph_ops.cc and graph_reduce.cc
/// because each MRA-TTG operator they instantiate costs minutes of nvcc/gcc time; in one
/// TU they were the build's critical path. Keep non-template members that build operators
/// out of this header (define them in the TU that uses them): an inline body here would
/// instantiate its operator in every TU, used or not.

#ifndef MADNESS_CHEM_DEVICE_GRAPH_IMPL_H__INCLUDED
#define MADNESS_CHEM_DEVICE_GRAPH_IMPL_H__INCLUDED

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

    /// a set of functions: compressed nodes
    struct Fn {
        NodeEdge compressed;
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
        fns.push_back(Fn{e, N});
        return Functions{int(fns.size()) - 1, N};
    }

    Fn& get(Functions f) {
        MADNESS_CHECK_THROW(f.id >= 0 && f.id < int(fns.size()), "DeviceGraph: invalid functions handle");
        return fns[f.id];
    }

    Res& add_result(long rows, long cols) {
        results.push_back(std::make_unique<Res>());
        Res& r = *results.back();
        r.value = cols > 0 ? Tensor<double>(rows, cols) : Tensor<double>(rows);
        return r;
    }

    /// sink a matrix arriving at the root into r.value, scaled by fac and accumulated (graph_reduce.cc)
    void add_matrix_sink(MatrixEdge e, Res& r, double fac, const std::string& name);

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

} // namespace madness

#endif // MADNESS_CHEM_DEVICE_GRAPH_IMPL_H__INCLUDED
