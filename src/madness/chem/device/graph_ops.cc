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

/// \file device/graph_ops.cc
/// \brief DeviceGraph operations producing functions: local potential, transform, concat.

#include "graph_impl.h"

namespace madness {

DeviceGraph::Functions DeviceGraph::load(const functionT& f) {
    auto& g = pimpl->g;
    pimpl->keepalive.push_back({f});   // the loader refers to it until the graph has run
    NodeEdge e;
    load_compressed_mrattg(pimpl->keepalive.back(), make_mrattg_functionset(1), g.K, g.functiondata, g.control, e,
                           g.tts, "host");
    return pimpl->add_functions(e, 1);
}

void DeviceGraph::store(Functions f, functionT& out, const functionT& like) {
    MADNESS_CHECK_THROW(f.size == 1, "DeviceGraph::store: one function can be stored to the host");
    pimpl->host_out.emplace_back(1);
    vecfuncT& v = pimpl->host_out.back();
    v[0].set_impl(like, false);
    pimpl->g.add(mra::vmra::make_vmra_store(v, pimpl->get(f).compressed, true /* add_leaves */,
                                            madness::TreeState::compressed, "store_host", pimpl->get(f).layout));
    pimpl->host_targets.push_back(&out);
}

DeviceGraph::Functions DeviceGraph::gaxpy(double a, Functions x, double b, Functions y, double tol) {
    MADNESS_CHECK_THROW(x.size == y.size, "DeviceGraph::gaxpy: x and y must have the same size");
    const DeviceLayout layout = pimpl->get(x).layout;
    MADNESS_CHECK_THROW(same_batches(*layout, *pimpl->get(y).layout), "DeviceGraph::gaxpy: x and y in different batches");
    auto& g = pimpl->g;
    auto fns = make_mrattg_functionset(layout);
    NodeEdge out_e, trunc_e;
    g.add(mra::make_gaxpy(a, b, fns, g.K, pimpl->get(x).compressed, pimpl->get(y).compressed, out_e, "gaxpy",
                          device_procmap(layout), device_devicemap(layout)));
    if (tol <= 0.0) return pimpl->add_functions(out_e, layout);
    g.add(mra::make_truncate(fns, g.K, tol, FunctionDefaults<3>::get_truncate_mode(),
                             FunctionDefaults<3>::get_cell_min_width(), out_e, trunc_e, "gaxpy-truncate", device_procmap(layout), device_devicemap(layout)));
    return pimpl->add_functions(trunc_e, layout);
}

DeviceGraph::Functions DeviceGraph::combine(const std::vector<double>& c, const std::vector<Functions>& f) {
    MADNESS_CHECK_THROW(f.size() >= 2 && c.size() == f.size(), "DeviceGraph::combine: two or more sets, a coefficient each");
    auto& g = pimpl->g;
    const std::size_t N = f.front().size;
    const DeviceLayout layout = pimpl->get(f.front()).layout;
    for (const auto& x : f) {
        MADNESS_CHECK_THROW(x.size == N, "DeviceGraph::combine: the sets must have the same size");
        MADNESS_CHECK_THROW(same_batches(*layout, *pimpl->get(x).layout), "DeviceGraph::combine: the sets in different batches");
    }
    auto fns = make_mrattg_functionset(layout);
    // a chain of gaxpys, none forwarding an input node: c0 f0 + c1 f1, then + c2 f2, ...
    NodeEdge acc = pimpl->get(f.front()).compressed;
    double ca = c.front();
    for (std::size_t k = 1; k < f.size(); ++k) {
        NodeEdge out_e;
        g.add(mra::make_gaxpy(ca, c[k], fns, g.K, acc, pimpl->get(f[k]).compressed, out_e, "combine",
                              device_procmap(layout), device_devicemap(layout), /*forward=*/false));
        acc = out_e;
        ca = 1.0;
    }
    return pimpl->add_functions(acc, layout);
}

DeviceGraph::Functions DeviceGraph::apply_local_potential(const functionT& v, Functions psi, double tol) {
    return apply_local_potential(load(v), psi, tol);
}

DeviceGraph::Functions DeviceGraph::apply_local_potential(Functions v, Functions psi, double tol) {
    MADNESS_CHECK_THROW(v.size == 1, "DeviceGraph::apply_local_potential: one potential applies to all functions");
    return multiply(v, psi, tol);
}

DeviceGraph::Functions DeviceGraph::multiply(Functions a, Functions b, double tol, const std::string& name) {
    using T = double;
    MADNESS_CHECK_THROW(a.size == b.size || a.size == 1, "DeviceGraph::multiply: a must have the size of b, or 1");
    const DeviceLayout layout = pimpl->get(b).layout;
    auto& g = pimpl->g;
    NodeEdge a_e = pimpl->get(a).compressed;
    DeviceLayout a_layout = pimpl->get(a).layout;
    if (a.size == b.size) {
        MADNESS_CHECK_THROW(same_batches(*a_layout, *layout), "DeviceGraph::multiply: a and b in different batches");
    } else if (layout->num_batches() > 1) {
        // the single factor, held in batch 0, in every batch of b: one copy per process group
        MADNESS_CHECK_THROW(a_layout->num_batches() == 1, "DeviceGraph::multiply: a single factor must be in one batch");
        NodeEdge rep_e;
        g.add(mra::make_replicate<double, 3>(a_e, rep_e, layout->num_batches(), name + "-replicate",
                                             device_procmap(a_layout), device_devicemap(a_layout)));
        a_e = rep_e;
        a_layout = std::make_shared<const mra::BatchLayout>(
                     std::vector<mra::size_type>(layout->num_batches(), 1), layout->groups());
    }
    const mra::size_type N = b.size;
    auto fns = make_mrattg_functionset(layout);
    RecEdge prod_e;
    NodeEdge comp_e;
    // multiplied in compressed form: the factors are not reconstructed for this
    g.add(mra::make_multiply_compressed<T, 3>(pimpl->domain(), make_mrattg_functionset(a_layout), fns, g.functiondata,
                                              g.K, a_e, pimpl->get(b).compressed, prod_e,
                                              name, device_procmap(layout), device_devicemap(layout)));
    add_compress(g.tts, fns, g.K, g.functiondata, prod_e, comp_e, tol, name + "-compress");
    return pimpl->add_functions(comp_e, layout);
}

namespace {
    /// out_j = sum_i in_i U(i,j) with U fixed or from an edge (at the root of each batch, cut to
    /// it), truncated with tol unless tol == 0
    template <typename UT>
    NodeEdge add_transform(MraGraph& g, NodeEdge in, const DeviceLayout& in_layout, const DeviceLayout& out_layout,
                           UT&& U, double tol, const std::string& name) {
        auto fns_out = make_mrattg_functionset(out_layout);
        NodeEdge out_e, trunc_e;
        g.add(mra::make_transform<double, 3>(make_mrattg_functionset(in_layout), fns_out, g.K, std::forward<UT>(U), in,
                                             out_e, name, device_procmap(out_layout), device_devicemap(out_layout)));
        if (tol <= 0.0) return out_e;
        g.add(mra::make_truncate(fns_out, g.K, tol, FunctionDefaults<3>::get_truncate_mode(),
                                 FunctionDefaults<3>::get_cell_min_width(), out_e, trunc_e, name + "-truncate",
                                 device_procmap(out_layout), device_devicemap(out_layout)));
        return trunc_e;
    }

    /// The layout of the result of a transform of a set in `in` by an N x M matrix: with
    /// several batches, each transforms its own functions (U block-diagonal, M == N); a single
    /// batch of M otherwise.
    DeviceLayout transform_layout(const DeviceLayout& in, std::size_t M, const std::string& name) {
        if (in->num_batches() == 1) return one_batch(M);
        MADNESS_CHECK_THROW(M == in->num_functions(),
                            (name + ": with several batches, U must be square (block-diagonal in the batches)").c_str());
        return in;
    }
}

namespace {
    /// whether U (N x N) is block-diagonal in the batches of `layout`: no function of a batch
    /// contributes to the outputs of another
    bool block_diagonal(const Tensor<double>& U, const DeviceLayout& layout) {
        for (mra::Batch b = 0; b < layout->num_batches(); ++b) {
            const long o = layout->offset(b), n = layout->batch_size(b);
            for (long i = 0; i < U.dim(0); ++i) {
                if (i >= o && i < o + n) continue;
                for (long j = o; j < o + n; ++j) if (U(i, j) != 0.0) return false;
            }
        }
        return true;
    }

    /// U, arriving once at the root of batch 0 on U_full (N x N), cut into blocks sent to the
    /// roots of `blocks`, block p being U's rows of batch rows(p) and columns of batch cols(p)
    template <typename Rows, typename Cols>
    MatrixEdge distribute_blocks(MraGraph& g, MatrixEdge U_full, const DeviceLayout& layout, std::size_t nblocks,
                                 Rows rows, Cols cols, bool check_others_zero, const std::string& name) {
        MatrixEdge blocks;
        g.add(ttg::make_tt([layout, nblocks, rows, cols, check_others_zero](const mra::Key<3>& key,
                                                                            const mra::DenseTensor<double, 2>& u) {
            const mra::size_type N = u.dim(0);
            const double* p = u.current_view().data();
            for (std::size_t q = 0; q < nblocks; ++q) {
                const mra::Batch bi = rows(q), bo = cols(q);
                const mra::size_type oi = layout->offset(bi), ni = layout->batch_size(bi);
                const mra::size_type oo = layout->offset(bo), no = layout->batch_size(bo);
                if (check_others_zero) {   // block-diagonal: nothing from other batches into bo
                    for (mra::size_type i = 0; i < N; ++i) {
                        if (i >= oi && i < oi + ni) continue;
                        for (mra::size_type j = oo; j < oo + no; ++j)
                            MADNESS_CHECK_THROW(p[i*N + j] == 0.0, "DeviceGraph::transform: U couples the functions of different batches");
                    }
                }
                mra::DenseTensor<double, 2> blk(std::array<mra::size_type, 2>{ni, no});
                double* b = blk.buffer().host_ptr();
                for (mra::size_type i = 0; i < ni; ++i)
                    for (mra::size_type j = 0; j < no; ++j) b[i*no + j] = p[(oi + i)*N + oo + j];
                ttg::send<0>(mra::Key<3>(mra::Batch(q), mra::Level(0)), std::move(blk));
            }
        }, ttg::edges(U_full), ttg::edges(blocks), name + "-distribute"));
        return blocks;
    }

    /**
     * out_j = sum_i in_i U(i,j) for a set in several batches and a U that couples them (N x N,
     * arriving once at the root of batch 0). Each pair of batches (bi, bo) transforms the
     * functions of bi by U's block of bi's rows and bo's columns, under the key of the pair on
     * the group of bi, where bi's nodes are; the partial results of the pairs (k, bo) of each
     * input batch k are then moved to bo, and those of all k summed (gaxpys, which reconcile
     * their trees): this is where the batches' groups exchange functions.
     */
    NodeEdge add_dense_transform(MraGraph& g, NodeEdge in, const DeviceLayout& layout, MatrixEdge U_full,
                                 double tol, const std::string& name) {
        const mra::Batch nb = layout->num_batches();
        MADNESS_CHECK_THROW(long(nb) * nb < 32767, "DeviceGraph::transform: too many pairs of batches");
        std::vector<std::pair<int, int>> ranges;
        std::vector<mra::size_type> sizes_in, sizes_out;
        for (mra::Batch bi = 0; bi < nb; ++bi) {
            for (mra::Batch bo = 0; bo < nb; ++bo) {
                ranges.push_back(layout->groups().batch_pes(bi));
                sizes_in.push_back(layout->batch_size(bi));
                sizes_out.push_back(layout->batch_size(bo));
            }
        }
        const mra::BatchGroups pg(ranges, layout->num_pes());
        const DeviceLayout pin = std::make_shared<const mra::BatchLayout>(sizes_in, pg);
        const DeviceLayout pout = std::make_shared<const mra::BatchLayout>(sizes_out, pg);
        // the input of batch bi to its pairs (bi, bo), q = bi*nb + bo
        NodeEdge in_p, part;
        g.add(mra::make_rebatch<double, 3>(in, in_p, [nb](mra::Batch bi) {
            std::vector<mra::Batch> q(nb);
            for (mra::Batch bo = 0; bo < nb; ++bo) q[bo] = mra::Batch(bi*nb + bo);
            return q;
        }, name + "-pairs", device_procmap(layout), device_devicemap(layout)));
        MatrixEdge U_p = distribute_blocks(g, U_full, layout, std::size_t(nb) * nb,
                                           [nb](std::size_t q) { return mra::Batch(q / nb); },
                                           [nb](std::size_t q) { return mra::Batch(q % nb); }, false, name);
        g.add(mra::make_transform<double, 3>(make_mrattg_functionset(pin), make_mrattg_functionset(pout), g.K, U_p,
                                             in_p, part, name, device_procmap(pout), device_devicemap(pout)));
        // the partial results of input batch k, moved to their output batches, summed over k
        auto stage = [&](mra::Batch k) {
            NodeEdge e;
            g.add(mra::make_rebatch<double, 3>(part, e, [nb, k](mra::Batch q) {
                return (q / nb == k) ? std::vector<mra::Batch>{mra::Batch(q % nb)} : std::vector<mra::Batch>{};
            }, name + "-stage-" + std::to_string(k), device_procmap(pout), device_devicemap(pout)));
            return e;
        };
        auto fns = make_mrattg_functionset(layout);
        NodeEdge sum = stage(0);
        for (mra::Batch k = 1; k < nb; ++k) {
            NodeEdge next;
            g.add(mra::make_gaxpy(1.0, 1.0, fns, g.K, sum, stage(k), next, "transform-sum",
                                  device_procmap(layout), device_devicemap(layout)));
            sum = next;
        }
        if (tol <= 0.0) return sum;
        NodeEdge trunc_e;
        g.add(mra::make_truncate(fns, g.K, tol, FunctionDefaults<3>::get_truncate_mode(),
                                 FunctionDefaults<3>::get_cell_min_width(), sum, trunc_e, name + "-truncate",
                                 device_procmap(layout), device_devicemap(layout)));
        return trunc_e;
    }
}

DeviceGraph::Functions DeviceGraph::transform(Functions in, const Tensor<double>& U, double tol, const std::string& name) {
    MADNESS_CHECK_THROW(U.dim(0) == long(in.size), "DeviceGraph::transform: U must have in.size rows");
    const DeviceLayout in_layout = pimpl->get(in).layout;
    const DeviceLayout out_layout = transform_layout(in_layout, U.dim(1), "DeviceGraph::transform");
    if (in_layout->num_batches() > 1 && !block_diagonal(U, in_layout)) {
        // a U that couples the batches: sent once into the graph, at the root of batch 0
        auto& g = pimpl->g;
        auto Uh = std::make_shared<Tensor<double>>(copy(U));
        MatrixEdge U_full;
        g.add(ttg::make_tt([Uh](const mra::Key<3>& key) {
            if (key.batch() != 0) return;
            ttg::send<0>(key, std::move(*device_matrix(*Uh)));
        }, ttg::edges(g.control), ttg::edges(U_full), name + "-U"));
        return pimpl->add_functions(add_dense_transform(g, pimpl->get(in).compressed, in_layout, U_full, tol, name),
                                    out_layout);
    }
    // MRA-TTG cuts U to each batch's block (one batch: all of it)
    return pimpl->add_functions(add_transform(pimpl->g, pimpl->get(in).compressed, in_layout, out_layout,
                                              device_matrix(U), tol, name), out_layout);
}

DeviceGraph::Functions DeviceGraph::transform(Functions in, Matrix U, std::size_t M, double tol, bool block_diagonal) {
    const DeviceLayout in_layout = pimpl->get(in).layout;
    const DeviceLayout out_layout = transform_layout(in_layout, M, "DeviceGraph::transform");
    MatrixEdge U_e = pimpl->matrix(U);
    if (in_layout->num_batches() > 1 && !block_diagonal) {
        return pimpl->add_functions(add_dense_transform(pimpl->g, pimpl->get(in).compressed, in_layout, U_e, tol,
                                                        "transform"), out_layout);
    }
    if (in_layout->num_batches() > 1) {
        // block-diagonal: each batch's root gets its block (checked when it arrives)
        U_e = distribute_blocks(pimpl->g, U_e, in_layout, in_layout->num_batches(),
                                [](std::size_t b) { return mra::Batch(b); }, [](std::size_t b) { return mra::Batch(b); },
                                true, "transform");
    }
    return pimpl->add_functions(add_transform(pimpl->g, pimpl->get(in).compressed, in_layout, out_layout, U_e, tol,
                                              "transform"), out_layout);
}

DeviceGraph::Layout DeviceGraph::layout(Functions f) {
    return pimpl->get(f).layout;
}

DeviceGraph::Functions DeviceGraph::rebatch(Functions in, Layout out, std::function<std::vector<int>(int)> targets,
                                            const std::string& name) {
    const DeviceLayout in_layout = pimpl->get(in).layout;
    auto to = [out, in_layout, targets, name](mra::Batch b) {
        std::vector<mra::Batch> t;
        for (int x : targets(b)) {
            MADNESS_CHECK_THROW(x >= 0 && x < out->num_batches() && out->batch_size(mra::Batch(x)) == in_layout->batch_size(b),
                                (name + ": a target batch of another size").c_str());
            t.push_back(mra::Batch(x));
        }
        return t;
    };
    NodeEdge out_e;
    pimpl->g.add(mra::make_rebatch<double, 3>(pimpl->get(in).compressed, out_e, std::move(to), name,
                                              device_procmap(in_layout), device_devicemap(in_layout)));
    return pimpl->add_functions(out_e, out);
}

DeviceGraph::Functions DeviceGraph::transform(Functions in, const std::vector<Tensor<double>>& blocks, Layout out,
                                              double tol, const std::string& name) {
    const DeviceLayout in_layout = pimpl->get(in).layout;
    const mra::Batch nb = in_layout->num_batches();
    MADNESS_CHECK_THROW(out->num_batches() == nb && long(blocks.size()) == nb,
                        (name + ": one block per batch, as many batches in and out").c_str());
    auto bh = std::make_shared<std::vector<mra::DenseTensor<double, 2>>>();
    for (mra::Batch b = 0; b < nb; ++b) {
        MADNESS_CHECK_THROW(blocks[b].dim(0) == long(in_layout->batch_size(b)) && blocks[b].dim(1) == long(out->batch_size(b)),
                            (name + ": a block of the wrong size").c_str());
        bh->push_back(std::move(*device_matrix(blocks[b])));
    }
    // each batch's block, sent once into the graph to the root of the batch
    auto& g = pimpl->g;
    MatrixEdge U_e;
    g.add(ttg::make_tt([bh](const mra::Key<3>& key) {
        if (key.batch() != 0) return;
        for (std::size_t b = 0; b < bh->size(); ++b)
            ttg::send<0>(mra::Key<3>(mra::Batch(b), mra::Level(0)), std::move((*bh)[b]));
    }, ttg::edges(g.control), ttg::edges(U_e), name + "-U"));
    return pimpl->add_functions(add_transform(g, pimpl->get(in).compressed, in_layout, out, U_e, tol, name), out);
}

DeviceGraph::Functions DeviceGraph::concat(Functions a, Functions b) {
    require_one_batch(*pimpl->get(a).layout, "DeviceGraph::concat");
    require_one_batch(*pimpl->get(b).layout, "DeviceGraph::concat");
    auto& g = pimpl->g;
    NodeEdge out_e;
    g.add(mra::make_concat<double, 3>(make_mrattg_functionset(a.size), make_mrattg_functionset(b.size), g.K,
                                      pimpl->get(a).compressed, pimpl->get(b).compressed, out_e, "concat",
                                      device_procmap(), device_devicemap(), pimpl->concat_group));
    return pimpl->add_functions(out_e, a.size + b.size);
}

} // namespace madness
