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

#ifndef MADNESS_CHEM_DEVICE_COMMON_H__INCLUDED
#define MADNESS_CHEM_DEVICE_COMMON_H__INCLUDED

/// \file device/common.h
/// \brief Helpers shared by the MRA-TTG graphs of the SCF (device/*.cc only).

#include <array>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <thread>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <madness/chem/SCF.h>
#include <mra/mra.h>
#include <mra/vmra/vmra.h>
#include <ttg.h>
#include "scf_device.h"

namespace madness {

struct DeviceFunctions::Impl {
    mra::FunctionVector<double, 3> v;
};

/// the MRA-TTG function vector behind a DeviceFunctions handle
inline mra::FunctionVector<double, 3>& fv(DeviceFunctions& d) { return d.impl().v; }

/// TTs of a graph whose discovered but not executed tasks are printed if the graph hangs
inline std::vector<ttg::TTBase*> mrattg_tt_list(const std::vector<std::unique_ptr<ttg::TTBase>>& tts,
                                                std::initializer_list<ttg::TTBase*> more = {}) {
    std::vector<ttg::TTBase*> list(more);
    for (const auto& tt : tts) list.push_back(tt.get());
    return list;
}

/// execute a graph from its start task, on all ranks.
///
/// Debugging aid: if the environment variable MRA_TTG_HANG_TIMEOUT is set to a number of
/// seconds and the graph has not completed by then, the tasks of `tts` (and of `start`)
/// that were discovered but have not executed are printed (TTBase::print_incomplete_tasks);
/// the graph keeps executing.
inline void execute_mra_ttg(auto&& start, const std::vector<ttg::TTBase*>& tts = {},
                            const std::function<void()>& on_hang = {}) {
    static const double timeout = [] {
        const char* v = std::getenv("MRA_TTG_HANG_TIMEOUT");
        return v ? std::atof(v) : 0.0;
    }();
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    std::thread watchdog;
    if (timeout > 0.0) {
        watchdog = std::thread([&] {
            std::unique_lock<std::mutex> lock(m);
            if (!cv.wait_for(lock, std::chrono::duration<double>(timeout), [&] { return done; })) {
                std::cerr << "MRA-TTG graph not completed after " << timeout << "s, incomplete tasks:" << std::endl;
                start->print_incomplete_tasks();
                for (auto* tt : tts) tt->print_incomplete_tasks();
                std::cout << std::flush;
                if (on_hang) on_hang();
                std::cerr << "MRA-TTG end of incomplete tasks" << std::endl;
            }
        });
    }
    auto connected = make_graph_executable(start.get());
    assert(connected);
    //ttg::trace_on();
    if (ttg::default_execution_context().rank() == 0) {
      // This kicks off the entire computation
      start->invoke();
    }
    ttg::execute();
    ttg::fence();
    {
        std::lock_guard<std::mutex> lock(m);
        done = true;
    }
    cv.notify_all();
    if (watchdog.joinable()) watchdog.join();
}

/// The keys of a tree held by a device vector, with each node's marks of which children exist
using TreeSnapshot = std::map<mra::Key<3>, std::array<bool, mra::Key<3>::num_children()>>;

inline TreeSnapshot snapshot_device_tree(DeviceFunctions& d) {
    TreeSnapshot snap;
    fv(d).for_each_local_node([&](const auto& key, const auto& node) {
        auto& marks = snap[key];
        for (auto child : children(key)) marks[child.childindex()] = !node.is_all_child_leaf(child);
    });
    return snap;
}

/// report violations of tree consistency on stderr: a node marks a child as present iff the
/// child is held, every node but the root has its parent. Single process only.
inline void report_tree(const TreeSnapshot& snap, const char* what) {
    long bad = 0;
    for (const auto& [key, marks] : snap) {
        if (key.level() > 0 && snap.count(key.parent()) == 0) {
            if (bad++ < 10) std::cerr << what << ": " << key << " has no parent" << std::endl;
        }
        for (auto child : children(key)) {
            const bool marked = marks[child.childindex()];
            const bool present = snap.count(child) > 0;
            if (marked != present) {
                if (bad++ < 10)
                    std::cerr << what << ": " << key << " marks child " << child << (marked ? " present" : " absent")
                              << " but it is " << (present ? "held" : "missing") << std::endl;
            }
        }
    }
    std::cerr << what << ": " << bad << " tree inconsistencies in " << snap.size() << " nodes" << std::endl;
}

/// Debugging aid: with MRA_TTG_CHECK_TREES set, check the tree held by a device vector.
inline void check_device_tree(DeviceFunctions& d, const char* what) {
    static const bool enabled = std::getenv("MRA_TTG_CHECK_TREES") != nullptr;
    if (!enabled) return;
    TreeSnapshot snap = snapshot_device_tree(d);
    long bad = 0;
    for (const auto& [key, marks] : snap) {
        if (key.level() > 0 && snap.count(key.parent()) == 0) ++bad;
        for (auto child : children(key)) bad += (marks[child.childindex()] != (snap.count(child) > 0));
    }
    if (bad) report_tree(snap, what);
}

/// Feed MADNESS functions into an MRA-TTG graph as compressed nodes on `out`.

/// They are loaded reconstructed (compressed trees are reconstructed first) and
/// compressed in the graph: a compressed MADNESS tree loaded as-is can carry
/// interior nodes without coefficients, which the loader marks as leafs while
/// still sending their children.
template <typename T, mra::Dimension NDIM, typename FunctionSetT>
void load_compressed_mrattg(const std::vector<Function<T, (std::size_t)NDIM>>& f,
                            const std::shared_ptr<FunctionSetT>& fns, mra::size_type K,
                            const mra::FunctionData<T, NDIM>& functiondata,
                            ttg::Edge<mra::Key<NDIM>, void>& control,
                            ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& out,
                            std::vector<std::unique_ptr<ttg::TTBase>>& tts,
                            const std::string& name) {
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> reconstructed;
    const auto state = f.front().get_impl()->get_tree_state();
    if (state == madness::TreeState::reconstructed) {
        tts.push_back(mra::vmra::make_vmra_load(f, control, reconstructed, "load_" + name));
    } else if (state == madness::TreeState::compressed) {
        ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> loaded;
        tts.push_back(mra::vmra::make_vmra_load(f, control, loaded, "load_" + name));
        tts.push_back(mra::make_reconstruct(fns, K, false, functiondata, loaded, reconstructed, "reconstruct_" + name));
    } else {
        throw std::runtime_error(name + " must be in reconstructed or compressed form");
    }
    tts.push_back(mra::make_compress(fns, K, false, functiondata, reconstructed, out, "compress_" + name));
}

/// Compress the reconstructed nodes on `in` onto `out`, truncated with tol unless tol == 0.
/// The truncation is folded into the compression, unless MRA_TTG_SEPARATE_TRUNCATE is set:
/// then a separate make_truncate pass follows, a workaround for MRA-TTG versions whose folded
/// truncation still sends nodes their parent drops.
template <typename FunctionSetT>
inline void add_compress(std::vector<std::unique_ptr<ttg::TTBase>>& tts, const std::shared_ptr<FunctionSetT>& fns,
                         mra::size_type K, const mra::FunctionData<double, 3>& functiondata,
                         ttg::Edge<mra::Key<3>, mra::FunctionsReconstructedNode<double, 3>>& in,
                         ttg::Edge<mra::Key<3>, mra::FunctionsCompressedNode<double, 3>>& out,
                         double tol, const std::string& name) {
    static const bool separate = std::getenv("MRA_TTG_SEPARATE_TRUNCATE") != nullptr;
    const int mode = FunctionDefaults<3>::get_truncate_mode();
    const double min_width = FunctionDefaults<3>::get_cell_min_width();
    if (tol <= 0.0 || !separate) {
        tts.push_back(mra::make_compress(fns, K, false, functiondata, in, out, name, ttg::Void{}, ttg::Void{},
                                         /*enable_truncate=*/tol > 0.0, tol, mode, min_width));
    } else {
        ttg::Edge<mra::Key<3>, mra::FunctionsCompressedNode<double, 3>> compressed;
        tts.push_back(mra::make_compress(fns, K, false, functiondata, in, compressed, name));
        tts.push_back(mra::make_truncate(fns, K, tol, mode, min_width, compressed, out, name + "-truncate"));
    }
}

/// Compress the non-standard nodes on `in` (e.g., a convolution result) onto `out` in a single
/// bottom-up pass, truncated as in add_compress. Replaces make_reconstruct(accumulate_NS = true)
/// followed by add_compress.
template <typename FunctionSetT>
inline void add_compress_from_ns(std::vector<std::unique_ptr<ttg::TTBase>>& tts, const std::shared_ptr<FunctionSetT>& fns,
                                 mra::size_type K, const mra::FunctionData<double, 3>& functiondata,
                                 ttg::Edge<mra::Key<3>, mra::FunctionsCompressedNode<double, 3>>& in,
                                 ttg::Edge<mra::Key<3>, mra::FunctionsCompressedNode<double, 3>>& out,
                                 double tol, const std::string& name) {
    static const bool separate = std::getenv("MRA_TTG_SEPARATE_TRUNCATE") != nullptr;
    const int mode = FunctionDefaults<3>::get_truncate_mode();
    const double min_width = FunctionDefaults<3>::get_cell_min_width();
    if (tol <= 0.0 || !separate) {
        tts.push_back(mra::make_compress_from_ns(fns, K, functiondata, in, out, name, ttg::Void{}, ttg::Void{},
                                                 /*enable_truncate=*/tol > 0.0, tol, mode, min_width));
    } else {
        ttg::Edge<mra::Key<3>, mra::FunctionsCompressedNode<double, 3>> compressed;
        tts.push_back(mra::make_compress_from_ns(fns, K, functiondata, in, compressed, name));
        tts.push_back(mra::make_truncate(fns, K, tol, mode, min_width, compressed, out, name + "-truncate"));
    }
}

/// function set of N functions in a single batch
inline auto make_mrattg_functionset(mra::size_type N) {
    auto pmap = mra::make_procmap<3>(N, 1 /* batch */);
    return mra::make_functionset<mra::Gaussian<double, 3>>(pmap.batch_manager());
}

// ---------------------------------------------------------------------------------------
// Orbitals held in device memory between MRA-TTG graphs (mra::FunctionVector). Each device_*
// function builds and executes one graph; vectors are fed into it and stored from it without
// their coefficients passing through the host.
// ---------------------------------------------------------------------------------------

using NodeEdge = ttg::Edge<mra::Key<3>, mra::FunctionsCompressedNode<double, 3>>;

/// boundary condition of the MRA-TTG derivative operators: that of FunctionDefaults, the same on all sides
inline mra::FunctionData<double, 3>::BCType device_bc() {
    const BoundaryConditions<3>& bc = FunctionDefaults<3>::get_bc();
    for (int d = 0; d < 3; ++d)
        for (int s = 0; s < 2; ++s)
            MADNESS_CHECK_THROW(bc(d, s) == bc(0, 0), "MRA-TTG: the boundary conditions must be the same on all sides");
    if (bc(0, 0) == BC_FREE) return mra::FunctionData<double, 3>::BC_FREE;
    if (bc(0, 0) == BC_ZERO) return mra::FunctionData<double, 3>::BC_ZERO;
    MADNESS_EXCEPTION("MRA-TTG: only free and zero boundary conditions are supported", bc(0, 0));
}

/// the tasks of one graph, started at the root key of a single batch
struct MraGraph {
    ttg::Edge<mra::Key<3>, void> control;
    std::vector<std::unique_ptr<ttg::TTBase>> tts;
    const mra::size_type K = FunctionDefaults<3>::get_k();
    mra::FunctionData<double, 3> functiondata{mra::size_type(FunctionDefaults<3>::get_k()), device_bc()};   // referenced by tasks

    std::function<void()> on_hang;   // debugging aid, see execute_mra_ttg

    void add(auto&& tt) { tts.push_back(std::move(tt)); }
    void execute() {
        auto fns = make_mrattg_functionset(1);
        auto start = mra::make_start(fns, control);
        execute_mra_ttg(start, mrattg_tt_list(tts), on_hang);
    }
};

/// N x M MADNESS matrix as the device matrix make_transform expects
inline std::shared_ptr<mra::DenseTensor<double, 2>> device_matrix(const Tensor<double>& U) {
    const mra::size_type N = U.dim(0), M = U.dim(1);
    auto Ud = std::make_shared<mra::DenseTensor<double, 2>>(std::array<mra::size_type, 2>{N, M});
    double* Uh = Ud->buffer().host_ptr();
    for (mra::size_type i = 0; i < N; ++i)
        for (mra::size_type j = 0; j < M; ++j) Uh[i*M + j] = U(i, j);
    return Ud;
}

/// the simulation cell as the MRA-TTG Domain (a cube, as for molecules)
inline ttg::Buffer<mra::Domain<3>> device_domain() {
    const Tensor<double>& cell = FunctionDefaults<3>::get_cell();
    for (int d = 1; d < 3; ++d)
        MADNESS_CHECK_THROW(cell(d, 0) == cell(0, 0) && cell(d, 1) == cell(0, 1),
                            "MRA-TTG: the simulation cell must be a cube");
    auto D = std::make_unique<mra::Domain<3>[]>(1);
    D[0].set_cube(cell(0, 0), cell(0, 1));
    return ttg::Buffer<mra::Domain<3>>(std::move(D), 1);
}

} // namespace madness

#endif // MADNESS_CHEM_DEVICE_COMMON_H__INCLUDED
