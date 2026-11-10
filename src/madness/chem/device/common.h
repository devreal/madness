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
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

/// MRA-TTG convolution operators kept across graphs, for one protocol (SCF::device_operators).
///
/// The device generates the 1D operator matrices of every distinct Gaussian term once per
/// level and keeps them in the operator's term tables (mra::GaussianTermTables). The Coulomb
/// operator does not change within a protocol, so its MRA-TTG operator is kept whole. The BSH
/// operators change every iteration, but only in their coefficients: their terms lie on a
/// lattice independent of the orbital energy, so the next iteration's operators find all their
/// terms in the tables of the previous ones and reuse the matrices generated so far.
/// The Poisson operator of the exchange (lo, thresh) differs from the Coulomb operator (lo,
/// 0.001 thresh); it is made here once per protocol, as the host Exchange makes it per call.
struct DeviceOperatorCache {
    using ConvOp = mra::GaussianConvolutionOperator<double, 3>;
    /// the operators of DeviceGraph::convolution (Coulomb, exchange), with their MRA-TTG operators
    std::vector<std::pair<poperatorT, std::shared_ptr<ConvOp>>> ops;
    /// the Poisson operator of the exchange (device_exchange), made on first use
    poperatorT exchange_op;
    /// the term tables of the last BSH operators
    std::shared_ptr<mra::GaussianTermTables<double, 3>> bsh_tables;
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
/// Debugging aid: the directory into which the dot (GraphViz) description of every MRA-TTG graph
/// is written right before it executes, NNNN-name.dot numbered in the order of execution (empty:
/// none). Set with set_device_dot_dir (moldft --mra-ttg-dot=dir) or MRA_TTG_DOT.
inline std::string& device_dot_dir_ref() {
    static std::string dir = [] {
        const char* v = std::getenv("MRA_TTG_DOT");
        return v ? std::string(v) : std::string();
    }();
    return dir;
}

inline void execute_mra_ttg(auto&& start, const std::vector<ttg::TTBase*>& tts = {},
                            const std::function<void()>& on_hang = {}, const std::string& name = "graph") {
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
    static int graph_id = 0;
    const int this_graph = graph_id++;
    // Debugging aid: the graph's dot description (device_dot_dir_ref), written by one process, as
    // every process builds the same graph; the traversal from the start reaches all of it
    if (const std::string& dir = device_dot_dir_ref(); !dir.empty() && ttg::default_execution_context().rank() == 0) {
        std::filesystem::create_directories(dir);
        char file[32];
        std::snprintf(file, sizeof(file), "%04d-", this_graph);
        std::ofstream(std::filesystem::path(dir) / (file + name + ".dot")) << ttg::Dot(true)(start.get());
    }
    // Debugging aid: with MRA_TTG_TRACE set (and TTG configured with TTG_ENABLE_TRACE), every graph
    // is traced to std::clog, between lines marking its begin and end
    static const bool trace = std::getenv("MRA_TTG_TRACE") != nullptr;
    if (trace) {
        std::clog << "=== MRA-TTG graph " << this_graph << " begin" << std::endl;
        ttg::trace_on();
    }
    if (ttg::default_execution_context().rank() == 0) {
      // This kicks off the entire computation
      start->invoke();
    }
    ttg::execute();
    ttg::fence();
    if (trace) {
        ttg::trace_off();
        std::clog << "=== MRA-TTG graph " << this_graph << " end" << std::endl;
    }
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

/// The batches of a set of functions in the graphs of the SCF (mra::BatchLayout)
using DeviceLayout = std::shared_ptr<const mra::BatchLayout>;

/// N functions in a single batch on all processes
inline DeviceLayout one_batch(mra::size_type N) {
    return std::make_shared<const mra::BatchLayout>(mra::BatchLayout::even(N, 1));
}

/// whether two layouts have the same batches (sizes), as operations on pairs of functions need
inline bool same_batches(const mra::BatchLayout& a, const mra::BatchLayout& b) {
    if (a.num_batches() != b.num_batches()) return false;
    for (mra::Batch i = 0; i < a.num_batches(); ++i)
        if (a.batch_size(i) != b.batch_size(i)) return false;
    return true;
}

/// MADNESS_CHECK_THROW for a message built at run time: MadnessException keeps only a pointer
/// to its message, which would outlive a temporary string
inline void device_check(bool ok, const std::string& msg) {
    if (!ok) throw std::runtime_error(msg);
}

/// Operations that combine functions of different batches (transforms, concatenation, matrices
/// of inner products) do not support several batches yet: check that a set has one.
inline void require_one_batch(const mra::BatchLayout& layout, const std::string& what) {
    device_check(layout.num_batches() == 1, what + ": functions in several batches are not supported yet");
}

/// a sink for per-function values arriving per batch (norms, <a_i|b_i>), at the root of each
/// batch: writes them into `value` at the batch's offset
inline auto make_batch_vector_sink(ttg::Edge<mra::Key<3>, mra::DenseTensor<double, 1>> in, Tensor<double>& value,
                                   const DeviceLayout& layout, const std::string& name) {
    return ttg::make_tt([&value, layout](const mra::Key<3>& key, const mra::DenseTensor<double, 1>& v) {
        const mra::Batch b = key.batch();
        std::copy(v.current_view().data(), v.current_view().data() + layout->batch_size(b),
                  value.ptr() + layout->offset(b));
    }, ttg::edges(in), ttg::edges(), name);
}

/// The number of batches the orbitals are split into in the device graphs (and every set of
/// their size, uploaded with device_upload), at most one per function: each batch on its own
/// group of processes (mra::BatchGroups). Process-wide, like FunctionDefaults; set it before orbitals are
/// uploaded, with set_device_orbital_batches or MRA_TTG_ORBITAL_BATCHES (default 1).
inline mra::Batch& device_orbital_batches_ref() {
    static mra::Batch nb = [] {
        const char* v = std::getenv("MRA_TTG_ORBITAL_BATCHES");
        return mra::Batch(v ? std::max(1, std::atoi(v)) : 1);
    }();
    return nb;
}


/// N orbitals (or functions of the orbitals, such as V psi) in the orbitals' batches, at most
/// one batch per function (a single function, e.g. a potential, is one batch on all processes).
/// Every set of N functions gets the same layout.
inline DeviceLayout device_orbital_layout(mra::size_type N) {
    const mra::Batch nb = mra::Batch(std::min<mra::size_type>(device_orbital_batches_ref(), std::max<mra::size_type>(N, 1)));
    return std::make_shared<const mra::BatchLayout>(mra::BatchLayout::even(N, nb));
}

/// The process map of the MRA-TTG graphs of the SCF: a node goes to the group of its batch,
/// and within it where MADNESS's process map (FunctionDefaults<3>) puts it, shared by the
/// group: with an SFCPmap the group's members take its parts in contiguous blocks
/// (SFCPmap::group_part). A group of all processes uses MADNESS's map as it is, so a node
/// is on the process that holds it in MADNESS and loading and storing host functions stays
/// on the process. Device vectors hold their nodes on the process the map gives them when
/// stored; fed into a graph after the map changed (SCF::loadbal), they move to it.
struct DeviceProcMap {
    std::shared_ptr<WorldDCPmapInterface<Key<3>>> pmap = FunctionDefaults<3>::get_pmap();
    std::shared_ptr<const SFCPmap<3>> sfc = std::dynamic_pointer_cast<const SFCPmap<3>>(pmap);
    mra::BatchGroups groups;   // of the set's batches (its layout's)
    explicit DeviceProcMap(const mra::BatchGroups& groups) : groups(groups) {}
    int operator()(const mra::Key<3>& key) const {
        const Key<3> k = key.to_madness_key();
        const auto [first, last] = groups.batch_pes(key.batch());
        const int size = last - first;
        if (size == groups.num_pes()) return pmap->owner(k);
        if (sfc) return first + sfc->group_part(k, size).first;
        return first + pmap->owner(k) % size;
    }
};

/// The device of a node within its process. Batches that share a process share its devices,
/// each in its share of them (mra::BatchGroups::batch_devices); within the share, a node goes
/// where the map assigns it (SFCPmap::device, within the batch's group as
/// SFCPmap::group_part, scaled to the share) or, for a map without devices, by a hash of the
/// node's ancestor at a coarse level, so that nearby nodes share a device.
struct DeviceDevMap {
    std::shared_ptr<const SFCPmap<3>> sfc =
        std::dynamic_pointer_cast<const SFCPmap<3>>(FunctionDefaults<3>::get_pmap());
    mra::BatchGroups groups;   // of the set's batches (its layout's)
    int ndevice;
    DeviceDevMap(const mra::BatchGroups& groups, int ndevice) : groups(groups), ndevice(ndevice) {}
    int operator()(const mra::Key<3>& key) const {
        const Key<3> k = key.to_madness_key();
        const auto [d0, d1] = groups.batch_devices(key.batch(), ndevice);
        const int width = d1 - d0;
        if (width == 1) return d0;
        if (sfc) {
            const auto [first, last] = groups.batch_pes(key.batch());
            const int part = sfc->group_part(k, last - first).second % ndevice;   // in [0, ndevice)
            return d0 + part * width / ndevice;
        }
        constexpr Level coarse = 3;
        return d0 + int((k.level() <= coarse ? k : k.parent(k.level() - coarse)).hash() % width);
    }
};

/// The maps of the tasks of a set of functions: by the groups of its batches (its layout's, or
/// that of the function set's layout). The orbitals' batches are on their groups; a set in one
/// batch (the density, potentials) is on all processes, as MADNESS holds such functions.
/// Without an argument: a set in one batch.
inline DeviceProcMap device_procmap(const mra::BatchGroups& groups) { return DeviceProcMap(groups); }
inline DeviceProcMap device_procmap(const DeviceLayout& layout) { return device_procmap(layout->groups()); }
template <typename FunctionSetT>
    requires requires(const FunctionSetT& f) { f.batchman()->groups(); }
inline DeviceProcMap device_procmap(const std::shared_ptr<FunctionSetT>& fns) {
    return device_procmap(fns->batchman()->groups());
}
inline DeviceProcMap device_procmap() {
    return device_procmap(mra::BatchGroups(1, ttg::default_execution_context().size()));
}

/// Host builds run every task on the host and take no device map (TTG would ask it for a device).
#if defined(MRA_ENABLE_HOST)
inline ttg::Void device_devicemap(const mra::BatchGroups&) { return {}; }
#else
inline DeviceDevMap device_devicemap(const mra::BatchGroups& groups) {
    return DeviceDevMap(groups, std::max(1, ttg::device::num_devices()));
}
#endif
inline auto device_devicemap(const DeviceLayout& layout) { return device_devicemap(layout->groups()); }
template <typename FunctionSetT>
    requires requires(const FunctionSetT& f) { f.batchman()->groups(); }
inline auto device_devicemap(const std::shared_ptr<FunctionSetT>& fns) {
    return device_devicemap(fns->batchman()->groups());
}
inline auto device_devicemap() {
    return device_devicemap(mra::BatchGroups(1, ttg::default_execution_context().size()));
}

/// Feed MADNESS functions into an MRA-TTG graph as compressed nodes on `out`.

/// Compressed trees are loaded as-is; reconstructed trees are compressed in the graph.
template <typename T, mra::Dimension NDIM, typename FunctionSetT>
void load_compressed_mrattg(const std::vector<Function<T, (std::size_t)NDIM>>& f,
                            const std::shared_ptr<FunctionSetT>& fns, mra::size_type K,
                            const mra::FunctionData<T, NDIM>& functiondata,
                            ttg::Edge<mra::Key<NDIM>, void>& control,
                            ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& out,
                            std::vector<std::unique_ptr<ttg::TTBase>>& tts,
                            const std::string& name) {
    const auto state = f.front().get_impl()->get_tree_state();
    if (state == madness::TreeState::reconstructed) {
        ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> reconstructed;
        tts.push_back(mra::vmra::make_vmra_load(f, control, reconstructed, "load_" + name,
                                                std::shared_ptr<const mra::BatchLayout>(fns->batchman())));
        tts.push_back(mra::make_compress(fns, K, false, functiondata, reconstructed, out, "compress_" + name,
                                         device_procmap(fns), device_devicemap(fns)));
    } else if (state == madness::TreeState::compressed) {
        tts.push_back(mra::vmra::make_vmra_load(f, control, out, "load_" + name,
                                                std::shared_ptr<const mra::BatchLayout>(fns->batchman())));
    } else {
        throw std::runtime_error(name + " must be in reconstructed or compressed form");
    }
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
        tts.push_back(mra::make_compress(fns, K, false, functiondata, in, out, name, device_procmap(fns), device_devicemap(fns),
                                         /*enable_truncate=*/tol > 0.0, tol, mode, min_width));
    } else {
        ttg::Edge<mra::Key<3>, mra::FunctionsCompressedNode<double, 3>> compressed;
        tts.push_back(mra::make_compress(fns, K, false, functiondata, in, compressed, name,
                                         device_procmap(fns), device_devicemap(fns)));
        tts.push_back(mra::make_truncate(fns, K, tol, mode, min_width, compressed, out, name + "-truncate",
                                         device_procmap(fns), device_devicemap(fns)));
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
        tts.push_back(mra::make_compress_from_ns(fns, K, functiondata, in, out, name, device_procmap(fns),
                                                 device_devicemap(fns), /*enable_truncate=*/tol > 0.0, tol, mode,
                                                 min_width));
    } else {
        ttg::Edge<mra::Key<3>, mra::FunctionsCompressedNode<double, 3>> compressed;
        tts.push_back(mra::make_compress_from_ns(fns, K, functiondata, in, compressed, name,
                                                 device_procmap(fns), device_devicemap(fns)));
        tts.push_back(mra::make_truncate(fns, K, tol, mode, min_width, compressed, out, name + "-truncate",
                                         device_procmap(fns), device_devicemap(fns)));
    }
}

/// the batches a device vector was stored in (one batch if it never was)
inline DeviceLayout layout_of(DeviceFunctions& d) { return fv(d).layout() ? fv(d).layout() : one_batch(d.size()); }

/// the function set of a layout: its batches (the functions themselves are not used)
inline auto make_mrattg_functionset(const DeviceLayout& layout) {
    return mra::make_functionset<mra::Gaussian<double, 3>>(std::const_pointer_cast<mra::BatchLayout>(layout));
}

/// function set of N functions in a single batch
inline auto make_mrattg_functionset(mra::size_type N) { return make_mrattg_functionset(one_batch(N)); }

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
    std::string name = "graph";   // of its dot file (device_dot_dir_ref)
    ttg::Edge<mra::Key<3>, void> control;
    std::vector<std::unique_ptr<ttg::TTBase>> tts;
    const mra::size_type K = FunctionDefaults<3>::get_k();
    mra::FunctionData<double, 3> functiondata{mra::size_type(FunctionDefaults<3>::get_k()), device_bc()};   // referenced by tasks

    std::function<void()> on_hang;   // debugging aid, see execute_mra_ttg

    void add(auto&& tt) { tts.push_back(std::move(tt)); }
    void execute() {
        auto fns = make_mrattg_functionset(1);
        auto start = mra::make_start(fns, control);
        execute_mra_ttg(start, mrattg_tt_list(tts), on_hang, name);
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
