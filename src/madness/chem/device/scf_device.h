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

#ifndef MADNESS_CHEM_DEVICE_SCF_DEVICE_H__INCLUDED
#define MADNESS_CHEM_DEVICE_SCF_DEVICE_H__INCLUDED

/// \file device/scf_device.h
/// \brief MRA-TTG operations of the SCF, compiled in device/ separately from SCF.cc.
///
/// This header does not include TTG or MRA-TTG so that SCF.cc compiles as plain C++;
/// the graphs are built in device/*.cc (as CUDA/HIP sources in device builds).

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <madness/chem/SCF.h>
#include <madness/mra/lbdeux.h>

namespace mra { class BatchLayout; }

namespace madness {

    /// Functions held by MRA-TTG between graphs (an mra::FunctionVector), with their
    /// coefficients in device memory. A handle: movable, not copyable.
    class DeviceFunctions {
    public:
        struct Impl;   // device/common.h

        DeviceFunctions();
        ~DeviceFunctions();
        DeviceFunctions(DeviceFunctions&&) noexcept;
        DeviceFunctions& operator=(DeviceFunctions&&) noexcept;
        DeviceFunctions(const DeviceFunctions&) = delete;
        DeviceFunctions& operator=(const DeviceFunctions&) = delete;

        /// number of functions
        std::size_t size() const;

        Impl& impl();

    private:
        std::unique_ptr<Impl> pimpl;
    };

    /**
     * One MRA-TTG graph built from several operations that pass sets of functions
     * along edges, instead of storing them into DeviceFunctions after each graph and
     * feeding them into the next. The caller decides which sets to persist (store()).
     *
     * Reductions (matrix_inner, norms, kinetic_matrix) yield results on the host after
     * execute(). A reduction can also feed decide(), which computes a matrix from it
     * on the host while the graph runs; transform() then applies that matrix, its input
     * nodes waiting in the graph until the matrix is there. decide() runs inside a
     * task on the process owning the root of the tree: the callback must not use
     * MADNESS collectives (world.gop), and host state it writes is set on that process
     * only.
     *
     * Handles are only valid for the graph that made them, until it is executed.
     */
    class DeviceGraph {
    public:
        struct Impl;
        /// a set of functions flowing through the graph (compressed nodes)
        struct Functions { int id = -1; std::size_t size = 0; };
        /// a reduction, available on the host (all processes) after execute()
        struct Result { int id = -1; };
        /// a matrix computed while the graph executes
        struct Matrix { int id = -1; };
        using Decision = std::function<Tensor<double>(const Tensor<double>&)>;

        explicit DeviceGraph(World& world);
        ~DeviceGraph();
        DeviceGraph(const DeviceGraph&) = delete;
        DeviceGraph& operator=(const DeviceGraph&) = delete;

        /// the functions of v; without keep, v is empty afterwards and can be stored into
        Functions feed(DeviceFunctions& v, bool keep);
        /// persist f into v, which must be empty (or fed without keep in this graph)
        void store(Functions f, DeviceFunctions& v);
        /// one host function, compressed in the graph unless it is already
        Functions load(const functionT& f);
        /// persist one function f into the host function out, compressed, with the process map
        /// of `like`; out is set by execute()
        void store(Functions f, functionT& out, const functionT& like);

        /// v psi_i for one (host) function v, truncated with tol unless tol == 0
        Functions apply_local_potential(const functionT& v, Functions psi, double tol);
        /// the same for one function v of this graph
        Functions apply_local_potential(Functions v, Functions psi, double tol);
        /// sum_i occ_i psi_i^2 (cf. SCF::make_density): one function
        Functions density(Functions psi, const tensorT& occ);
        /// a_i b_i (a and b of the same size) or a_0 b_i (a of size 1), compressed, truncated
        /// with tol unless tol == 0
        Functions multiply(Functions a, Functions b, double tol, const std::string& name = "multiply");
        /// op(in_i) for one convolution operator (e.g., Coulomb), compressed and truncated with
        /// tol unless tol == 0. With a cache, the MRA-TTG operator made for op is kept there and
        /// reused, with every level of 1D matrices it generated, while op stays the same.
        Functions convolution(const poperatorT& op, Functions in, double tol, DeviceOperatorCache* cache = nullptr);
        /// a x_i + b y_i, truncated with tol unless tol == 0
        Functions gaxpy(double a, Functions x, double b, Functions y, double tol);
        /// sum_k c[k] f[k]_i for two or more sets f[k] of the same size, not truncated. Every node of the
        /// result is new, so the sets may be fed with keep (see mra::FunctionVector).
        Functions combine(const std::vector<double>& c, const std::vector<Functions>& f);
        /// out_j = sum_i in_i U(i,j), truncated with tol unless tol == 0
        Functions transform(Functions in, const Tensor<double>& U, double tol, const std::string& name = "transform");
        /// the same with U computed in the graph; M is the number of columns of U. With the
        /// functions in several batches, block_diagonal declares that U does not couple them
        /// (e.g., a scaling): each batch then transforms its own, without communication.
        Functions transform(Functions in, Matrix U, std::size_t M, double tol, bool block_diagonal = false);
        /// [a, b]: the functions of a followed by those of b
        Functions concat(Functions a, Functions b);

        /// the batches of a set: their sizes and process groups
        using Layout = std::shared_ptr<const mra::BatchLayout>;
        /// the batches of f
        Layout layout(Functions f);
        /// a set in the batches of `out`, batch t holding the functions of the batch b of in with
        /// t in targets(b) (each t in the targets of one b, of b's size): b's nodes are sent to the
        /// group of t. in is only read (it may be used again).
        Functions rebatch(Functions in, Layout out, std::function<std::vector<int>(int)> targets,
                          const std::string& name = "rebatch");
        /// batch c of a set in the batches of `out`: the sum of the batches b of the sets
        /// sources[k].first with sources[k].second(b) == c (-1: not summed); every batch of out
        /// receives at least one, of its size. The sets are only read.
        Functions sum_into(const std::vector<std::pair<Functions, std::function<int(int)>>>& sources, Layout out,
                           const std::string& name = "sum");
        /// out_j = sum_i in_i U_b(i,j) within each batch b, from the batches of in to those of
        /// `out` (as many): blocks[b] has b's sizes in in and out, truncated with tol unless tol == 0
        Functions transform(Functions in, const std::vector<Tensor<double>>& blocks, Layout out, double tol,
                            const std::string& name = "transform");

        /// <a_i|b_j>; a and b may be the same
        Result matrix_inner(Functions a, Functions b);
        /// <a_i|b_i> for a and b of the same size
        Result inner(Functions a, Functions b);
        /// ||a_i||
        Result norms(Functions a);
        /// 1/2 sum_axis <d psi_i|d psi_j> (abgv derivative)
        Result kinetic_matrix(Functions psi);

        /// a matrix computed from a reduction (matrix_inner or norms) while the graph runs
        Matrix decide(Result r, Decision f);
        /// a matrix computed from another one
        Matrix decide(Matrix m, Decision f);

        /// build and run the graph (collective)
        void execute();
        /// a reduction's value, after execute()
        Tensor<double> get(Result r) const;

    private:
        std::unique_ptr<Impl> pimpl;
    };

    /// start and end a timer of the SCF (SCF.cc), printed with the SCF timings
    void scf_start_timer(World& world);
    void scf_end_timer(World& world, const char* msg);

    /// Marks a device-offloaded region of the SCF for the profiler: an NVTX range in the
    /// "MRA-TTG SCF" domain from construction to destruction (CUDA builds; a no-op otherwise).
    /// Must be destroyed on the thread that created it.
    class DeviceRange {
    public:
        explicit DeviceRange(const char* name);
        ~DeviceRange();
        DeviceRange(const DeviceRange&) = delete;
        DeviceRange& operator=(const DeviceRange&) = delete;
    };

    /// the number of devices MRA-TTG uses per process: the largest of all processes (collective)
    int device_count(World& world);

    /// The number of batches the orbitals are split into on the device (device_upload of a set of
    /// orbitals or of functions of them): each batch is held and processed by its own group of
    /// processes. Process-wide; set it on all processes before uploading orbitals. Default 1, or
    /// MRA_TTG_ORBITAL_BATCHES.
    void set_device_orbital_batches(int nbatch);
    int device_orbital_batches();

    /// Adds the cost of the nodes of a device vector on this process to the load balancer:
    /// per function, 8 for a node with children and 1 for a leaf (lbcost(1, 8)). A function's
    /// leafs are not nodes of the vector; they are added as the children marked leaf.
    /// Not while a graph executes.
    void device_add_lb_cost(LoadBalanceDeux<3>& lb, DeviceFunctions& v);

    /// copy MADNESS functions into a device vector: a set of orbitals (or of functions of them) in
    /// the orbitals' batches (set_device_orbital_batches), a single function on all processes
    DeviceFunctions device_upload(World& world, const vecfuncT& f);

    /// copy a device vector into (compressed) MADNESS functions with the process map of `like`
    vecfuncT device_download(World& world, DeviceFunctions& d, const functionT& like);

    /// <a_i|b_j>; a and b may be the same vector
    Tensor<double> device_matrix_inner(World& world, DeviceFunctions& a, DeviceFunctions& b);

    /// ||a_i||
    std::vector<double> device_norms(World& world, DeviceFunctions& a);

    /// v_j <- sum_i v_i U(i,j), truncated with tol unless tol == 0 (cf. madness::transform)
    void device_transform(World& world, DeviceFunctions& v, const Tensor<double>& U, double tol);

    /// [a, b]: the functions of a followed by those of b, in a new vector; a and b are kept or consumed
    DeviceFunctions device_concat(World& world, DeviceFunctions& a, bool keep_a, DeviceFunctions& b, bool keep_b);

    /// out_j = sum_i v_i U(i,j) in a new vector, truncated with tol unless tol == 0; v is kept
    DeviceFunctions device_transformed(World& world, DeviceFunctions& v, const Tensor<double>& U, double tol);

    /// ||sum_i v_i U(i,j)|| for every column j of U; v is kept
    std::vector<double> device_transform_norms(World& world, DeviceFunctions& v, const Tensor<double>& U);

    /// v_i <- c_i v_i
    void device_scale(World& world, DeviceFunctions& v, const std::vector<double>& c);

    /// v_i <- v_i / ||v_i||
    void device_normalize(World& world, DeviceFunctions& v);

    /// a_i <- a_i - c_i r_i; consumes r
    void device_update(World& world, DeviceFunctions& a, DeviceFunctions& r, const std::vector<double>& c);

    /// rneg = G(-2 (Vpsi - psi fockp)) - psi and its norms; psi is kept, Vpsi is consumed. With a
    /// cache, the BSH operators reuse the 1D matrices of earlier ones (see DeviceOperatorCache).
    DeviceFunctions device_bsh_residual(World& world, DeviceFunctions& psi, DeviceFunctions& Vpsi,
                                        const Tensor<double>& fockp, const tensorT& eps,
                                        const CalculationParameters& param, std::vector<double>& rnorm,
                                        DeviceOperatorCache* cache = nullptr);

    /// Vpsi_i = v psi_i for one (host) function v, truncated with tol unless tol == 0; psi is kept
    void device_apply_local_potential(World& world, const functionT& v, DeviceFunctions& psi,
                                      DeviceFunctions& Vpsi, double tol);

    /// tile of device_exchange: chosen by device_exchange_tile
    inline constexpr std::size_t device_exchange_auto_tile = std::size_t(-1);

    /// the number of orbital pairs per exchange graph whose data fits in device memory (0: all of
    /// them), estimated from the size of the orbitals (their nonzero nodes), k and the device memory
    /// pools of all processes. Collective.
    std::size_t device_exchange_tile(World& world, DeviceFunctions& psi);

    /// the Hartree-Fock exchange K psi_i = sum_j psi_j G(psi_j psi_i) with the Poisson operator G
    /// (lo, thresh) of the host Exchange, in tiles of `tile` pairs i <= j (0: all in one graph;
    /// device_exchange_auto_tile: see device_exchange_tile); psi is kept. G is kept in the cache.
    DeviceFunctions device_exchange(World& world, DeviceFunctions& psi, double lo, DeviceOperatorCache& cache,
                                    std::size_t tile = device_exchange_auto_tile);

    /// sum_i occ_i psi_i^2, compressed, with the process map of `like`; psi is kept
    functionT device_density(World& world, DeviceFunctions& psi, const tensorT& occ, const functionT& like);

    /// 1/2 sum_axis <d psi_i|d psi_j> (abgv derivative); psi is kept
    Tensor<double> device_kinetic_matrix(World& world, DeviceFunctions& psi);

    /// compare the MRA-TTG inner products and transform against MADNESS on f and g (same size)
    void validate_mrattg_linalg(World& world, const vecfuncT& f, const vecfuncT& g);

    /// print max ||ref_i - d_i|| and max ||ref_i||, d downloaded with the process map of `like`; d is kept
    void validate_device_functions(World& world, const vecfuncT& ref, DeviceFunctions& d, const functionT& like,
                                   const std::string& what);

    /// compare the load-balancing cost of device vector d (device_add_lb_cost) with that of the
    /// same functions ref on the host
    void validate_device_lb_cost(World& world, const vecfuncT& ref, DeviceFunctions& d);

    /// print max |ref - d| and max |ref|
    void validate_device_values(World& world, const Tensor<double>& ref, const Tensor<double>& d,
                                const std::string& what);
    void validate_device_values(World& world, const std::vector<double>& ref, const std::vector<double>& d,
                                const std::string& what);

} // namespace madness

#endif // MADNESS_CHEM_DEVICE_SCF_DEVICE_H__INCLUDED
