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
#include <memory>
#include <vector>
#include <madness/chem/SCF.h>

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

    /// start and end a timer of the SCF (SCF.cc), printed with the SCF timings
    void scf_start_timer(World& world);
    void scf_end_timer(World& world, const char* msg);

    /// copy MADNESS functions into a device vector
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

    /// rneg = G(-2 (Vpsi - psi fockp)) - psi and its norms; psi is kept, Vpsi is consumed
    DeviceFunctions device_bsh_residual(World& world, DeviceFunctions& psi, DeviceFunctions& Vpsi,
                                        const Tensor<double>& fockp, const tensorT& eps,
                                        const CalculationParameters& param, std::vector<double>& rnorm);

    /// Vpsi_i = v psi_i for one (host) function v, truncated with tol unless tol == 0; psi is kept
    void device_apply_local_potential(World& world, const functionT& v, DeviceFunctions& psi,
                                      DeviceFunctions& Vpsi, double tol);

    /// sum_i occ_i psi_i^2, compressed, with the process map of `like`; psi is kept
    functionT device_density(World& world, DeviceFunctions& psi, const tensorT& occ, const functionT& like);

    /// 1/2 sum_axis <d psi_i|d psi_j> (abgv derivative); psi is kept
    Tensor<double> device_kinetic_matrix(World& world, DeviceFunctions& psi);

    /// compare the MRA-TTG inner products and transform against MADNESS on f and g (same size)
    void validate_mrattg_linalg(World& world, const vecfuncT& f, const vecfuncT& g);

} // namespace madness

#endif // MADNESS_CHEM_DEVICE_SCF_DEVICE_H__INCLUDED
