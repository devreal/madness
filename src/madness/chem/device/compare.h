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

#ifndef MADNESS_CHEM_DEVICE_COMPARE_H__INCLUDED
#define MADNESS_CHEM_DEVICE_COMPARE_H__INCLUDED

/// \file device/compare.h
/// \brief Node-by-node comparison of MADNESS functions, to validate MRA-TTG results.

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <madness/mra/mra.h>

namespace madness {

  namespace detail {
    std::string madfunc_state(auto& madfunc) {
      std::string state;
      if (madfunc.is_compressed()) state += "compressed ";
      if (madfunc.is_nonstandard()) state += "nonstandard ";
      if (madfunc.is_reconstructed()) state += "reconstructed ";
      if (madfunc.is_redundant()) state += "redundant ";
      return state;
    }
  } // namespace detail

  template<typename T, std::size_t NDIM>
  inline void compare_mra_madness(const std::vector<madness::Function<T, NDIM>>& madfunc1,
                                  const std::vector<madness::Function<T, NDIM>>& madfunc2,
                                  const std::string name, T precision = 1e-15,
                                  bool ignore_zero = true,
                                  bool fail_on_mismatch = false)
  {
    if (madfunc1.size() != madfunc2.size()) {
      std::cout << name << ": number of functions in MADNESS vector 1 (" << madfunc1.size() << ") does not match number of functions in MADNESS vector 2 (" << madfunc2.size() << ")" << std::endl;
      if (fail_on_mismatch) {
        throw std::runtime_error(name + ": mismatch in number of functions between MADNESS vectors");
      }
    }
    for (std::size_t i = 0; i < madfunc1.size(); ++i) {
      auto impl1 = madfunc1[i].get_impl();
      auto impl2 = madfunc2[i].get_impl();
      if (impl1->get_k() != impl2->get_k()) {
        std::cout << name << ": MADNESS function " << i << " in vector 1 has k=" << impl1->get_k() << " but in vector 2 has k=" << impl2->get_k() << std::endl;
        if (fail_on_mismatch) {
          throw std::runtime_error(name + ": mismatch in k between MADNESS vectors");
        }
      }
      auto tree1_size = impl1->tree_size();
      auto tree2_size = impl2->tree_size();
      if (tree1_size != tree2_size) {
        std::cout << name << ": MADNESS function " << i << " in vector 1 has "
        << tree1_size << " nodes but in vector 2 has "
        << tree2_size << " nodes" << std::endl;
      }
    }
    bool check = true;
    for (std::size_t i = 0; i < madfunc1.size(); ++i) {
      std::unordered_map<madness::Key<NDIM>, bool> node_seen_map;
      auto impl1 = madfunc1[i].get_impl();
      auto impl2 = madfunc2[i].get_impl();
      if (impl1->get_tree_state() != impl2->get_tree_state()) {
        std::cout << name << ": MADNESS function " << i << " in vector 1 is in state "
                  << detail::madfunc_state(madfunc1[i]) << " but in vector 2 is in state "
                  << detail::madfunc_state(madfunc2[i]) << std::endl;
        check = false;
        continue;
      }
      for (auto& node1 : impl1->get_coeffs()) {
        auto node2 = impl2->get_coeffs().find(node1.first);
        auto node1_norm = node1.second.coeff().normf();
        if (node2.get() == impl2->get_coeffs().end()) {
          if ((node1_norm > precision || !ignore_zero)) {
            std::cout << name << ": node " << node1.first << " in MADNESS vector 1 function "
                        << i << " (norm " << node1_norm
                        << ") not found in MADNESS vector 2" << std::endl;
            check = false;
          }
          continue;
        }

        node_seen_map[node1.first] = true;

        auto norm1 = node1.second.coeff().normf();
        auto norm2 = node2.get()->second.coeff().normf();
        if (std::abs(norm1 - norm2) > precision) {
          std::cout << name << ": node " << node1.first << " in MADNESS function " << i
                    << " has norm " << norm1 << " in vector 1 but norm " << norm2 << " in vector 2 (absdiff "
                    << std::abs(norm1 - norm2) << ")" << std::endl;
          check = false;
          continue;
        }

        if (!ignore_zero && node2.get()->second.has_children() != node1.second.has_children()) {
          std::cout << name << ": node " << node1.first << " in MADNESS vector 1 function " << i
                    << " has children " << node1.second.has_children() << " but in vector 2 has children "
                    << node2.get()->second.has_children() << std::endl;
          check = false;
          /* non-fatal error */
        }
      }

      // check if all nodes in vector 2 are in vector 1
      for (auto& node2 : impl2->get_coeffs()) {
        if (node_seen_map.find(node2.first) == node_seen_map.end()) {
            if (node2.second.has_coeff() &&
                (node2.second.coeff().normf() > precision || !ignore_zero) ) {
                std::cout << name << ": node " << node2.first << " in MADNESS vector 2 function "
                          << i << " (norm " << node2.second.coeff().normf()
                          << ") not found in MADNESS vector 1" << std::endl;
                check = false;
                continue;
            }
        }
      }
    }
    if (!check) {
      if (fail_on_mismatch) {
        throw std::runtime_error(name + ": mismatch in MADNESS vectors");
      }
    } else {
      std::cout << name << ": all nodes match between MADNESS vectors" << std::endl;
    }
  }

} // namespace madness

#endif // MADNESS_CHEM_DEVICE_COMPARE_H__INCLUDED
