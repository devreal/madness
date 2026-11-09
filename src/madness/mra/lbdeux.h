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

  $Id$
*/
#ifndef MADNESS_MRA_IBDEUX_H__INCLUDED
#define MADNESS_MRA_IBDEUX_H__INCLUDED

#include <madness/madness_config.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <madness/world/atomicint.h>
#include <madness/world/worlddc.h>

#include <madness/mra/key.h>
#include <madness/mra/funcdefaults.h>

/// \file mra/lbdeux.h
/// \brief Implements (2nd generation) static load/data balancing for functions
/// \ingroup function

namespace madness {

	template<typename T, std::size_t NDIM>
	class FunctionNode;

	template<typename T, std::size_t NDIM>
	class Function;

    template <std::size_t NDIM>
    class LBDeuxPmap : public WorldDCPmapInterface< Key<NDIM> > {
        typedef Key<NDIM> keyT;
        typedef std::pair<keyT,ProcessID> pairT;
        typedef std::map<keyT,ProcessID> mapT;
        mapT map;
        typedef typename mapT::const_iterator iteratorT;

    public:
        LBDeuxPmap(const std::vector<pairT>& v) {
            for (unsigned int i=0; i<v.size(); ++i) {
                map.insert(v[i]);
            }
        }

        // If level 0 is not entered as a node this will
        // be an infinite loop.
        ProcessID owner(const keyT& key) const {
            while (key.level() >= 0) {
                iteratorT it = map.find(key);
                if (it == map.end()) {
                    return owner(key.parent());
                }
                else {
                    return it->second;
                }
            }
            madness::print("Mon Dieux!", key);
            throw "LBDeuxPmap: lookup failed";
        }

        void print() const {
            madness::print("LBDeuxPmap");
        }
    };


    /// Position of boxes along the Morton (Z-order) space-filling curve.

    /// The curve is resolved at the finest level whose translations fit 64 bits interleaved
    /// (max_level = 63/NDIM); a deeper box takes the position of its ancestor at that level.
    /// A box covers the contiguous range [code(key), code(key) + extent(key)) of the curve,
    /// and so does its subtree.
    template <std::size_t NDIM>
    struct SFCCode {
        static constexpr Level max_level = 63/NDIM;

        static uint64_t code(const Key<NDIM>& key) {
            const Level n = key.level();
            std::array<uint64_t, NDIM> t;   // translations at max_level
            for (std::size_t d=0; d<NDIM; ++d) {
                const uint64_t l = key.translation()[d];
                t[d] = (n <= max_level) ? (l << (max_level - n)) : (l >> (n - max_level));
            }
            uint64_t c = 0;
            for (int b=max_level-1; b>=0; --b) {
                for (std::size_t d=0; d<NDIM; ++d) c = (c << 1) | ((t[d] >> b) & 1);
            }
            return c;
        }

        static uint64_t extent(const Key<NDIM>& key) {
            const Level n = key.level();
            return (n >= max_level) ? 1 : (uint64_t(1) << (NDIM*(max_level - n)));
        }
    };


    /// A process map that cuts the space-filling curve (SFCCode) into contiguous pieces.

    /// Made by LoadBalanceDeux::load_balance_sfc. It has two levels: the piece of each process
    /// is cut further into one piece per device of the process, for runtimes that distribute
    /// the work of a process across several devices (device()). A key belongs to the piece
    /// holding the start of its box, so a box spanning pieces goes with its first descendant.
    /// Lookup is a binary search over the pieces (a few per device).
    template <std::size_t NDIM>
    class SFCPmap : public WorldDCPmapInterface< Key<NDIM> > {
        std::vector<uint64_t> start;    ///< first code of each piece, ascending, start[0] == 0
        std::vector<int> part;          ///< the piece's part, process*ndevice + device
        int nproc;
        int ndevice;
        std::vector<double> part_cost;  ///< cost assigned to each part by the load balancer

        int find(const Key<NDIM>& key) const {
            const uint64_t c = SFCCode<NDIM>::code(key);
            return part[std::upper_bound(start.begin(), start.end(), c) - start.begin() - 1];
        }

    public:
        /// pieces: (first code, part) in ascending order of codes, the first at code 0, with parts
        /// in [0, nproc*ndevice)
        SFCPmap(const std::vector<std::pair<uint64_t,int>>& pieces, int nproc, int ndevice,
                std::vector<double> part_cost = {})
            : nproc(nproc), ndevice(ndevice), part_cost(std::move(part_cost)) {
            MADNESS_CHECK_THROW(nproc > 0 && ndevice > 0, "SFCPmap: need at least one process and device");
            MADNESS_CHECK_THROW(!pieces.empty() && pieces.front().first == 0, "SFCPmap: the first piece must start at 0");
            for (const auto& [c, p] : pieces) {
                MADNESS_CHECK_THROW(start.empty() || c > start.back(), "SFCPmap: pieces must ascend");
                MADNESS_CHECK_THROW(p >= 0 && p < nproc*ndevice, "SFCPmap: part out of range");
                start.push_back(c);
                part.push_back(p);
            }
        }

        ProcessID owner(const Key<NDIM>& key) const override { return find(key)/ndevice; }

        /// the device of the owner that key belongs to, in [0, num_devices())
        int device(const Key<NDIM>& key) const { return find(key)%ndevice; }

        int num_devices() const { return ndevice; }

        int num_processes() const { return nproc; }

        /// the part, process*num_devices() + device, that key belongs to
        int part_of(const Key<NDIM>& key) const { return find(key); }

        /// The part of key when the cut is shared by a group of `size` processes (and their
        /// devices), as (member of the group in [0, size), device): the group's members and
        /// devices take the parts in contiguous blocks, in order, so they get about equal shares
        /// of the cost, exactly when size divides the number of processes. A group of all
        /// processes gets (owner, device).
        std::pair<int,int> group_part(const Key<NDIM>& key, int size) const {
            MADNESS_CHECK(size > 0 && size <= nproc);
            const long s = long(find(key)) * size / nproc;   // in [0, size*ndevice)
            return {int(s / ndevice), int(s % ndevice)};
        }

        /// number of contiguous pieces of the curve
        std::size_t num_pieces() const { return start.size(); }

        /// cost assigned to each part (process*num_devices() + device) by the load balancer
        const std::vector<double>& costs() const { return part_cost; }

        void print() const override {
            madness::print("SFCPmap:", start.size(), "pieces,", ndevice, "devices per process");
        }
    };



    template <std::size_t NDIM>
    class LBNodeDeux {
        static const int nchild = (1<<NDIM);
        typedef Key<NDIM> keyT;
        typedef LBNodeDeux<NDIM> nodeT;
        typedef WorldContainer<keyT,nodeT> treeT;
        
        double child_cost[nchild]; // Removed volatile since all parallel updates will be under mutex from active message and no unprotected reads
        double my_cost;
        double total_cost;
        bool gotkids;
        
        AtomicInt nsummed;

        /// Computes index of child key in this node using last bit of translations
        int index(const keyT& key) {
            int ind = 0;
            for (std::size_t d=0; d<NDIM; ++d) ind += ((key.translation()[d])&0x1) << d;
            return ind;
        }

    public:
        LBNodeDeux()
                : my_cost(0.0), total_cost(0.0), gotkids(false) {
            nsummed = 0;
            for (int i=0; i<nchild; ++i)
                child_cost[i] = 0.0;
        }

        LBNodeDeux(const LBNodeDeux<NDIM>& other) :
            my_cost(other.my_cost), total_cost(other.total_cost), gotkids(other.gotkids)
        {
            nsummed = other.nsummed;
            for (int i=0; i<nchild; ++i)
                child_cost[i] = other.child_cost[i];
        }

        LBNodeDeux<NDIM>& operator=(const LBNodeDeux<NDIM>& other) {
            for (int i=0; i<nchild; ++i)
                child_cost[i] = other.child_cost[i];
            my_cost = other.my_cost;
            total_cost = other.total_cost;
            gotkids = other.gotkids;
            nsummed = other.nsummed;

            return *this;
        }

        bool has_children() const {
            return gotkids;
        }

        double get_total_cost() const {
            return total_cost;
        }

        /// Accumulates cost into this node
        void add(double cost, bool got_kids) {
            total_cost = (my_cost += cost);
            gotkids = gotkids || got_kids;
        }

        /// Accumulates cost up the tree from children
        void sum(const treeT& tree, const keyT& child, double value) {
            child_cost[index(child)] = value;
            ++nsummed;
            if (nsummed == nchild) {
                for (int i=0; i<nchild; ++i) total_cost += child_cost[i];
                if (child.level() > 1) {
                    keyT key = child.parent();
                    keyT parent = key.parent();
                    const_cast<treeT&>(tree).task(parent, &nodeT::sum, tree, key, double(total_cost));
                }
            }
        }


        /// Logically deletes this node by setting cost to -1

        /// Cannot actually erase this node from the container since the send() handler
        /// is holding an accessor to it.
        void deleter(const treeT& tree, const keyT& key) {
            total_cost = my_cost = -1.0;
            if (has_children()) {
                for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                    const keyT child = kit.key();
                    const_cast<treeT&>(tree).task(child, &nodeT::deleter, tree, child);
                }
            }
        }

        /// Descends tree deleting all except internal nodes and sub-tree parents
        void partition(const treeT& tree, const keyT& key, double avg) {
            if (has_children()) {
                // Sort children in descending cost order
                keyT keys[nchild];
                double vals[nchild];
                for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                    const keyT child = kit.key();
                    int ind = index(child);
                    keys[ind] = child;
                    vals[ind] = child_cost[ind];
                }
                for (int i=0; i<nchild; ++i) {
                    for (int j=i+1; j<nchild; ++j) {
                        if (vals[i] < vals[j]) {
                            std::swap(vals[i],vals[j]);
                            std::swap(keys[i],keys[j]);
                        }
                    }
                }

                // Split off subtrees in decreasing cost order
                for (int i=0; i<nchild; ++i) {
                    if (total_cost <= avg) {
                        const_cast<treeT&>(tree).task(keys[i], &nodeT::deleter, tree, keys[i]);
                    }
                    else {
                        total_cost -= vals[i];
                        const_cast<treeT&>(tree).task(keys[i], &nodeT::partition, tree, keys[i], avg);
                    }
                }
            }
        }

        template <typename Archive>
        void serialize(Archive& ar) {
            ar & archive::wrap_opaque(this,1);
        }
    };


    template <std::size_t NDIM>
    class LoadBalanceDeux {
        typedef Key<NDIM> keyT;
        typedef LBNodeDeux<NDIM> nodeT;
        typedef WorldContainer<keyT,nodeT> treeT;
        typedef typename treeT::iterator iteratorT;
        typedef typename treeT::const_iterator const_iteratorT;
        World& world;
        treeT tree;


        template <typename T, typename costT>
        struct add_op {
            LoadBalanceDeux* lb;
            const costT& costfn;
            add_op(LoadBalanceDeux* lb, const costT& costfn) : lb(lb), costfn(costfn) {}
            void operator()(const keyT& key, const FunctionNode<T,NDIM>& node) const {
 	        if (lb->tree.is_local(key))
		    lb->tree.send(key, &nodeT::add, costfn(key,node), node.has_children());
		else
		    lb->tree.task(key, &nodeT::add, costfn(key,node), node.has_children());
            }
        };

        /// Sums costs up the tree returning to everyone the total cost
        double sum() {
            world.gop.fence();
            const_iteratorT end = tree.end();
            for (const_iteratorT it=tree.begin(); it!=end; ++it) {
                const keyT& key = it->first;
                const nodeT& node = it->second;
                if (!node.has_children() && key.level() > 0) {
                    tree.task(key.parent(), &nodeT::sum, tree, key, node.get_total_cost());
                }
            }
            world.gop.fence();
            double total;
            keyT key0(0);
            if (world.rank() == tree.owner(key0)) {
                total = tree.find(key0).get()->second.get_total_cost();
            }
            world.gop.broadcast(total, tree.owner(key0));
            world.gop.fence();

            return total;
        }

        /// Used to sort results into descending order
        static bool compare(const std::pair<keyT,double>& a, const std::pair<keyT,double>& b) {
            return a.second < b.second;
        }


    public:
        LoadBalanceDeux(World& world)
                : world(world)
                , tree(world, FunctionDefaults<NDIM>::get_pmap()) {
            world.gop.fence();
        };

        /// Accumulates cost from a function
        template <typename T, typename costT>
        void add_tree(const Function<T,NDIM>& f, const costT& costfn, bool fence=false) {
            const_cast<Function<T,NDIM>&>(f).unaryop_node(add_op<T,costT>(this,costfn), fence);
        }

        /// Accumulates the cost of one node, from any process (non-blocking)

        /// For costs from other sources than a Function. As in a Function, a node that has
        /// children must have all of them added (by some source), possibly with zero cost.
        /// Collective with load_balance: fence before it.
        void add_cost(const keyT& key, double cost, bool has_children) {
            if (tree.is_local(key))
                tree.send(key, &nodeT::add, cost, has_children);
            else
                tree.task(key, &nodeT::add, cost, has_children);
        }

        /// Printing for the curious
        void print_tree(const keyT& key = keyT(0)) {
            Future<iteratorT> futit = tree.find(key);
            iteratorT it = futit.get();
            if (it != tree.end()) {
                for (int i=0; i<key.level(); ++i) std::cout << "  ";
                print(key, it->second.get_total_cost());

                if (it->second.has_children()) {
                    for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                        print_tree(kit.key());
                    }
                }
            }
        }

        /// Per-rank cost bar chart under the given pmap.
        ///
        /// Walks the aggregator tree locally; each entry with total_cost >= 0
        /// contributes its total_cost to bucket pmap->owner(key). Buckets are
        /// summed across ranks and rank 0 prints a horizontal bar chart with
        /// total/avg/max/avg/max/min ratios.
        ///
        /// Call BEFORE load_balance() with the old pmap to see the per-key
        /// distribution that the partitioner is about to act on; call AFTER
        /// with the returned pmap to see how the bin-packed subtrees fell out.
        /// Both invocations sum to the same total work, so the bars are
        /// directly comparable.
        void print_cost_per_rank(const std::string& label,
                                 const std::shared_ptr< WorldDCPmapInterface<keyT> >& pmap,
                                 std::ostream& os = std::cout) {
            const int nproc = world.size();
            std::vector<double> cost(nproc, 0.0);

            const_iteratorT end = tree.end();
            for (const_iteratorT it = tree.begin(); it != end; ++it) {
                double c = it->second.get_total_cost();
                if (c < 0.0) continue;
                cost[pmap->owner(it->first)] += c;
            }
            world.gop.sum(cost.data(), nproc);
            world.gop.fence();
            if (world.rank() != 0) return;

            double total = 0.0;
            double mx = 0.0;
            double mn = std::numeric_limits<double>::max();
            for (double c : cost) {
                total += c;
                if (c > mx) mx = c;
                if (c < mn) mn = c;
            }
            const double avg = (nproc > 0) ? total / nproc : 0.0;
            const int barwidth = 40;

            std::ios::fmtflags saved_flags = os.flags();
            std::streamsize saved_prec = os.precision();

            os << "=== Load balance summary: " << label << " ===\n";
            os << std::fixed << std::setprecision(2)
               << "  total=" << total
               << "  avg=" << avg
               << "  max/avg=" << (avg > 0.0 ? mx / avg : 0.0);
            if (mn > 0.0) os << "  max/min=" << mx / mn;
            os << "\n";
            for (int p = 0; p < nproc; ++p) {
                int n = (mx > 0.0) ? static_cast<int>(std::lround(barwidth * cost[p] / mx)) : 0;
                if (n < 0) n = 0;
                if (n > barwidth) n = barwidth;
                os << "  rank " << std::setw(3) << p << " |"
                   << std::string(n, '#')
                   << std::string(barwidth - n, ' ')
                   << "| " << std::setw(10) << cost[p]
                   << "  (" << (avg > 0.0 ? cost[p] / avg : 0.0) << "x avg)\n";
            }
            os << "================================" << std::endl;

            os.flags(saved_flags);
            os.precision(saved_prec);
        }

        struct CostPerProc {
            double cost;
            int proc;
            CostPerProc() : cost(0.0), proc(0) {}
            CostPerProc(double cost, int proc) : cost(cost), proc(proc) {}
            bool operator<(const CostPerProc& other) const {
                return cost > other.cost;  // Want ascending order
            }
        };

        /// Splits the summed tree into subtrees costing at most about avg

        /// Returns, on process 0, the root of each subtree with its cost: that of the root's
        /// subtree without the subtrees split off below it. Every node of the tree belongs to
        /// the subtree of its nearest listed ancestor (or its own), and the root is listed.
        std::vector< std::pair<keyT,double> > partition_subtrees(double avg) {
            keyT key0(0);
            if (world.rank() == tree.owner(key0)) {
                tree.send(key0, &nodeT::partition, tree, key0, avg*1.1);
            }
            world.gop.fence();

            // Collect entire vector onto node0
            std::vector< std::pair<keyT,double> > results;
            const_iteratorT end = tree.end();
            for (const_iteratorT it=tree.begin(); it!=end; ++it) {
                if (it->second.get_total_cost() >= 0) {
                    results.push_back(std::make_pair(it->first,it->second.get_total_cost()));
                }
            }
            results = world.gop.concat0(results, 128*1024*1024);
            world.gop.fence();
            return results;
        }

        /// Cuts the subtrees, ordered along the space-filling curve, into nproc*ndevice parts

        /// The cut is by cost: first into one contiguous part per process, then each of those
        /// into one per device. A subtree goes to the part its cost midpoint falls into.
        /// Returns the pieces of the curve, (first code, part) ascending from code 0, and the
        /// cost of each part.
        static std::pair<std::vector<std::pair<uint64_t,int>>, std::vector<double>>
        sfc_cut(const std::vector< std::pair<keyT,double> >& subtrees, int nproc, int ndevice) {
            struct Piece { uint64_t begin, end; Level n; double cost; int part; };
            std::vector<Piece> p;
            p.reserve(subtrees.size());
            for (const auto& [key, cost] : subtrees) {
                const uint64_t b = SFCCode<NDIM>::code(key);
                p.push_back({b, b + SFCCode<NDIM>::extent(key), key.level(), cost, 0});
            }
            // along the curve, a subtree before the subtrees split off below it
            std::sort(p.begin(), p.end(), [](const Piece& a, const Piece& b) {
                return a.begin < b.begin || (a.begin == b.begin && a.n < b.n);
            });

            // the slot of [0, n) that the cost midpoint falls into
            auto slot = [](double mid, double width, int n) {
                return (width > 0.0) ? std::min(n-1, int(mid/width)) : 0;
            };
            double total = 0.0;
            for (const auto& x : p) total += x.cost;
            std::vector<int> proc(p.size());
            double prefix = 0.0;
            for (std::size_t i=0; i<p.size(); ++i) {
                proc[i] = slot(prefix + 0.5*p[i].cost, total/nproc, nproc);
                prefix += p[i].cost;
            }
            // the subtrees of a process are contiguous: cut them across its devices
            std::vector<double> part_cost(nproc*ndevice, 0.0);
            for (std::size_t i0=0; i0<p.size(); ) {
                std::size_t i1 = i0;
                double share = 0.0;
                while (i1 < p.size() && proc[i1] == proc[i0]) share += p[i1++].cost;
                double local = 0.0;
                for (std::size_t i=i0; i<i1; ++i) {
                    p[i].part = proc[i]*ndevice + slot(local + 0.5*p[i].cost, share/ndevice, ndevice);
                    local += p[i].cost;
                    part_cost[p[i].part] += p[i].cost;
                }
                i0 = i1;
            }

            // Flatten the nested ranges of the subtrees into pieces: a subtree's range minus the
            // ranges of the subtrees below it reverts to the part of the enclosing subtree.
            std::vector<std::pair<uint64_t,int>> bounds;
            std::vector<std::pair<uint64_t,int>> open;   // (end, part) of the enclosing subtrees
            auto close = [&](uint64_t pos) {
                while (!open.empty() && open.back().first <= pos) {
                    const uint64_t end = open.back().first;
                    open.pop_back();
                    if (!open.empty()) bounds.emplace_back(end, open.back().second);
                }
            };
            for (const auto& x : p) {
                close(x.begin);
                bounds.emplace_back(x.begin, x.part);
                open.emplace_back(x.end, x.part);
            }
            close(std::numeric_limits<uint64_t>::max());

            // of bounds at the same code the last holds; merge neighbors of the same part
            std::vector<std::pair<uint64_t,int>> pieces;
            for (const auto& b : bounds) {
                if (!pieces.empty() && pieces.back().first == b.first) pieces.back().second = b.second;
                else pieces.push_back(b);
            }
            std::vector<std::pair<uint64_t,int>> merged;
            for (const auto& b : pieces) {
                if (merged.empty() || merged.back().second != b.second) merged.push_back(b);
            }
            if (merged.empty()) merged.emplace_back(0, 0);
            merged.front().first = 0;   // the root starts the curve; guard against an absent root
            return {merged, part_cost};
        }

        /// Partitions the tree into contiguous pieces of the space-filling curve, two levels deep

        /// Each process gets about an equal share of the cost, cut further into about equal
        /// shares for its ndevice devices (SFCPmap::device). As in load_balance, the tree is
        /// first split into subtrees of at most total/(nproc*ndevice*fac); fac > 1 makes them
        /// finer, so the cuts can balance better. Collective; the tree is consumed.
        std::shared_ptr< SFCPmap<NDIM> > load_balance_sfc(int ndevice = 1, double fac = 1.0, bool printstuff=false) {
            MADNESS_CHECK_THROW(ndevice > 0, "load_balance_sfc: need at least one device per process");
            world.gop.fence();
            const int nproc = world.size();
            const double avg = sum()/(nproc*ndevice*fac);
            world.gop.fence();
            std::vector< std::pair<keyT,double> > subtrees = partition_subtrees(avg);

            std::pair<std::vector<std::pair<uint64_t,int>>, std::vector<double>> cut;
            if (world.rank() == 0) {
                cut = sfc_cut(subtrees, nproc, ndevice);
                if (printstuff) {
                    print("load_balance_sfc:", subtrees.size(), "subtrees,", cut.first.size(), "pieces");
                    for (int i=0; i<nproc*ndevice; ++i) print("  process", i/ndevice, "device", i%ndevice, "cost", cut.second[i]);
                }
            }
            world.gop.broadcast_serializable(cut, 0);
            world.gop.fence();
            return std::make_shared< SFCPmap<NDIM> >(cut.first, nproc, ndevice, std::move(cut.second));
        }

        /// Actually does the partitioning of the tree
        std::shared_ptr< WorldDCPmapInterface<keyT> > load_balance(double fac = 1.0, bool printstuff=false) {
            world.gop.fence();
            // Compute full tree of costs
            double avg = sum()/(world.size()*fac);
            //if (world.rank() == 0) print_tree();
            world.gop.fence();

            // Create partitioning and collect entire vector onto node0
            std::vector< std::pair<keyT,double> > results = partition_subtrees(avg);

            std::vector< std::pair<keyT,ProcessID> > map;

            if (world.rank() == 0) {

                std::sort(results.begin(), results.end(), compare);
                if (printstuff) {
                    print("THESE ARE THE INITIAL SUBTREES");
                    for (unsigned int i=0; i<results.size(); ++i) print(i,results[i]);
                }

                // Now use bin packing to cram the results together
                map.reserve(results.size());

                // Shove the first nproc entries directly into the queue
                unsigned int nproc = world.size();
                std::priority_queue<CostPerProc> costs;
                for (unsigned int p=0; p<nproc && !results.empty(); ++p) {
                    const std::pair<keyT,double>& f = results.back();
                    costs.push(CostPerProc(f.second,p));
                    map.push_back(std::make_pair(f.first,p));
                    results.pop_back();
                }

                // Process the remainder using the sorting maintained by the priority queue
                while (!results.empty()) {
                    const std::pair<keyT,double>& f = results.back();
                    CostPerProc top = costs.top();
                    costs.pop();
                    top.cost += f.second;
                    costs.push(top);
                    map.push_back(std::make_pair(f.first,top.proc));
                    results.pop_back();
                }
                if (printstuff) {
                    print("THIS IS THE MAP");
                    print(map);
                    print("THESE ARE THE COSTS PER PROCESSOR");
                    while (!costs.empty()) {
                        print(costs.top().proc,costs.top().cost);
                        costs.pop();
                    }
                }
            }

            world.gop.fence();
            world.gop.broadcast_serializable(map, 0);
            world.gop.fence();

            // Return the Procmap

            return std::shared_ptr< WorldDCPmapInterface<keyT> >(new LBDeuxPmap<NDIM>(map));
        }
    };
}


#endif // MADNESS_MRA_IBDEUX_H__INCLUDED

