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
*/

/// \file test_device_convolution.cc
/// \brief Stress test of the MRA-TTG Coulomb convolution of many functions, as in the HF exchange.
///
/// N "orbitals" (sums of random Gaussians) are formed on the host and uploaded. On the device,
/// a graph forms all pair products psi_i psi_j (i <= j) and convolves them with the Poisson
/// operator in one convolution (as device_exchange does in one tile); the potential norms are
/// compared against the host (the same products and operator in MADNESS). Run it with a small
/// PaRSEC device pool to put the graph under memory pressure, e.g.
///
///   PARSEC_MCA_device_cuda_memory_use=4 MRA_TTG_HANG_TIMEOUT=60 ./test_device_convolution n=10
///
/// Options (key=value): n (orbitals, 10), k (8), thresh (1e-6), L (50), lo (1e-10),
/// reps (device repetitions, 2), seed (1), pairs (1: convolve the pair products; 0: convolve the
/// orbitals themselves), store_p (1: also store and check the products, as MRA_TTG_EXCHANGE_CHECK_V),
/// split (1: convolve the stored products in a separate graph), cache (1: keep the device operator across
/// graphs), warm (1: cache, and first convolve one orbital alone), tol (relative tolerance of the norms, 1e-3). Exit code 0 if every repetition matches the host.

#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include <madness/chem/SCF.h>
#include <madness/chem/device/scf_device.h>
#include <madness/chem/device/common.h>
#include <mra/mra.h>

#include <cmath>
#include <cstdlib>
#include <map>
#include <random>
#include <string>

using namespace madness;

namespace {

struct Gaussians {
    std::vector<coord_3d> centers;
    std::vector<double> exponents, coeffs;
    double operator()(const coord_3d& r) const {
        double s = 0.0;
        for (std::size_t g = 0; g < centers.size(); ++g) {
            const double d2 = (r[0] - centers[g][0]) * (r[0] - centers[g][0]) +
                              (r[1] - centers[g][1]) * (r[1] - centers[g][1]) +
                              (r[2] - centers[g][2]) * (r[2] - centers[g][2]);
            s += coeffs[g] * std::exp(-exponents[g] * d2);
        }
        return s;
    }
};

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        const auto eq = a.find('=');
        if (a.rfind("-", 0) == 0 || eq == std::string::npos) continue;
        args[a.substr(0, eq)] = a.substr(eq + 1);
    }
    return args;
}

int run(World& world, const std::map<std::string, std::string>& args) {
    auto get = [&](const char* key, const char* def) {
        const auto it = args.find(key);
        return it == args.end() ? std::string(def) : it->second;
    };
    const long N = std::stol(get("n", "10"));
    const int k = std::stoi(get("k", "8"));
    const double thresh = std::stod(get("thresh", "1e-6"));
    const double L = std::stod(get("L", "50"));
    const double lo = std::stod(get("lo", "1e-10"));
    const int reps = std::stoi(get("reps", "2"));
    const unsigned seed = std::stoul(get("seed", "1"));
    const bool pairs = std::stoi(get("pairs", "1")) != 0;
    const bool store_p = std::stoi(get("store_p", "1")) != 0;
    const double tol = std::stod(get("tol", "1e-3"));
    // split=1: compute and store the products in one graph, convolve the stored products in a second one
    const bool split = std::stoi(get("split", "0")) != 0;
    // cache=1: keep the device operator (and its generated tables) across the graphs;
    // warm=1 (implies cache): first convolve one orbital alone, which generates its tables without memory pressure
    const bool warm = std::stoi(get("warm", "0")) != 0;
    const bool use_cache = warm || std::stoi(get("cache", "0")) != 0;

    FunctionDefaults<3>::set_cubic_cell(-L, L);
    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_truncate_mode(1);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_initial_level(2);

    // the orbitals: 3 Gaussians each, one tight (core-like) and two diffuse, near the origin
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> pos(-2.0, 2.0), coef(-1.0, 1.0);
    const double exps[3] = {30.0, 2.0, 0.4};
    vecfuncT psi;
    for (long i = 0; i < N; ++i) {
        Gaussians g;
        for (int e = 0; e < 3; ++e) {
            g.centers.push_back(coord_3d{pos(gen), pos(gen), pos(gen)});
            g.exponents.push_back(exps[e] * (0.75 + 0.5 * (coef(gen) + 1.0) / 2.0));
            g.coeffs.push_back(coef(gen));
        }
        psi.push_back(FunctionFactory<double, 3>(world).functor([g](const coord_3d& r) { return g(r); }));
    }
    truncate(world, psi);
    normalize(world, psi);
    truncate(world, psi);

    std::vector<std::pair<long, long>> ij;
    if (pairs) {
        for (long i = 0; i < N; ++i)
            for (long j = i; j < N; ++j) ij.emplace_back(i, j);
    }
    const long T = pairs ? long(ij.size()) : N;

    // host reference
    poperatorT op(CoulombOperatorPtr(world, lo, thresh));
    vecfuncT P;
    if (pairs) {
        for (const auto& [i, j] : ij) P.push_back(psi[i] * psi[j]);
        truncate(world, P, thresh);
    } else {
        P = copy(world, psi);
    }
    vecfuncT V = apply(world, *op, P);
    truncate(world, V, thresh);
    const std::vector<double> vref = norm2s(world, V);
    const std::vector<double> pref = norm2s(world, P);
    if (world.rank() == 0)
        print("device convolution test: N", N, "functions", T, "k", k, "thresh", thresh, "L", L,
              pairs ? "(pair products)" : "(orbitals)");

    Tensor<double> Ua(N, T), Ub(N, T);
    for (long p = 0; p < T; ++p) {
        Ua(pairs ? ij[p].first : p, p) = 1.0;
        Ub(pairs ? ij[p].second : p, p) = 1.0;
    }

    compress(world, psi);
    DeviceFunctions psi_d = device_upload(world, psi);
    DeviceOperatorCache cache;
    DeviceOperatorCache* const cachep = use_cache ? &cache : nullptr;
    if (warm) {
        Tensor<double> U1(N, 1L);
        U1(0, 0) = 1.0;
        DeviceFunctions W;
        DeviceGraph g(world);
        g.store(g.convolution(op, g.transform(g.feed(psi_d, true), U1, 0.0, "t-warm"), thresh, cachep), W);
        g.execute();
    }
    int failures = 0;
    for (int rep = 0; rep < reps; ++rep) {
        DeviceFunctions Pd, Vd;
        const double t0 = wall_time();
        {
            DeviceGraph g(world);
            auto f = g.feed(psi_d, true);
            auto in = pairs ? g.multiply(g.transform(f, Ua, 0.0, "t-A"), g.transform(f, Ub, 0.0, "t-B"), thresh, "t-pair")
                            : g.transform(f, Ua, 0.0, "t-copy");
            if (store_p || split) g.store(in, Pd);
            if (!split) g.store(g.convolution(op, in, thresh, cachep), Vd);
            g.execute();
        }
        if (split) {
            DeviceGraph g(world);
            g.store(g.convolution(op, g.feed(Pd, true), thresh, cachep), Vd);
            g.execute();
        }
        const double t1 = wall_time();
        if (store_p || split) {
            // the products themselves, against the host's (both truncated with the same threshold)
            const std::vector<double> pd = device_norms(world, Pd);
            double pmax = 0.0;
            long pbad = 0;
            for (long p = 0; p < T; ++p) {
                const double rel = std::abs(pd[p] - pref[p]) / std::max(pref[p], 1e-12);
                if (!std::isfinite(pd[p]) || rel > tol) ++pbad;
                if (std::isfinite(rel)) pmax = std::max(pmax, rel);
            }
            if (world.rank() == 0)
                print("repetition", rep, ": products max relative norm error", pmax, ", wrong", pbad, "of", T);
        }
        const std::vector<double> vd = device_norms(world, Vd);
        double maxrel = 0.0;
        long bad = 0, nan = 0;
        for (long p = 0; p < T; ++p) {
            if (!std::isfinite(vd[p])) { ++nan; ++bad; continue; }
            const double rel = std::abs(vd[p] - vref[p]) / std::max(vref[p], 1e-12);
            maxrel = std::max(maxrel, rel);
            if (rel > tol) ++bad;
        }
        if (world.rank() == 0) {
            print("repetition", rep, ": time", t1 - t0, "s, max relative norm error", maxrel, ", wrong", bad, "of", T,
                  ", non-finite", nan, bad ? "  FAIL" : "  ok");
            if (bad) {
                std::cout << "  (host, device):";
                for (long p = 0; p < T; ++p) {
                    const double rel = std::abs(vd[p] - vref[p]) / std::max(vref[p], 1e-12);
                    if (!std::isfinite(vd[p]) || rel > tol) std::cout << " " << p << ":" << vref[p] << "," << vd[p];
                }
                std::cout << std::endl;
            }
        }
        if (bad) ++failures;
    }
    if (world.rank() == 0) print(failures ? "device convolution test FAILED" : "device convolution test passed");
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    mra::initialize(argc, argv, -1);
    if (const char* bs = std::getenv("MRA_TTG_BATCH_SIZE")) mra::set_batch_size(std::stoul(bs));
    int rc = 0;
    {
        World& world = World::get_default();
        startup(world, argc, argv, true);
        rc = run(world, parse_args(argc, argv));
        world.gop.fence();
    }
    mra::finalize();
    return rc;
}
