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

/// \file test_device_batches.cc
/// \brief The device operations of the SCF on orbitals split into batches, against MADNESS.
///
/// The orbitals (sums of random Gaussians) are uploaded in 1, 2 and 3 batches
/// (set_device_orbital_batches), each held and processed by its own group of processes, and the
/// operations that stay within the batches are compared with MADNESS: upload and download, norms
/// and <psi_i|psi_i>, a potential applied to all orbitals (replicated into every batch), a
/// transform block-diagonal in the batches, normalization (a decision from the norms of all
/// batches), the update psi_i - c_i r_i, a linear combination of sets, and the density (partial
/// densities of the batches summed into one function on all processes) with its Coulomb
/// potential, matrices of inner products across the batches (<psi_i|V psi_j>, <psi_i|psi_j>, the
/// kinetic matrix against one batch), a decision from such a matrix, and transforms that couple
/// the batches (by a fixed U and by one decided in the graph). Concatenation must refuse several.
///
/// Options (key=value): n (orbitals, 5), k (6), thresh (1e-5), L (20), seed (1), tol (relative
/// tolerance, 1e-10), ctol (of the Coulomb potential, 1e-6). Exit code 0 if every check passes.

#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include <madness/chem/SCF.h>
#include <madness/chem/device/scf_device.h>
#include <madness/chem/device/common.h>
#include <mra/mra.h>

#include <cmath>
#include <map>
#include <random>
#include <set>
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

int failures = 0;

void check(World& world, bool ok, const std::string& what, double err, double tol) {
    if (world.rank() == 0) print(ok ? "PASSED" : "FAILED", what, ": error", err, "tolerance", tol);
    if (!ok) ++failures;
}

/// max_i ||ref_i - d_i|| / max_i ||ref_i||, d downloaded (d is kept)
double rel_diff(World& world, const vecfuncT& ref, DeviceFunctions& d, const functionT& like) {
    vecfuncT dv = device_download(world, d, like);
    const std::vector<double> diff = norm2s(world, sub(world, ref, dv));
    const std::vector<double> n = norm2s(world, ref);
    return *std::max_element(diff.begin(), diff.end()) / std::max(*std::max_element(n.begin(), n.end()), 1e-300);
}

int run(World& world, const std::map<std::string, std::string>& args) {
    auto get = [&](const char* key, const char* def) {
        const auto it = args.find(key);
        return it == args.end() ? std::string(def) : it->second;
    };
    const long N = std::stol(get("n", "5"));
    const int k = std::stoi(get("k", "6"));
    const double thresh = std::stod(get("thresh", "1e-5"));
    const double L = std::stod(get("L", "20"));
    const unsigned seed = std::stoul(get("seed", "1"));
    const double tol = std::stod(get("tol", "1e-10"));
    // the Coulomb potential: two convolutions that screen at the threshold
    const double ctol = std::stod(get("ctol", "1e-6"));

    FunctionDefaults<3>::set_cubic_cell(-L, L);
    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_truncate_mode(1);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_initial_level(2);

    // the orbitals: 3 Gaussians each near the origin; a potential of 2 Gaussians
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> pos(-2.0, 2.0), coef(-1.0, 1.0);
    auto random_gaussians = [&](int n, const double* exps) {
        Gaussians g;
        for (int e = 0; e < n; ++e) {
            g.centers.push_back(coord_3d{pos(gen), pos(gen), pos(gen)});
            g.exponents.push_back(exps[e] * (0.75 + 0.25 * (coef(gen) + 1.0)));
            g.coeffs.push_back(coef(gen));
        }
        return FunctionFactory<double, 3>(world).functor([g](const coord_3d& r) { return g(r); });
    };
    const double exps[3] = {10.0, 2.0, 0.4};
    vecfuncT psi;
    for (long i = 0; i < N; ++i) psi.push_back(random_gaussians(3, exps));
    truncate(world, psi);
    normalize(world, psi);
    truncate(world, psi);
    functionT V = random_gaussians(2, exps + 1);
    compress(world, psi);
    V.compress();
    const functionT like = psi.front();
    poperatorT coulop(CoulombOperatorPtr(world, 1e-6, thresh));

    Tensor<double> kin_one;   // the kinetic matrix with one batch
    vecfuncT rneg_one;        // the BSH residual with one batch, and its norms
    vecfuncT K_one;           // the exchange with one batch
    std::vector<double> rnorm_one;
    for (int nb : {1, 2, 3}) {
        if (nb > N) continue;
        set_device_orbital_batches(nb);
        compress(world, psi);   // the references of the previous batches change its form
        const std::string tag = std::to_string(nb) + " batches: ";
        const DeviceLayout layout = device_orbital_layout(N);

        // upload, download, and where the nodes are
        DeviceFunctions psi_d = device_upload(world, psi);
        check(world, rel_diff(world, psi, psi_d, like) < tol, tag + "upload and download", rel_diff(world, psi, psi_d, like), tol);
        long misplaced = 0;
        const mra::BatchGroups groups = layout->groups();
        fv(psi_d).for_each_local_node([&](const auto& key, const auto&) {
            misplaced += !groups.is_member(key.batch(), world.rank());
        });
        world.gop.sum(misplaced);
        check(world, misplaced == 0, tag + "nodes on their batch's group", misplaced, 0);

        // the devices of the nodes (4 per process): batches sharing a process each in their own
        // share of its devices, a batch with a group of its own on all of them
        {
            constexpr int ndev = 4;
            const DeviceDevMap devmap(groups, ndev);
            long off_share = 0, shared = 0, spread = 0;
            std::vector<std::set<int>> used(nb);
            fv(psi_d).for_each_local_node([&](const auto& key, const auto&) {
                const auto [d0, d1] = groups.batch_devices(key.batch(), ndev);
                const int d = devmap(key);
                off_share += (d < d0 || d >= d1);
                used[key.batch()].insert(d);
            });
            for (mra::Batch a = 0; a < nb; ++a) {
                const auto [sa, na] = groups.batch_slot(a);
                // with no more batches on the process than devices, a batch's share is its own
                for (mra::Batch b = a + 1; b < nb && na <= ndev; ++b)
                    if (groups.batch_pes(a) == groups.batch_pes(b))
                        for (int d : used[a]) shared += used[b].count(d);
                // a batch alone on its processes uses all their devices
                if (na == 1 && !used[a].empty() && int(used[a].size()) < ndev) ++spread;
            }
            world.gop.sum(off_share);
            world.gop.sum(shared);
            world.gop.sum(spread);
            check(world, off_share == 0, tag + "nodes on their batch's devices", off_share, 0);
            check(world, shared == 0, tag + "batches on one process on different devices", shared, 0);
            check(world, spread == 0, tag + "a batch alone on its processes on all devices", spread, 0);
        }

        // norms and <psi_i|psi_i>
        {
            const std::vector<double> ref = norm2s(world, psi), d = device_norms(world, psi_d);
            double err = 0.0;
            for (long i = 0; i < N; ++i) err = std::max(err, std::abs(ref[i] - d[i]));
            check(world, err < tol, tag + "norms", err, tol);
            DeviceGraph g(world);
            auto f = g.feed(psi_d, true);
            auto r = g.inner(f, f);
            g.execute();
            const Tensor<double> dd = g.get(r), rr = inner(world, psi, psi);
            check(world, (dd - rr).absmax() < tol, tag + "<psi_i|psi_i>", (dd - rr).absmax(), tol);
        }

        // V psi_i, V replicated into every batch
        DeviceFunctions Vpsi_d;
        device_apply_local_potential(world, V, psi_d, Vpsi_d, 0.0);
        vecfuncT Vpsi = mul(world, V, psi);
        compress(world, Vpsi);
        V.compress();
        check(world, rel_diff(world, Vpsi, Vpsi_d, like) < tol, tag + "V psi_i, V replicated",
              rel_diff(world, Vpsi, Vpsi_d, like), tol);

        // the BSH residual: one operator per orbital (its own eps), each batch applying those of
        // its orbitals; against one batch
        {
            tensorT eps(N), fockp(N, N);
            for (long i = 0; i < N; ++i) {
                eps(i) = -0.4 - 0.3 * i;
                fockp(i, i) = 0.05 * i;
            }
            CalculationParameters param;
            DeviceFunctions Vpsi_u = device_upload(world, Vpsi);
            std::vector<double> rnorm;
            DeviceFunctions r_d = device_bsh_residual(world, psi_d, Vpsi_u, fockp, eps, param, rnorm);
            if (nb == 1) {
                rneg_one = device_download(world, r_d, like);
                rnorm_one = rnorm;
            }
            double dn = 0.0;
            for (long i = 0; i < N; ++i) dn = std::max(dn, std::abs(rnorm[i] - rnorm_one[i]));
            check(world, rel_diff(world, rneg_one, r_d, like) < tol, tag + "BSH residual, against one batch",
                  rel_diff(world, rneg_one, r_d, like), tol);
            check(world, dn < tol, tag + "BSH residual norms, against one batch", dn, tol);
        }

        // the exchange K psi_i = sum_j psi_j G(psi_j psi_i): blocks of two batches each on one
        // group, in one graph and in tiles of one round; against one batch (and its tiles)
        {
            DeviceOperatorCache cache;
            DeviceFunctions K_d = device_exchange(world, psi_d, 1e-4, cache, 0);
            if (nb == 1) K_one = device_download(world, K_d, like);
            check(world, rel_diff(world, K_one, K_d, like) < tol, tag + "exchange, against one batch",
                  rel_diff(world, K_one, K_d, like), tol);
            // tiles of one pair with one batch, of parts of rounds with several (one pair of
            // each block), and of two and four pairs (rounds cut, parts joined)
            for (std::size_t tile : {1, 2, 4}) {
                DeviceFunctions K_t = device_exchange(world, psi_d, 1e-4, cache, tile);
                check(world, rel_diff(world, K_one, K_t, like) < tol,
                      tag + "exchange in tiles of " + std::to_string(tile) + ", against one batch",
                      rel_diff(world, K_one, K_t, like), tol);
            }
        }

        // a transform block-diagonal in the batches
        {
            Tensor<double> U(N, N);
            for (mra::Batch b = 0; b < layout->num_batches(); ++b) {
                const long o = layout->offset(b), n = layout->batch_size(b);
                for (long i = 0; i < n; ++i)
                    for (long j = 0; j < n; ++j) U(o + i, o + j) = (i == j ? 1.0 : 0.0) + 0.2 * coef(gen);
            }
            DeviceFunctions t_d;
            {
                DeviceGraph g(world);
                g.store(g.transform(g.feed(psi_d, true), U, 0.0, "block-transform"), t_d);
                g.execute();
            }
            check(world, rel_diff(world, transform(world, psi, U), t_d, like) < tol, tag + "transform block-diagonal",
                  rel_diff(world, transform(world, psi, U), t_d, like), tol);
        }

        // normalization: the scaling decided from the norms of all batches
        {
            vecfuncT ref = copy(world, Vpsi);
            normalize(world, ref);
            DeviceFunctions w_d = device_upload(world, Vpsi);
            device_normalize(world, w_d);
            check(world, rel_diff(world, ref, w_d, like) < tol, tag + "normalize", rel_diff(world, ref, w_d, like), tol);
        }

        // psi_i - c_i r_i, r = V psi
        {
            std::vector<double> c(N);
            for (long i = 0; i < N; ++i) c[i] = 0.1 * double(i + 1);
            vecfuncT ref = copy(world, psi);
            for (long i = 0; i < N; ++i) ref[i].gaxpy(1.0, Vpsi[i], -c[i], false);
            world.gop.fence();
            DeviceFunctions a_d = device_upload(world, psi), r_d = device_upload(world, Vpsi);
            device_update(world, a_d, r_d, c);
            check(world, rel_diff(world, ref, a_d, like) < tol, tag + "psi_i - c_i r_i", rel_diff(world, ref, a_d, like), tol);
        }

        // a linear combination of sets: 0.5 psi - 1.5 V psi
        {
            vecfuncT ref = copy(world, psi);
            gaxpy(world, 0.5, ref, -1.5, Vpsi);
            DeviceFunctions c_d;
            DeviceGraph g(world);
            g.store(g.combine({0.5, -1.5}, {g.feed(psi_d, true), g.feed(Vpsi_d, true)}), c_d);
            g.execute();
            check(world, rel_diff(world, ref, c_d, like) < tol, tag + "combine", rel_diff(world, ref, c_d, like), tol);
        }

        // the density sum_i occ_i psi_i^2 (partial densities of the batches, summed into one function
        // on all processes) and its Coulomb potential, in one graph
        {
            tensorT occ(N);
            for (long i = 0; i < N; ++i) occ[i] = 2.0 - 0.25 * double(i);
            vecfuncT sq = mul(world, psi, psi);
            functionT rho_ref = FunctionFactory<double, 3>(world);
            rho_ref.compress();
            for (long i = 0; i < N; ++i) {
                sq[i].compress();
                rho_ref.gaxpy(1.0, sq[i], occ[i], false);
            }
            world.gop.fence();
            functionT v_ref = apply(*coulop, rho_ref);
            functionT rho, v;
            {
                DeviceGraph g(world);
                auto r = g.density(g.feed(psi_d, true), occ);
                g.store(r, rho, like);
                g.store(g.convolution(coulop, r, 0.0), v, like);
                g.execute();
            }
            const double drho = (rho_ref - rho).norm2() / rho_ref.norm2();
            const double dv = (v_ref - v).norm2() / v_ref.norm2();
            check(world, drho < tol, tag + "density", drho, tol);
            check(world, dv < ctol, tag + "Coulomb potential of the density", dv, ctol);
        }

        // matrices of inner products across the batches: <psi_i|V psi_j>, and <psi_i|psi_j> (one set)
        {
            Tensor<double> pv, pp;
            {
                DeviceGraph g(world);
                auto f = g.feed(psi_d, true);
                auto rpv = g.matrix_inner(f, g.feed(Vpsi_d, true));
                auto rpp = g.matrix_inner(f, f);
                g.execute();
                pv = g.get(rpv);
                pp = g.get(rpp);
            }
            const Tensor<double> pv_ref = matrix_inner(world, psi, Vpsi), pp_ref = matrix_inner(world, psi, psi);
            check(world, (pv - pv_ref).absmax() < tol * pv_ref.absmax(), tag + "<psi_i|V psi_j>",
                  (pv - pv_ref).absmax(), tol * pv_ref.absmax());
            check(world, (pp - pp_ref).absmax() < tol, tag + "<psi_i|psi_j>", (pp - pp_ref).absmax(), tol);
        }

        // the kinetic matrix (derivatives per batch, the matrix across them), against one batch
        {
            const Tensor<double> kin = device_kinetic_matrix(world, psi_d);
            if (nb == 1) kin_one = copy(kin);
            const double dk = (kin - kin_one).absmax();
            check(world, dk < tol * kin_one.absmax(), tag + "kinetic matrix, against one batch", dk, tol * kin_one.absmax());
        }

        // a decision from a matrix of all batches (the overlap): psi_i / sqrt(S_ii)
        {
            vecfuncT ref = copy(world, psi);
            normalize(world, ref);
            DeviceFunctions n_d;
            {
                DeviceGraph g(world);
                auto f = g.feed(psi_d, true);
                auto D = g.decide(g.matrix_inner(f, f), [](const Tensor<double>& S) {
                    Tensor<double> d(S.dim(0), S.dim(0));
                    for (long i = 0; i < S.dim(0); ++i) d(i, i) = 1.0 / std::sqrt(S(i, i));
                    return d;
                });
                g.store(g.transform(f, D, N, 0.0), n_d);
                g.execute();
            }
            check(world, rel_diff(world, ref, n_d, like) < tol, tag + "decision from the overlap of all batches",
                  rel_diff(world, ref, n_d, like), tol);
        }

        // transforms that couple the batches: a fixed U, and U decided in the graph (the overlap S)
        {
            Tensor<double> U(N, N);
            for (long i = 0; i < N; ++i)
                for (long j = 0; j < N; ++j) U(i, j) = (i == j ? 1.0 : 0.0) + 0.3 * coef(gen);
            DeviceFunctions t_d, s_d;
            {
                DeviceGraph g(world);
                auto f = g.feed(psi_d, true);
                g.store(g.transform(f, U, 0.0, "dense"), t_d);
                auto S = g.decide(g.matrix_inner(f, f), [](const Tensor<double>& S) { return copy(S); });
                g.store(g.transform(f, S, N, 0.0), s_d);
                g.execute();
            }
            check(world, rel_diff(world, transform(world, psi, U), t_d, like) < tol, tag + "transform coupling the batches",
                  rel_diff(world, transform(world, psi, U), t_d, like), tol);
            const Tensor<double> S = matrix_inner(world, psi, psi);
            check(world, rel_diff(world, transform(world, psi, S), s_d, like) < tol,
                  tag + "transform by a U decided from all batches", rel_diff(world, transform(world, psi, S), s_d, like), tol);
        }

        // concatenation combines batches: not supported yet
        if (nb > 1) {
            bool refused = false;
            try {
                DeviceGraph g(world);
                auto f = g.feed(psi_d, true);
                g.concat(f, g.feed(Vpsi_d, true));
            } catch (const std::exception& e) {
                // and says why (the message must outlive the throw)
                refused = std::string(e.what()).find("several batches") != std::string::npos;
            }
            check(world, refused, tag + "concat refuses several batches, and says why", refused ? 0 : 1, 0);
            // a rebatch into batches of another size: refused when the graph is built
            refused = false;
            try {
                DeviceGraph g(world);
                g.rebatch(g.feed(psi_d, true), one_batch(N + 1), [](int) { return std::vector<int>{0}; });
            } catch (const std::exception& e) {
                refused = std::string(e.what()).find("another size") != std::string::npos;
            }
            check(world, refused, tag + "rebatch refuses a target of another size", refused ? 0 : 1, 0);
        }
    }
    // more batches than orbitals: one batch per orbital
    {
        set_device_orbital_batches(mra::Batch(N + 2));
        const DeviceLayout layout = device_orbital_layout(N);
        check(world, layout->num_batches() == N, "more batches than orbitals: one per orbital",
              double(layout->num_batches()) - double(N), 0);
        compress(world, psi);
        DeviceFunctions psi_d = device_upload(world, psi);
        check(world, rel_diff(world, psi, psi_d, like) < tol, "more batches than orbitals: upload and download",
              rel_diff(world, psi, psi_d, like), tol);
    }
    set_device_orbital_batches(1);
    if (world.rank() == 0) print(failures ? "device batches test FAILED" : "device batches test passed");
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    mra::initialize(argc, argv, -1);
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
