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
/// batches), the update psi_i - c_i r_i, and a linear combination of sets. Operations that
/// combine batches (matrix_inner) must refuse several.
///
/// Options (key=value): n (orbitals, 5), k (6), thresh (1e-5), L (20), seed (1), tol (relative
/// tolerance, 1e-10). Exit code 0 if every check passes.

#include <madness/mra/mra.h>
#include <madness/chem/SCF.h>
#include <madness/chem/device/scf_device.h>
#include <madness/chem/device/common.h>
#include <mra/mra.h>

#include <cmath>
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

    for (int nb : {1, 2, 3}) {
        if (nb > N) continue;
        set_device_orbital_batches(nb);
        const std::string tag = std::to_string(nb) + " batches: ";
        const DeviceLayout layout = device_orbital_layout(N);

        // upload, download, and where the nodes are
        DeviceFunctions psi_d = device_upload(world, psi);
        check(world, rel_diff(world, psi, psi_d, like) < tol, tag + "upload and download", rel_diff(world, psi, psi_d, like), tol);
        long misplaced = 0;
        const mra::BatchGroups groups = device_batch_groups();
        fv(psi_d).for_each_local_node([&](const auto& key, const auto&) {
            misplaced += !groups.is_member(key.batch(), world.rank());
        });
        world.gop.sum(misplaced);
        check(world, misplaced == 0, tag + "nodes on their batch's group", misplaced, 0);

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

        // matrices of inner products combine batches: not supported yet
        if (nb > 1) {
            bool refused = false;
            try {
                DeviceGraph g(world);
                auto f = g.feed(psi_d, true);
                g.matrix_inner(f, f);
            } catch (const std::exception&) {
                refused = true;
            }
            check(world, refused, tag + "matrix_inner refuses several batches", refused ? 0 : 1, 0);
        }
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
