// ============================================================================
// Crank-Nicolson finite-difference pricer for a down-and-out barrier option
// under a *local* volatility model (sigma = sigma(S, t), not a single flat
// number). Exposed to Python via pybind11.
//
// ----------------------------------------------------------------------------
// THE PDE
// ----------------------------------------------------------------------------
// Under local volatility, the option value V(S, t) satisfies the generalized
// Black-Scholes PDE:
//
//     dV/dt + 0.5 * sigma(S,t)^2 * S^2 * d2V/dS2 + (r-q) * S * dV/dS - r*V = 0
//
// This is solved BACKWARD in time: we know the payoff at maturity t=T, and
// step backward to t=0 to get today's price. Time in the code always
// indexes forward (n=0 is t=0, n=N is t=T) -- "backward in time" just means
// we fill in the array from index N down to index 0.
//
// ----------------------------------------------------------------------------
// WHY CRANK-NICOLSON (AND NOT FULLY EXPLICIT OR FULLY IMPLICIT)
// ----------------------------------------------------------------------------
// Write the spatial part of the PDE as an operator L, so the PDE reads
// dV/dt = -L(V). Two easy discretizations:
//   - Fully EXPLICIT (use L at the known, later time level): trivial to
//     solve (no linear system) but only STABLE if dt is small enough
//     relative to dS^2 (a CFL-type condition) -- for a fine S-grid that
//     forces painfully many time steps.
//   - Fully IMPLICIT (use L at the unknown, earlier time level): stable
//     for any dt, but only first-order accurate in time (errors shrink
//     linearly as dt shrinks).
// Crank-Nicolson averages the two: 0.5*L(known) + 0.5*L(unknown). It's
// unconditionally stable (like implicit) AND second-order accurate in time
// (like a centered difference), at the cost of solving a linear system
// every step -- which is cheap here because that system is TRIDIAGONAL
// (each grid point only talks to its two neighbours), solvable in O(N) via
// the Thomas algorithm instead of a general O(N^3) solve.
//
// ----------------------------------------------------------------------------
// DERIVING THE TRIDIAGONAL SYSTEM
// ----------------------------------------------------------------------------
// On a uniform space grid S_0..S_{N-1} with spacing dS, central-difference
// the spatial derivatives at interior node i:
//
//     dV/dS   ~= (V_{i+1} - V_{i-1}) / (2*dS)
//     d2V/dS2 ~= (V_{i+1} - 2*V_i + V_{i-1}) / dS^2
//
// so L_i(V) = a_i*V_{i-1} + b_i*V_i + c_i*V_{i+1}, with (sigma_i is the
// local vol at (S_i, current time level)):
//
//     a_i =  0.5*sigma_i^2*S_i^2/dS^2 - (r-q)*S_i/(2*dS)
//     b_i = -sigma_i^2*S_i^2/dS^2 - r
//     c_i =  0.5*sigma_i^2*S_i^2/dS^2 + (r-q)*S_i/(2*dS)
//
// Crank-Nicolson between time level n+1 (known, later/closer to maturity)
// and n (unknown, being solved for) with step dt:
//
//     (V^n_i - V^{n+1}_i)/dt = 0.5*[L_i(V^{n+1}) + L_i(V^n)]
//
// Collect knowns on the right, unknowns on the left:
//
//     -0.5*dt*a_i * V^n_{i-1}  +  (1 - 0.5*dt*b_i) * V^n_i  -  0.5*dt*c_i * V^n_{i+1}
//         =  0.5*dt*a_i * V^{n+1}_{i-1}  +  (1 + 0.5*dt*b_i) * V^{n+1}_i  +  0.5*dt*c_i * V^{n+1}_{i+1}
//
// That's one row of a tridiagonal system A * V^n = rhs, for every interior
// i. sigma_i is allowed to depend on the time level -- we use the average
// of the local vol at levels n and n+1 for that step's a_i/b_i/c_i, which
// keeps a single set of coefficients for both sides of the equation above
// (a small, standard simplification -- "frozen coefficients" over one
// Crank-Nicolson step -- rather than re-deriving separate operators for
// each time level).
//
// ----------------------------------------------------------------------------
// BOUNDARY CONDITIONS
// ----------------------------------------------------------------------------
// Down-and-out barrier: the option is worthless the instant S touches the
// barrier B. The cheapest way to enforce that in a PDE grid is to make the
// barrier the grid's LOWER edge (S_0 = B) and pin V = 0 there at every time
// step -- so the domain itself is [B, S_max], not [0, S_max]. That's exactly
// the projection real desks use for a single continuously-monitored
// barrier; no need to hunt for the barrier's position mid-grid.
//
// Upper edge S_max: use the standard deep-ITM linear boundary condition for
// a call, V(S_max, t) ~= S_max*exp(-q*tau) - K*exp(-r*tau) where
// tau = T - t is time remaining -- i.e. assume S_max is far enough into the
// money that the option is certain to be exercised and just behaves like
// the forward contract value. S_max should be picked comfortably away from
// where the price actually matters (several sigma*sqrt(T) above spot) so
// this approximation doesn't contaminate the answer near S_0.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

// Thomas algorithm: O(N) solve of a tridiagonal system.
//   sub[i]  = subdiagonal entry (multiplies x[i-1]), sub[0] unused
//   diag[i] = diagonal entry
//   sup[i]  = superdiagonal entry (multiplies x[i+1]), sup[N-1] unused
//   rhs[i]  = right-hand side
// Modifies sup/rhs in place (standard in-place forward-sweep/back-substitution)
// and returns the solution.
static std::vector<double> thomas_solve(
    std::vector<double> sub, std::vector<double> diag,
    std::vector<double> sup, std::vector<double> rhs) {
    const int n = static_cast<int>(diag.size());

    // forward sweep: eliminate the subdiagonal
    for (int i = 1; i < n; ++i) {
        double w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }

    // back substitution
    std::vector<double> x(n);
    x[n - 1] = rhs[n - 1] / diag[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = (rhs[i] - sup[i] * x[i + 1]) / diag[i];
    }
    return x;
}

// Prices a down-and-out barrier call (or put) under local volatility.
//
// S_grid:     spatial grid, length N_S, UNIFORM spacing, S_grid[0] == barrier
// local_vol:  (N_t+1, N_S) array; local_vol[n][i] = sigma_local(S_grid[i], t_n)
//             where t_n = n * T / N_t runs from 0 (today) to T (maturity)
// r, q:       risk-free rate, dividend yield (continuously compounded)
// K:          strike
// T:          time to maturity in years
// is_call:    true for call, false for put payoff at maturity
//
// Returns the price vector V(S_grid[i], t=0) -- i.e. today's price as a
// function of spot, so the Python side can interpolate to a specific S0.
py::array_t<double> price_barrier_cn(
    py::array_t<double, py::array::c_style | py::array::forcecast> S_grid_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_vol_arr,
    double r, double q, double K, double T, bool is_call) {

    auto S_buf = S_grid_arr.request();
    auto lv_buf = local_vol_arr.request();

    if (S_buf.ndim != 1) throw std::runtime_error("S_grid must be 1-D");
    if (lv_buf.ndim != 2) throw std::runtime_error("local_vol must be 2-D (N_t+1, N_S)");

    const int N_S = static_cast<int>(S_buf.shape[0]);
    const int N_t = static_cast<int>(lv_buf.shape[0]) - 1;  // rows = N_t+1 time points
    if (N_S < 3 || N_t < 1 || !std::isfinite(T) || T <= 0 ||
        !std::isfinite(K) || K <= 0 || !std::isfinite(r) || !std::isfinite(q)) {
        throw std::runtime_error("Need valid prices/maturity and at least 3 space nodes, 2 time rows");
    }
    if (static_cast<int>(lv_buf.shape[1]) != N_S) {
        throw std::runtime_error("local_vol second dimension must match len(S_grid)");
    }

    const double* S = static_cast<double*>(S_buf.ptr);
    const double* LV = static_cast<double*>(lv_buf.ptr);  // row-major: LV[n*N_S + i]

    const double dS = S[1] - S[0];
    if (!std::isfinite(dS) || dS <= 0) throw std::runtime_error("Grid must increase");
    for (int i = 0; i < N_S; ++i) {
        if (!std::isfinite(S[i]) || S[i] <= 0 ||
            std::abs(S[i] - (S[0] + i*dS)) > 1e-8*std::max(1.0, S[i]))
            throw std::runtime_error("Grid must be positive, finite, and uniform");
    }
    for (int i = 0; i < (N_t+1)*N_S; ++i) {
        if (!std::isfinite(LV[i]) || LV[i] < 0)
            throw std::runtime_error("Local volatility must be finite and nonnegative");
    }
    const double dt = T / N_t;

    // --- terminal condition: payoff at t = T (time index n = N_t) ---
    std::vector<double> V(N_S);
    for (int i = 0; i < N_S; ++i) {
        V[i] = is_call ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }
    // the barrier is the grid's lower edge, so V[0] should already be ~0 for
    // a call struck above the barrier; enforce it explicitly for safety /
    // for puts where the payoff at S=barrier need not be zero on its own.
    V[0] = 0.0;

    // --- march backward from n = N_t (t=T) down to n = 0 (t=0) ---
    for (int n = N_t - 1; n >= 0; --n) {
        const double tau = T - n * dt;  // time remaining at the level we're solving FOR

        std::vector<double> sub(N_S), diag(N_S), sup(N_S), rhs(N_S);

        // row 0: barrier boundary, V = 0 always
        sub[0] = 0.0; diag[0] = 1.0; sup[0] = 0.0; rhs[0] = 0.0;

        // interior rows
        for (int i = 1; i < N_S - 1; ++i) {
            // local vol at this step: average of the two time levels being
            // bridged (n and n+1), so both sides of the CN average share
            // one operator -- see the derivation above.
            double sigma = 0.5 * (LV[n * N_S + i] + LV[(n + 1) * N_S + i]);
            double sig2S2 = sigma * sigma * S[i] * S[i];

            double a_i = 0.5 * sig2S2 / (dS * dS) - (r - q) * S[i] / (2 * dS);
            double b_i = -sig2S2 / (dS * dS) - r;
            double c_i = 0.5 * sig2S2 / (dS * dS) + (r - q) * S[i] / (2 * dS);

            sub[i]  = -0.5 * dt * a_i;
            diag[i] =  1.0 - 0.5 * dt * b_i;
            sup[i]  = -0.5 * dt * c_i;

            rhs[i] = 0.5 * dt * a_i * V[i - 1]
                   + (1.0 + 0.5 * dt * b_i) * V[i]
                   + 0.5 * dt * c_i * V[i + 1];
        }

        // row N_S-1: deep-ITM linear boundary (call) / deep-OTM->0 (put)
        sub[N_S - 1] = 0.0; diag[N_S - 1] = 1.0; sup[N_S - 1] = 0.0;
        if (is_call) {
            rhs[N_S - 1] = S[N_S - 1] * std::exp(-q * tau) - K * std::exp(-r * tau);
        } else {
            rhs[N_S - 1] = 0.0;  // far OTM put is worthless
        }

        V = thomas_solve(sub, diag, sup, rhs);
        V[0] = 0.0;  // re-assert the barrier exactly (row 0 already enforces this, belt-and-braces)
    }

    // package the t=0 price curve for return to Python
    py::array_t<double> result(N_S);
    auto res_buf = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);
    std::copy(V.begin(), V.end(), res_ptr);
    return result;
}

PYBIND11_MODULE(cn_pricer, m) {
    m.doc() = "Crank-Nicolson local-volatility barrier option pricer";
    m.def("price_barrier_cn", &price_barrier_cn,
          py::arg("S_grid"), py::arg("local_vol"),
          py::arg("r"), py::arg("q"), py::arg("K"), py::arg("T"), py::arg("is_call"),
          "Price a down-and-out barrier option; barrier = S_grid[0]. "
          "Returns the price curve V(S, t=0) over S_grid.");
}
