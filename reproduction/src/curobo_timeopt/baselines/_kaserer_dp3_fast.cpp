#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace {

void require_1d_double_cpu(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(value.scalar_type() == torch::kFloat64, name, " must be float64");
    TORCH_CHECK(value.dim() == 1, name, " must be 1-D");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

void require_1d_int64_cpu(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(value.scalar_type() == torch::kInt64, name, " must be int64");
    TORCH_CHECK(value.dim() == 1, name, " must be 1-D");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

void require_2d_double_cpu(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(value.scalar_type() == torch::kFloat64, name, " must be float64");
    TORCH_CHECK(value.dim() == 2, name, " must be 2-D");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

inline bool intersect_abs_linear_scalar(
    double coefficient,
    double constant,
    double limit,
    double& lower,
    double& upper
) {
    constexpr double eps = 1.0e-14;
    if (std::abs(coefficient) <= eps) {
        return std::abs(constant) <= limit;
    }
    const double a = (-limit - constant) / coefficient;
    const double b = (limit - constant) / coefficient;
    const double lo = std::min(a, b);
    const double hi = std::max(a, b);
    lower = std::max(lower, lo);
    upper = std::min(upper, hi);
    return lower <= upper;
}

inline bool c3_coefficients_from_speeds(
    double v0,
    double v1,
    double right_slope,
    double ds,
    double& c0,
    double& c1,
    double& c2
) {
    if (!(v0 > 0.0) || !(v1 > 0.0) || !(ds > 0.0) ||
        !std::isfinite(v0) || !std::isfinite(v1) ||
        !std::isfinite(right_slope)) {
        return false;
    }
    c0 = v1;
    c1 = right_slope * ds / (2.0 * v1);
    c2 = v0 + c1 - v1;
    if (!std::isfinite(c0) || !std::isfinite(c1) || !std::isfinite(c2)) {
        return false;
    }

    // Endpoints are v(-1)=sqrt(z0)>0 and v(0)=sqrt(z1)>0.
    // A quadratic can dip below both endpoints only at a convex interior
    // vertex.
    if (c2 > 0.0) {
        const double root = -c1 / (2.0 * c2);
        if (root > -1.0 && root < 0.0) {
            const double v = c0 + c1 * root + c2 * root * root;
            if (!(v > 1.0e-14) || !std::isfinite(v)) {
                return false;
            }
        }
    }
    return true;
}

inline bool c3_coefficients(
    double z0,
    double z1,
    double right_slope,
    double ds,
    double& c0,
    double& c1,
    double& c2
) {
    if (!(z0 > 0.0) || !(z1 > 0.0)) {
        return false;
    }
    return c3_coefficients_from_speeds(
        std::sqrt(z0),
        std::sqrt(z1),
        right_slope,
        ds,
        c0,
        c1,
        c2
    );
}

inline double c3_max_speed(
    double c0,
    double c1,
    double c2
) {
    double vmax = std::max(c0, c0 - c1 + c2);
    // Only a concave quadratic can have an interior maximum above both
    // endpoints.
    if (c2 < 0.0) {
        const double root = -c1 / (2.0 * c2);
        if (root > -1.0 && root < 0.0) {
            const double value = c0 + c1 * root + c2 * root * root;
            vmax = std::max(vmax, value);
        }
    }
    return vmax;
}

inline void eval_c3(
    double c0,
    double c1,
    double c2,
    double ds,
    double eta,
    double& z,
    double& zp,
    double& zpp,
    double& speed
) {
    speed = c0 + c1 * eta + c2 * eta * eta;
    const double v_eta = c1 + 2.0 * c2 * eta;
    const double v_etaeta = 2.0 * c2;
    z = speed * speed;
    zp = 2.0 * speed * v_eta / ds;
    zpp = 2.0 * (v_eta * v_eta + speed * v_etaeta) / (ds * ds);
}

bool check_c3_samples(
    double c0,
    double c1,
    double c2,
    double ds,
    const double* eta,
    int64_t samples,
    const double* qs,
    const double* qss,
    const double* qsss,
    const double* m,
    const double* c,
    const double* gravity,
    const double* dm,
    const double* dc,
    const double* dg,
    bool analytic_rate,
    int64_t dof,
    const double* velocity,
    const double* acceleration,
    const double* torque,
    const double* jerk,
    const double* torque_rate,
    double tolerance,
    int64_t stride,
    bool check_torque_rate
) {
    TORCH_CHECK(dof <= 64, "Kaserer DP3 native kernel supports at most 64 DoF");
    TORCH_CHECK(stride >= 1, "stride must be positive");

    double previous_speed = 0.0;
    double previous_tau[64];
    double previous_eta = 0.0;
    bool have_previous = false;

    for (int64_t k = 0;; k = std::min<int64_t>(samples - 1, k + stride)) {
        double z, zp, zpp, speed;
        eval_c3(c0, c1, c2, ds, eta[k], z, zp, zpp, speed);
        if (!(speed > 1.0e-14) || !std::isfinite(speed) ||
            !std::isfinite(z) || !std::isfinite(zp) || !std::isfinite(zpp)) {
            return false;
        }

        const double* qsk = qs + k * dof;
        const double* qssk = qss + k * dof;
        const double* qsssk = qsss + k * dof;
        const double* mk = m + k * dof;
        const double* ck = c + k * dof;
        const double* gk = gravity + k * dof;

        double tau_now[64];
        for (int64_t joint = 0; joint < dof; ++joint) {
            const double qd = qsk[joint] * speed;
            if (std::abs(qd) > velocity[joint] + tolerance) {
                return false;
            }

            const double qdd =
                qssk[joint] * z + 0.5 * qsk[joint] * zp;
            if (std::abs(qdd) > acceleration[joint] + tolerance) {
                return false;
            }

            const double qddd =
                qsssk[joint] * z * speed
                + 1.5 * qssk[joint] * speed * zp
                + 0.5 * qsk[joint] * speed * zpp;
            if (std::abs(qddd) > jerk[joint] + tolerance) {
                return false;
            }

            const double tau =
                0.5 * mk[joint] * zp + ck[joint] * z + gk[joint];
            if (std::abs(tau) > torque[joint] + tolerance) {
                return false;
            }
            tau_now[joint] = tau;
            if (check_torque_rate && analytic_rate) {
                const double rate = speed * (
                    0.5 * dm[k * dof + joint] * zp
                    + 0.5 * mk[joint] * zpp
                    + dc[k * dof + joint] * z
                    + ck[joint] * zp
                    + dg[k * dof + joint]
                );
                if (!std::isfinite(rate) ||
                    std::abs(rate) > torque_rate[joint] + tolerance) return false;
            }
        }

        if (check_torque_rate && !analytic_rate && have_previous) {
            const double delta_eta = eta[k] - previous_eta;
            const double denominator = previous_speed + speed;
            if (!(delta_eta > 0.0) || !(denominator > 1.0e-14)) {
                return false;
            }
            const double dt = 2.0 * ds * delta_eta / denominator;
            if (!(dt > 0.0) || !std::isfinite(dt)) {
                return false;
            }
            for (int64_t joint = 0; joint < dof; ++joint) {
                const double rate =
                    (tau_now[joint] - previous_tau[joint]) / dt;
                if (std::abs(rate) > torque_rate[joint] + tolerance) {
                    return false;
                }
            }
        }

        for (int64_t joint = 0; joint < dof; ++joint) {
            previous_tau[joint] = tau_now[joint];
        }
        previous_speed = speed;
        previous_eta = eta[k];
        have_previous = true;
        if (k == samples - 1) break;
    }
    return true;
}


inline bool c4_coefficients(
    double z0,
    double z1,
    double left_slope,
    double right_slope,
    double ds,
    double& c0,
    double& c1,
    double& c2,
    double& c3
) {
    if (!(z0 > 0.0) || !(z1 > 0.0) || !(ds > 0.0) ||
        !std::isfinite(z0) || !std::isfinite(z1) ||
        !std::isfinite(left_slope) || !std::isfinite(right_slope)) {
        return false;
    }
    const double v0 = std::sqrt(z0);
    const double v1 = std::sqrt(z1);
    c0 = v1;
    c1 = right_slope * ds / (2.0 * v1);
    const double r0 = left_slope * ds / (2.0 * v0);
    const double A = v0 + c1 - v1;
    c3 = r0 + 2.0 * A - c1;
    c2 = r0 + 3.0 * A - c1;
    if (!std::isfinite(c0) || !std::isfinite(c1) ||
        !std::isfinite(c2) || !std::isfinite(c3)) {
        return false;
    }

    // Check the cubic speed polynomial at both endpoints and all stationary
    // points inside [-1,0].
    double min_v = std::min(v0, v1);
    const double a = 3.0 * c3;
    const double b = 2.0 * c2;
    const double c = c1;
    if (std::abs(a) <= 1.0e-14) {
        if (std::abs(b) > 1.0e-14) {
            const double root = -c / b;
            if (root > -1.0 && root < 0.0) {
                const double v =
                    c0 + c1 * root + c2 * root * root
                    + c3 * root * root * root;
                min_v = std::min(min_v, v);
            }
        }
    } else {
        const double disc = b * b - 4.0 * a * c;
        if (disc >= 0.0) {
            const double sd = std::sqrt(disc);
            const double r1 = (-b - sd) / (2.0 * a);
            const double r2 = (-b + sd) / (2.0 * a);
            for (double root : {r1, r2}) {
                if (root > -1.0 && root < 0.0) {
                    const double v =
                        c0 + c1 * root + c2 * root * root
                        + c3 * root * root * root;
                    min_v = std::min(min_v, v);
                }
            }
        }
    }
    return min_v > 1.0e-14 && std::isfinite(min_v);
}

inline void eval_c4(
    double c0,
    double c1,
    double c2,
    double c3,
    double ds,
    double eta,
    double& z,
    double& zp,
    double& zpp,
    double& speed
) {
    const double e2 = eta * eta;
    const double e3 = e2 * eta;
    speed = c0 + c1 * eta + c2 * e2 + c3 * e3;
    const double v_eta = c1 + 2.0 * c2 * eta + 3.0 * c3 * e2;
    const double v_etaeta = 2.0 * c2 + 6.0 * c3 * eta;
    z = speed * speed;
    zp = 2.0 * speed * v_eta / ds;
    zpp = 2.0 * (v_eta * v_eta + speed * v_etaeta) / (ds * ds);
}

bool check_c4_samples(
    double c0,
    double c1,
    double c2,
    double c3,
    double ds,
    const double* eta,
    int64_t samples,
    const double* qs,
    const double* qss,
    const double* qsss,
    const double* m,
    const double* c,
    const double* gravity,
    const double* dm,
    const double* dc,
    const double* dg,
    bool analytic_rate,
    int64_t dof,
    const double* velocity,
    const double* acceleration,
    const double* torque,
    const double* jerk,
    const double* torque_rate,
    double tolerance,
    int64_t stride,
    bool check_torque_rate
) {
    TORCH_CHECK(dof <= 64, "Kaserer DP3 native kernel supports at most 64 DoF");
    TORCH_CHECK(stride >= 1, "stride must be positive");
    double previous_speed = 0.0;
    double previous_tau[64];
    double previous_eta = 0.0;
    bool have_previous = false;

    for (int64_t k = 0;; k = std::min<int64_t>(samples - 1, k + stride)) {
        double z, zp, zpp, speed;
        eval_c4(c0, c1, c2, c3, ds, eta[k], z, zp, zpp, speed);
        if (!(speed > 1.0e-14) || !std::isfinite(speed) ||
            !std::isfinite(z) || !std::isfinite(zp) || !std::isfinite(zpp)) {
            return false;
        }

        const double* qsk = qs + k * dof;
        const double* qssk = qss + k * dof;
        const double* qsssk = qsss + k * dof;
        const double* mk = m + k * dof;
        const double* ck = c + k * dof;
        const double* gk = gravity + k * dof;
        double tau_now[64];

        for (int64_t joint = 0; joint < dof; ++joint) {
            const double qd = qsk[joint] * speed;
            if (std::abs(qd) > velocity[joint] + tolerance) return false;

            const double qdd =
                qssk[joint] * z + 0.5 * qsk[joint] * zp;
            if (std::abs(qdd) > acceleration[joint] + tolerance) return false;

            const double qddd =
                qsssk[joint] * z * speed
                + 1.5 * qssk[joint] * speed * zp
                + 0.5 * qsk[joint] * speed * zpp;
            if (std::abs(qddd) > jerk[joint] + tolerance) return false;

            const double tau =
                0.5 * mk[joint] * zp + ck[joint] * z + gk[joint];
            if (std::abs(tau) > torque[joint] + tolerance) return false;
            tau_now[joint] = tau;
            if (check_torque_rate && analytic_rate) {
                const double rate = speed * (
                    0.5 * dm[k * dof + joint] * zp
                    + 0.5 * mk[joint] * zpp
                    + dc[k * dof + joint] * z
                    + ck[joint] * zp
                    + dg[k * dof + joint]
                );
                if (!std::isfinite(rate) ||
                    std::abs(rate) > torque_rate[joint] + tolerance) return false;
            }
        }

        if (check_torque_rate && !analytic_rate && have_previous) {
            const double delta_eta = eta[k] - previous_eta;
            const double denominator = previous_speed + speed;
            if (!(delta_eta > 0.0) || !(denominator > 1.0e-14)) return false;
            const double dt = 2.0 * ds * delta_eta / denominator;
            if (!(dt > 0.0) || !std::isfinite(dt)) return false;
            for (int64_t joint = 0; joint < dof; ++joint) {
                const double rate =
                    (tau_now[joint] - previous_tau[joint]) / dt;
                if (std::abs(rate) > torque_rate[joint] + tolerance) return false;
            }
        }

        for (int64_t joint = 0; joint < dof; ++joint) {
            previous_tau[joint] = tau_now[joint];
        }
        previous_speed = speed;
        previous_eta = eta[k];
        have_previous = true;
        if (k == samples - 1) break;
    }
    return true;
}

bool check_c2_samples(
    double z0,
    double z1,
    double ds,
    const double* eta,
    int64_t samples,
    const double* qs,
    const double* qss,
    const double* qsss,
    const double* m,
    const double* c,
    const double* gravity,
    const double* dm,
    const double* dc,
    const double* dg,
    bool analytic_rate,
    int64_t dof,
    const double* velocity,
    const double* acceleration,
    const double* torque,
    const double* jerk,
    const double* torque_rate,
    double tolerance,
    int64_t stride,
    bool check_torque_rate
) {
    TORCH_CHECK(dof <= 64, "Kaserer DP3 native kernel supports at most 64 DoF");
    TORCH_CHECK(stride >= 1, "stride must be positive");
    const double coeff = z1 - z0;
    const double zp = coeff / ds;
    const double zpp = 0.0;
    double previous_speed = 0.0;
    double previous_tau[64];
    double previous_eta = 0.0;
    bool have_previous = false;

    for (int64_t k = 0;; k = std::min<int64_t>(samples - 1, k + stride)) {
        const double z = z1 + coeff * eta[k];
        if (z < -tolerance || !std::isfinite(z)) return false;
        const double speed = std::sqrt(std::max(0.0, z));

        const double* qsk = qs + k * dof;
        const double* qssk = qss + k * dof;
        const double* qsssk = qsss + k * dof;
        const double* mk = m + k * dof;
        const double* ck = c + k * dof;
        const double* gk = gravity + k * dof;
        double tau_now[64];

        for (int64_t joint = 0; joint < dof; ++joint) {
            const double qd = qsk[joint] * speed;
            if (std::abs(qd) > velocity[joint] + tolerance) return false;

            const double qdd =
                qssk[joint] * z + 0.5 * qsk[joint] * zp;
            if (std::abs(qdd) > acceleration[joint] + tolerance) return false;

            const double qddd =
                qsssk[joint] * z * speed
                + 1.5 * qssk[joint] * speed * zp
                + 0.5 * qsk[joint] * speed * zpp;
            if (std::abs(qddd) > jerk[joint] + tolerance) return false;

            const double tau =
                0.5 * mk[joint] * zp + ck[joint] * z + gk[joint];
            if (std::abs(tau) > torque[joint] + tolerance) return false;
            tau_now[joint] = tau;
            if (check_torque_rate && analytic_rate) {
                const double rate = speed * (
                    0.5 * dm[k * dof + joint] * zp
                    + 0.5 * mk[joint] * zpp
                    + dc[k * dof + joint] * z
                    + ck[joint] * zp
                    + dg[k * dof + joint]
                );
                if (!std::isfinite(rate) ||
                    std::abs(rate) > torque_rate[joint] + tolerance) return false;
            }
        }

        if (check_torque_rate && !analytic_rate && have_previous) {
            const double delta_eta = eta[k] - previous_eta;
            const double denominator = previous_speed + speed;
            if (!(delta_eta > 0.0) || !(denominator > 1.0e-14)) return false;
            const double dt = 2.0 * ds * delta_eta / denominator;
            if (!(dt > 0.0) || !std::isfinite(dt)) return false;
            for (int64_t joint = 0; joint < dof; ++joint) {
                const double rate =
                    (tau_now[joint] - previous_tau[joint]) / dt;
                if (std::abs(rate) > torque_rate[joint] + tolerance) return false;
            }
        }

        for (int64_t joint = 0; joint < dof; ++joint) {
            previous_tau[joint] = tau_now[joint];
        }
        previous_speed = speed;
        previous_eta = eta[k];
        have_previous = true;
        if (k == samples - 1) break;
    }
    return true;
}

inline double c4_time(
    double c0,
    double c1,
    double c2,
    double c3,
    double ds,
    const double* eta,
    const double* weight,
    int64_t samples
) {
    double integral = 0.0;
    for (int64_t k = 0; k < samples; ++k) {
        const double e = eta[k];
        const double speed =
            c0 + c1 * e + c2 * e * e + c3 * e * e * e;
        if (!(speed > 1.0e-14) || !std::isfinite(speed)) {
            return std::numeric_limits<double>::infinity();
        }
        integral += weight[k] / speed;
    }
    return ds * integral;
}

inline double c2_time(double z0, double z1, double ds) {
    const double denom =
        std::sqrt(std::max(0.0, z0)) + std::sqrt(std::max(0.0, z1));
    if (!(denom > 0.0) || !std::isfinite(denom)) {
        return std::numeric_limits<double>::infinity();
    }
    return 2.0 * ds / denom;
}

inline double c3_time(
    double c0,
    double c1,
    double c2,
    double ds,
    const double* eta,
    const double* weight,
    int64_t samples
) {
    double integral = 0.0;
    for (int64_t k = 0; k < samples; ++k) {
        const double x = eta[k];
        const double speed = c0 + c1 * x + c2 * x * x;
        if (!(speed > 1.0e-14) || !std::isfinite(speed)) {
            return std::numeric_limits<double>::infinity();
        }
        integral += weight[k] / speed;
    }
    return ds * integral;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> search_ranges(
    torch::Tensor current_z,
    torch::Tensor next_z,
    double ds,
    torch::Tensor node_qs,
    torch::Tensor node_qss,
    torch::Tensor node_m,
    torch::Tensor node_c,
    torch::Tensor node_gravity,
    torch::Tensor acceleration,
    torch::Tensor torque
) {
    require_1d_double_cpu(current_z, "current_z");
    require_1d_double_cpu(next_z, "next_z");
    require_1d_double_cpu(node_qs, "node_qs");
    require_1d_double_cpu(node_qss, "node_qss");
    require_1d_double_cpu(node_m, "node_m");
    require_1d_double_cpu(node_c, "node_c");
    require_1d_double_cpu(node_gravity, "node_gravity");
    require_1d_double_cpu(acceleration, "acceleration");
    require_1d_double_cpu(torque, "torque");
    TORCH_CHECK(ds > 0.0 && std::isfinite(ds), "ds must be positive");

    const int64_t dof = node_qs.numel();
    TORCH_CHECK(
        node_qss.numel() == dof && node_m.numel() == dof &&
        node_c.numel() == dof && node_gravity.numel() == dof &&
        acceleration.numel() == dof && torque.numel() == dof,
        "node/limit size mismatch"
    );

    const int64_t rows = current_z.numel();
    const int64_t cols = next_z.numel();
    auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto out_lo = torch::full({rows}, -1, i64);
    auto out_hi = torch::full({rows}, -1, i64);

    const double* z_ptr = current_z.data_ptr<double>();
    const double* nz = next_z.data_ptr<double>();
    const double* qs = node_qs.data_ptr<double>();
    const double* qss = node_qss.data_ptr<double>();
    const double* m = node_m.data_ptr<double>();
    const double* c = node_c.data_ptr<double>();
    const double* g = node_gravity.data_ptr<double>();
    const double* acc = acceleration.data_ptr<double>();
    const double* tau = torque.data_ptr<double>();
    int64_t* lo_ptr = out_lo.data_ptr<int64_t>();
    int64_t* hi_ptr = out_hi.data_ptr<int64_t>();

    at::parallel_for(1, rows, 1, [&](int64_t begin, int64_t end) {
        for (int64_t j = begin; j < end; ++j) {
            const double z = z_ptr[j];
            if (!(z > 0.0) || !std::isfinite(z)) continue;

            double lower = -std::numeric_limits<double>::infinity();
            double upper = std::numeric_limits<double>::infinity();
            bool valid = true;
            for (int64_t joint = 0; joint < dof; ++joint) {
                if (!intersect_abs_linear_scalar(
                        0.5 * qs[joint],
                        qss[joint] * z,
                        acc[joint],
                        lower,
                        upper)) {
                    valid = false;
                    break;
                }
                if (!intersect_abs_linear_scalar(
                        0.5 * m[joint],
                        c[joint] * z + g[joint],
                        tau[joint],
                        lower,
                        upper)) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;

            const double a = z + lower * ds;
            const double b = z + upper * ds;
            const double search_lower = std::min(a, b);
            const double search_upper = std::max(a, b);
            if (
                !(std::isfinite(search_lower) ||
                  std::isinf(search_lower)) ||
                !(std::isfinite(search_upper) ||
                  std::isinf(search_upper))
            ) {
                continue;
            }

            const double* first = nz;
            const double* last = nz + cols;
            int64_t lo = static_cast<int64_t>(
                std::lower_bound(first, last, search_lower) - first
            );
            int64_t hi_exclusive = static_cast<int64_t>(
                std::upper_bound(first, last, search_upper) - first
            );
            lo = std::max<int64_t>(0, lo);
            const int64_t hi = std::min<int64_t>(cols - 1, hi_exclusive - 1);
            if (lo <= hi) {
                lo_ptr[j] = lo;
                hi_ptr[j] = hi;
            }
        }
    });

    return {out_lo, out_hi};
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> c3_stage(
    torch::Tensor current_z,
    torch::Tensor next_z,
    torch::Tensor next_cost,
    torch::Tensor next_slope,
    torch::Tensor search_lo,
    torch::Tensor search_hi,
    double ds,
    torch::Tensor eta,
    torch::Tensor edge_qs,
    torch::Tensor edge_qss,
    torch::Tensor edge_qsss,
    torch::Tensor edge_m,
    torch::Tensor edge_c,
    torch::Tensor edge_gravity,
    torch::Tensor quad_eta,
    torch::Tensor quad_weight,
    torch::Tensor velocity,
    torch::Tensor acceleration,
    torch::Tensor torque,
    torch::Tensor jerk,
    torch::Tensor torque_rate,
    double tolerance,
    torch::Tensor edge_dm,
    torch::Tensor edge_dc,
    torch::Tensor edge_dg,
    bool analytic_rate
) {
    if (analytic_rate) {
        require_2d_double_cpu(edge_dm, "edge_dm");
        require_2d_double_cpu(edge_dc, "edge_dc");
        require_2d_double_cpu(edge_dg, "edge_dg");
        TORCH_CHECK(
            edge_dm.sizes() == edge_m.sizes() &&
            edge_dc.sizes() == edge_m.sizes() &&
            edge_dg.sizes() == edge_m.sizes(),
            "analytic rate coefficient tensor shape mismatch"
        );
    }
    require_1d_double_cpu(current_z, "current_z");
    require_1d_double_cpu(next_z, "next_z");
    require_1d_double_cpu(next_cost, "next_cost");
    require_1d_double_cpu(next_slope, "next_slope");
    require_1d_int64_cpu(search_lo, "search_lo");
    require_1d_int64_cpu(search_hi, "search_hi");
    require_1d_double_cpu(eta, "eta");
    require_1d_double_cpu(quad_eta, "quad_eta");
    require_1d_double_cpu(quad_weight, "quad_weight");
    require_1d_double_cpu(velocity, "velocity");
    require_1d_double_cpu(acceleration, "acceleration");
    require_1d_double_cpu(torque, "torque");
    require_1d_double_cpu(jerk, "jerk");
    require_1d_double_cpu(torque_rate, "torque_rate");
    require_2d_double_cpu(edge_qs, "edge_qs");
    require_2d_double_cpu(edge_qss, "edge_qss");
    require_2d_double_cpu(edge_qsss, "edge_qsss");
    require_2d_double_cpu(edge_m, "edge_m");
    require_2d_double_cpu(edge_c, "edge_c");
    require_2d_double_cpu(edge_gravity, "edge_gravity");

    const int64_t eta_samples = eta.numel();
    const int64_t quad_samples = quad_eta.numel();

    TORCH_CHECK(ds > 0.0 && std::isfinite(ds), "ds must be finite and positive");
    TORCH_CHECK(next_z.numel() == next_cost.numel(), "next cost size mismatch");
    TORCH_CHECK(next_z.numel() == next_slope.numel(), "next slope size mismatch");
    TORCH_CHECK(search_lo.numel() == current_z.numel(), "search_lo size mismatch");
    TORCH_CHECK(search_hi.numel() == current_z.numel(), "search_hi size mismatch");
    TORCH_CHECK(quad_samples == quad_weight.numel(), "quadrature size mismatch");

    const int64_t dof = velocity.numel();
    TORCH_CHECK(
        acceleration.numel() == dof && torque.numel() == dof &&
        jerk.numel() == dof && torque_rate.numel() == dof,
        "limit size mismatch"
    );
    TORCH_CHECK(
        edge_qs.size(0) == eta_samples && edge_qs.size(1) == dof,
        "edge geometry shape mismatch"
    );
    TORCH_CHECK(
        edge_qss.sizes() == edge_qs.sizes() &&
        edge_qsss.sizes() == edge_qs.sizes() &&
        edge_m.sizes() == edge_qs.sizes() &&
        edge_c.sizes() == edge_qs.sizes() &&
        edge_gravity.sizes() == edge_qs.sizes(),
        "edge geometry arrays must have identical shape"
    );

    const int64_t rows = current_z.numel();
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto int_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto output_cost = torch::full(
        {rows}, std::numeric_limits<double>::infinity(), options
    );
    auto output_successor = torch::full({rows}, -1, int_options);
    auto output_slope = torch::full(
        {rows}, std::numeric_limits<double>::quiet_NaN(), options
    );
    auto checked = torch::zeros({rows}, int_options);
    auto feasible = torch::zeros({rows}, int_options);

    const double* cz = current_z.data_ptr<double>();
    const double* nz = next_z.data_ptr<double>();
    const double* nc = next_cost.data_ptr<double>();
    const double* ns = next_slope.data_ptr<double>();
    const int64_t* search_lo_ptr = search_lo.data_ptr<int64_t>();
    const int64_t* search_hi_ptr = search_hi.data_ptr<int64_t>();

    const double* ep = eta.data_ptr<double>();
    const double* eqs = edge_qs.data_ptr<double>();
    const double* eqss = edge_qss.data_ptr<double>();
    const double* eqsss = edge_qsss.data_ptr<double>();
    const double* em = edge_m.data_ptr<double>();
    const double* ec = edge_c.data_ptr<double>();
    const double* eg = edge_gravity.data_ptr<double>();

    const double* qeta = quad_eta.data_ptr<double>();
    const double* qw = quad_weight.data_ptr<double>();
    double quad_weight_sum = 0.0;
    for (int64_t k = 0; k < quad_weight.numel(); ++k) {
        quad_weight_sum += qw[k];
    }

    const double* vel = velocity.data_ptr<double>();
    const double* acc = acceleration.data_ptr<double>();
    const double* tau_lim = torque.data_ptr<double>();
    const double* jerk_lim = jerk.data_ptr<double>();
    const double* rate_lim = torque_rate.data_ptr<double>();
    const double* dmp = analytic_rate ? edge_dm.data_ptr<double>() : nullptr;
    const double* dcp = analytic_rate ? edge_dc.data_ptr<double>() : nullptr;
    const double* dgp = analytic_rate ? edge_dg.data_ptr<double>() : nullptr;

    double* out_cost = output_cost.data_ptr<double>();
    int64_t* out_successor = output_successor.data_ptr<int64_t>();
    double* out_slope = output_slope.data_ptr<double>();
    int64_t* out_checked = checked.data_ptr<int64_t>();
    int64_t* out_feasible = feasible.data_ptr<int64_t>();

    const int64_t cols = next_z.numel();
    const int64_t screen_stride_17 =
        eta_samples > 33
            ? std::max<int64_t>(1, (eta_samples - 1) / 16)
            : 1;
    const bool seeded_interval_search = rows >= 1024;
    const bool deep_prescreen =
        seeded_interval_search && eta_samples >= 129;
    const int64_t screen_stride_33 =
        deep_prescreen
            ? std::max<int64_t>(1, (eta_samples - 1) / 32)
            : 1;
    const int64_t screen_stride_65 =
        deep_prescreen
            ? std::max<int64_t>(1, (eta_samples - 1) / 64)
            : 1;

    std::vector<double> next_speed(static_cast<size_t>(cols), 0.0);
    for (int64_t l = 0; l < cols; ++l) {
        if (nz[l] > 0.0 && std::isfinite(nz[l])) {
            next_speed[static_cast<size_t>(l)] = std::sqrt(nz[l]);
        }
    }

    // One exact lower-bound ordering for the whole stage. Because every edge
    // traversal time is strictly positive, next_cost alone is a valid lower
    // bound on total cost.
    std::vector<int64_t> successor_order;
    successor_order.reserve(static_cast<size_t>(cols));
    for (int64_t l = 0; l < cols; ++l) {
        if (
            std::isfinite(nc[l]) &&
            nz[l] > 0.0 &&
            std::isfinite(ns[l])
        ) {
            successor_order.push_back(l);
        }
    }
    std::sort(
        successor_order.begin(),
        successor_order.end(),
        [&](int64_t a, int64_t b) {
            if (nc[a] < nc[b]) return true;
            if (nc[a] > nc[b]) return false;
            return a < b;
        }
    );

    at::parallel_for(1, rows, 1, [&](int64_t begin, int64_t end) {
        for (int64_t j = begin; j < end; ++j) {
            const double z0 = cz[j];
            if (!(z0 > 0.0) || !std::isfinite(z0)) {
                continue;
            }
            const double v0 = std::sqrt(z0);

            const int64_t lo = search_lo_ptr[j];
            const int64_t hi = search_hi_ptr[j];
            if (lo < 0 || hi < lo || lo >= cols) {
                continue;
            }
            const int64_t hi_exclusive = std::min<int64_t>(cols, hi + 1);

            double best = std::numeric_limits<double>::infinity();
            int64_t best_l = -1;
            double best_left_slope = std::numeric_limits<double>::quiet_NaN();
            int64_t checked_local = 0;
            int64_t feasible_local = 0;

            // Phase 1: use the global next-cost ordering only to seed a
            // strong feasible incumbent inside this state's reachable range.
            // This usually takes very few expensive candidate evaluations.
            int64_t seed_l = -1;
            for (const int64_t l : successor_order) {
                if (
                    !seeded_interval_search &&
                    best_l >= 0 &&
                    nc[l] >= best
                ) {
                    break;
                }
                if (l < lo || l >= hi_exclusive) {
                    continue;
                }

                const double right_slope = ns[l];

                double c0, c1, c2;
                if (!c3_coefficients_from_speeds(
                        v0,
                        next_speed[static_cast<size_t>(l)],
                        right_slope,
                        ds,
                        c0,
                        c1,
                        c2
                    )) {
                    continue;
                }
                ++checked_local;

                // Exact lower bound on this kernel's quadrature edge time.
                if (best_l >= 0) {
                    const double vmax = c3_max_speed(c0, c1, c2);
                    if (vmax > 0.0 && std::isfinite(vmax)) {
                        const double total_lb =
                            nc[l] + ds * quad_weight_sum / vmax;
                        if (
                            total_lb > best ||
                            (total_lb == best && l >= best_l)
                        ) {
                            continue;
                        }
                    }
                }

                if (screen_stride_17 > 1 &&
                    !check_c3_samples(
                        c0, c1, c2, ds,
                        ep, eta_samples,
                        eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                        dof,
                        vel, acc, tau_lim, jerk_lim, rate_lim,
                        tolerance, screen_stride_17, false)) {
                    continue;
                }

                // The deeper pointwise cascade only pays off for large-z
                // stages. Keep the small benchmark on the cheaper 17->full
                // path.
                if (deep_prescreen) {
                    if (
                        screen_stride_33 > 1 &&
                        screen_stride_33 < screen_stride_17 &&
                        !check_c3_samples(
                            c0, c1, c2, ds,
                            ep, eta_samples,
                            eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                            dof,
                            vel, acc, tau_lim, jerk_lim, rate_lim,
                            tolerance, screen_stride_33, false)
                    ) {
                        continue;
                    }

                    if (
                        screen_stride_65 > 1 &&
                        screen_stride_65 < screen_stride_33 &&
                        !check_c3_samples(
                            c0, c1, c2, ds,
                            ep, eta_samples,
                            eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                            dof,
                            vel, acc, tau_lim, jerk_lim, rate_lim,
                            tolerance, screen_stride_65, false)
                    ) {
                        continue;
                    }
                }

                // Delay quadrature until after the cheap pointwise screens.
                const double edge = c3_time(
                    c0, c1, c2, ds,
                    qeta, qw, quad_samples
                );
                if (!std::isfinite(edge)) {
                    continue;
                }
                const double total = edge + nc[l];
                if (
                    best_l >= 0 &&
                    (total > best || (total == best && l >= best_l))
                ) {
                    continue;
                }

                if (!check_c3_samples(
                        c0, c1, c2, ds,
                        ep, eta_samples,
                        eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                        dof,
                        vel, acc, tau_lim, jerk_lim, rate_lim,
                        tolerance, 1, true)) {
                    continue;
                }
                ++feasible_local;

                if (
                    total < best ||
                    (total == best && (best_l < 0 || l < best_l))
                ) {
                    best = total;
                    best_l = l;
                    const double v_eta_left = c1 - 2.0 * c2;
                    best_left_slope = 2.0 * v0 * v_eta_left / ds;
                }

                if (seeded_interval_search && best_l >= 0) {
                    seed_l = best_l;
                    break;
                }
            }

            // Phase 2: scan only the exact reachable interval.  Once an
            // incumbent exists, next_cost >= best is enough to reject a
            // candidate without constructing its C3 profile because edge
            // traversal time is strictly positive.
            if (seeded_interval_search && best_l >= 0) {
                for (int64_t l = lo; l < hi_exclusive; ++l) {
                    if (l == seed_l) {
                        continue;
                    }
                    if (!std::isfinite(nc[l]) || nc[l] >= best) {
                        continue;
                    }
                    if (!(nz[l] > 0.0) || !std::isfinite(ns[l])) {
                        continue;
                    }

                const double right_slope = ns[l];

                double c0, c1, c2;
                if (!c3_coefficients_from_speeds(
                        v0,
                        next_speed[static_cast<size_t>(l)],
                        right_slope,
                        ds,
                        c0,
                        c1,
                        c2
                    )) {
                    continue;
                }
                ++checked_local;

                // Exact lower bound on this kernel's quadrature edge time.
                if (best_l >= 0) {
                    const double vmax = c3_max_speed(c0, c1, c2);
                    if (vmax > 0.0 && std::isfinite(vmax)) {
                        const double total_lb =
                            nc[l] + ds * quad_weight_sum / vmax;
                        if (
                            total_lb > best ||
                            (total_lb == best && l >= best_l)
                        ) {
                            continue;
                        }
                    }
                }

                if (screen_stride_17 > 1 &&
                    !check_c3_samples(
                        c0, c1, c2, ds,
                        ep, eta_samples,
                        eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                        dof,
                        vel, acc, tau_lim, jerk_lim, rate_lim,
                        tolerance, screen_stride_17, false)) {
                    continue;
                }

                // The deeper pointwise cascade only pays off for large-z
                // stages. Keep the small benchmark on the cheaper 17->full
                // path.
                if (deep_prescreen) {
                    if (
                        screen_stride_33 > 1 &&
                        screen_stride_33 < screen_stride_17 &&
                        !check_c3_samples(
                            c0, c1, c2, ds,
                            ep, eta_samples,
                            eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                            dof,
                            vel, acc, tau_lim, jerk_lim, rate_lim,
                            tolerance, screen_stride_33, false)
                    ) {
                        continue;
                    }

                    if (
                        screen_stride_65 > 1 &&
                        screen_stride_65 < screen_stride_33 &&
                        !check_c3_samples(
                            c0, c1, c2, ds,
                            ep, eta_samples,
                            eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                            dof,
                            vel, acc, tau_lim, jerk_lim, rate_lim,
                            tolerance, screen_stride_65, false)
                    ) {
                        continue;
                    }
                }

                // Delay quadrature until after the cheap pointwise screens.
                const double edge = c3_time(
                    c0, c1, c2, ds,
                    qeta, qw, quad_samples
                );
                if (!std::isfinite(edge)) {
                    continue;
                }
                const double total = edge + nc[l];
                if (
                    best_l >= 0 &&
                    (total > best || (total == best && l >= best_l))
                ) {
                    continue;
                }

                if (!check_c3_samples(
                        c0, c1, c2, ds,
                        ep, eta_samples,
                        eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                        dof,
                        vel, acc, tau_lim, jerk_lim, rate_lim,
                        tolerance, 1, true)) {
                    continue;
                }
                ++feasible_local;

                if (
                    total < best ||
                    (total == best && (best_l < 0 || l < best_l))
                ) {
                    best = total;
                    best_l = l;
                    const double v_eta_left = c1 - 2.0 * c2;
                    best_left_slope = 2.0 * v0 * v_eta_left / ds;
                }

                }
            }

            out_checked[j] = checked_local;
            out_feasible[j] = feasible_local;
            if (best_l >= 0) {
                out_cost[j] = best;
                out_successor[j] = best_l;
                out_slope[j] = best_left_slope;
            }
        }
    });

    return {
        output_cost,
        output_successor,
        output_slope,
        checked,
        feasible
    };
}


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> start_stage(
    double z_start,
    torch::Tensor node2_z,
    torch::Tensor stage3_z,
    torch::Tensor stage3_cost,
    torch::Tensor stage3_slope,
    torch::Tensor first_valid,
    torch::Tensor second_lo,
    torch::Tensor second_hi,
    double ds0,
    double ds1,
    torch::Tensor eta0,
    torch::Tensor eta1,
    torch::Tensor edge0_qs,
    torch::Tensor edge0_qss,
    torch::Tensor edge0_qsss,
    torch::Tensor edge0_m,
    torch::Tensor edge0_c,
    torch::Tensor edge0_gravity,
    torch::Tensor edge1_qs,
    torch::Tensor edge1_qss,
    torch::Tensor edge1_qsss,
    torch::Tensor edge1_m,
    torch::Tensor edge1_c,
    torch::Tensor edge1_gravity,
    torch::Tensor quad_eta,
    torch::Tensor quad_weight,
    torch::Tensor velocity,
    torch::Tensor acceleration,
    torch::Tensor torque,
    torch::Tensor jerk,
    torch::Tensor torque_rate,
    double tolerance,
    torch::Tensor edge0_dm,
    torch::Tensor edge0_dc,
    torch::Tensor edge0_dg,
    torch::Tensor edge1_dm,
    torch::Tensor edge1_dc,
    torch::Tensor edge1_dg,
    bool analytic_rate
) {
    if (analytic_rate) {
        require_2d_double_cpu(edge0_dm, "edge0_dm");
        require_2d_double_cpu(edge0_dc, "edge0_dc");
        require_2d_double_cpu(edge0_dg, "edge0_dg");
        require_2d_double_cpu(edge1_dm, "edge1_dm");
        require_2d_double_cpu(edge1_dc, "edge1_dc");
        require_2d_double_cpu(edge1_dg, "edge1_dg");
        TORCH_CHECK(
            edge0_dm.sizes() == edge0_m.sizes() &&
            edge0_dc.sizes() == edge0_m.sizes() &&
            edge0_dg.sizes() == edge0_m.sizes() &&
            edge1_dm.sizes() == edge1_m.sizes() &&
            edge1_dc.sizes() == edge1_m.sizes() &&
            edge1_dg.sizes() == edge1_m.sizes(),
            "analytic start-stage coefficient tensor shape mismatch"
        );
    }
    require_1d_double_cpu(node2_z, "node2_z");
    require_1d_double_cpu(stage3_z, "stage3_z");
    require_1d_double_cpu(stage3_cost, "stage3_cost");
    require_1d_double_cpu(stage3_slope, "stage3_slope");
    require_1d_int64_cpu(first_valid, "first_valid");
    require_1d_int64_cpu(second_lo, "second_lo");
    require_1d_int64_cpu(second_hi, "second_hi");
    require_1d_double_cpu(eta0, "eta0");
    require_1d_double_cpu(eta1, "eta1");
    require_1d_double_cpu(quad_eta, "quad_eta");
    require_1d_double_cpu(quad_weight, "quad_weight");
    require_1d_double_cpu(velocity, "velocity");
    require_1d_double_cpu(acceleration, "acceleration");
    require_1d_double_cpu(torque, "torque");
    require_1d_double_cpu(jerk, "jerk");
    require_1d_double_cpu(torque_rate, "torque_rate");

    for (const auto& pair : std::vector<std::pair<torch::Tensor, const char*>>{
        {edge0_qs, "edge0_qs"}, {edge0_qss, "edge0_qss"},
        {edge0_qsss, "edge0_qsss"}, {edge0_m, "edge0_m"},
        {edge0_c, "edge0_c"}, {edge0_gravity, "edge0_gravity"},
        {edge1_qs, "edge1_qs"}, {edge1_qss, "edge1_qss"},
        {edge1_qsss, "edge1_qsss"}, {edge1_m, "edge1_m"},
        {edge1_c, "edge1_c"}, {edge1_gravity, "edge1_gravity"}}) {
        require_2d_double_cpu(pair.first, pair.second);
    }

    TORCH_CHECK(ds0 > 0.0 && ds1 > 0.0, "ds must be positive");
    TORCH_CHECK(stage3_z.numel() == stage3_cost.numel(), "stage3 cost size mismatch");
    TORCH_CHECK(stage3_z.numel() == stage3_slope.numel(), "stage3 slope size mismatch");
    TORCH_CHECK(first_valid.numel() == node2_z.numel(), "first_valid size mismatch");
    TORCH_CHECK(second_lo.numel() == node2_z.numel(), "second_lo size mismatch");
    TORCH_CHECK(second_hi.numel() == node2_z.numel(), "second_hi size mismatch");

    const int64_t dof = velocity.numel();
    TORCH_CHECK(dof <= 64, "Kaserer DP3 native kernel supports at most 64 DoF");
    TORCH_CHECK(
        acceleration.numel() == dof && torque.numel() == dof &&
        jerk.numel() == dof && torque_rate.numel() == dof,
        "limit size mismatch"
    );
    TORCH_CHECK(
        edge0_qs.size(0) == eta0.numel() && edge0_qs.size(1) == dof &&
        edge1_qs.size(0) == eta1.numel() && edge1_qs.size(1) == dof,
        "edge geometry shape mismatch"
    );

    const int64_t rows = node2_z.numel();
    auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto out_cost = torch::full(
        {rows}, std::numeric_limits<double>::infinity(), f64
    );
    auto out_successor = torch::full({rows}, -1, i64);
    auto out_slope = torch::full(
        {rows}, std::numeric_limits<double>::quiet_NaN(), f64
    );
    auto out_first_cost = torch::full(
        {rows}, std::numeric_limits<double>::infinity(), f64
    );
    auto checked = torch::zeros({rows}, i64);
    auto feasible = torch::zeros({rows}, i64);

    const double* z2_ptr = node2_z.data_ptr<double>();
    const double* z3_ptr = stage3_z.data_ptr<double>();
    const double* cost3_ptr = stage3_cost.data_ptr<double>();
    const double* slope3_ptr = stage3_slope.data_ptr<double>();
    const int64_t* first_ptr = first_valid.data_ptr<int64_t>();
    const int64_t* lo_ptr = second_lo.data_ptr<int64_t>();
    const int64_t* hi_ptr = second_hi.data_ptr<int64_t>();
    const double* eta0_ptr = eta0.data_ptr<double>();
    const double* eta1_ptr = eta1.data_ptr<double>();
    const double* qeta = quad_eta.data_ptr<double>();
    const double* qw = quad_weight.data_ptr<double>();
    const double* vel = velocity.data_ptr<double>();
    const double* acc = acceleration.data_ptr<double>();
    const double* tau_lim = torque.data_ptr<double>();
    const double* jerk_lim = jerk.data_ptr<double>();
    const double* rate_lim = torque_rate.data_ptr<double>();
    const double* dm0 = analytic_rate ? edge0_dm.data_ptr<double>() : nullptr;
    const double* dc0 = analytic_rate ? edge0_dc.data_ptr<double>() : nullptr;
    const double* dg0 = analytic_rate ? edge0_dg.data_ptr<double>() : nullptr;
    const double* dm1 = analytic_rate ? edge1_dm.data_ptr<double>() : nullptr;
    const double* dc1 = analytic_rate ? edge1_dc.data_ptr<double>() : nullptr;
    const double* dg1 = analytic_rate ? edge1_dg.data_ptr<double>() : nullptr;

    const double* e0_qs = edge0_qs.data_ptr<double>();
    const double* e0_qss = edge0_qss.data_ptr<double>();
    const double* e0_qsss = edge0_qsss.data_ptr<double>();
    const double* e0_m = edge0_m.data_ptr<double>();
    const double* e0_c = edge0_c.data_ptr<double>();
    const double* e0_g = edge0_gravity.data_ptr<double>();

    const double* e1_qs = edge1_qs.data_ptr<double>();
    const double* e1_qss = edge1_qss.data_ptr<double>();
    const double* e1_qsss = edge1_qsss.data_ptr<double>();
    const double* e1_m = edge1_m.data_ptr<double>();
    const double* e1_c = edge1_c.data_ptr<double>();
    const double* e1_g = edge1_gravity.data_ptr<double>();

    double* out_cost_ptr = out_cost.data_ptr<double>();
    int64_t* out_succ_ptr = out_successor.data_ptr<int64_t>();
    double* out_slope_ptr = out_slope.data_ptr<double>();
    double* out_first_ptr = out_first_cost.data_ptr<double>();
    int64_t* checked_ptr = checked.data_ptr<int64_t>();
    int64_t* feasible_ptr = feasible.data_ptr<int64_t>();
    const int64_t cols = stage3_z.numel();

    at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
        for (int64_t j = begin; j < end; ++j) {
            if (first_ptr[j] == 0) continue;
            const double z2 = z2_ptr[j];

            int64_t checked_local = 1;
            const int64_t screen0_stride_17 =
                eta0.numel() > 33
                    ? std::max<int64_t>(1, (eta0.numel() - 1) / 16)
                    : 1;
            if (screen0_stride_17 > 1 &&
                !check_c2_samples(
                    z_start, z2, ds0,
                    eta0_ptr, eta0.numel(),
                    e0_qs, e0_qss, e0_qsss, e0_m, e0_c, e0_g,
                        dm0, dc0, dg0, analytic_rate,
                    dof, vel, acc, tau_lim, jerk_lim, rate_lim,
                    tolerance, screen0_stride_17, false)) {
                checked_ptr[j] = checked_local;
                continue;
            }
            const int64_t screen0_stride_65 =
                eta0.numel() > 129
                    ? std::max<int64_t>(1, (eta0.numel() - 1) / 64)
                    : 1;
            if (screen0_stride_65 > 1 &&
                screen0_stride_65 < screen0_stride_17 &&
                !check_c2_samples(
                    z_start, z2, ds0,
                    eta0_ptr, eta0.numel(),
                    e0_qs, e0_qss, e0_qsss, e0_m, e0_c, e0_g,
                        dm0, dc0, dg0, analytic_rate,
                    dof, vel, acc, tau_lim, jerk_lim, rate_lim,
                    tolerance, screen0_stride_65, false)) {
                checked_ptr[j] = checked_local;
                continue;
            }
            if (!check_c2_samples(
                    z_start, z2, ds0,
                    eta0_ptr, eta0.numel(),
                    e0_qs, e0_qss, e0_qsss, e0_m, e0_c, e0_g,
                        dm0, dc0, dg0, analytic_rate,
                    dof, vel, acc, tau_lim, jerk_lim, rate_lim,
                    tolerance, 1, true)) {
                checked_ptr[j] = checked_local;
                continue;
            }
            const double edge1 = c2_time(z_start, z2, ds0);
            if (!std::isfinite(edge1)) {
                checked_ptr[j] = checked_local;
                continue;
            }
            int64_t feasible_local = 1;
            out_first_ptr[j] = edge1;

            const double left_slope = (z2 - z_start) / ds0;
            const int64_t lo = lo_ptr[j];
            const int64_t hi = hi_ptr[j];
            if (lo < 0 || hi < lo || lo >= cols) {
                checked_ptr[j] = checked_local;
                feasible_ptr[j] = feasible_local;
                continue;
            }
            const int64_t hi_exclusive = std::min<int64_t>(cols, hi + 1);
            double best = std::numeric_limits<double>::infinity();
            int64_t best_l = -1;

            for (int64_t l = lo; l < hi_exclusive; ++l) {
                if (!std::isfinite(cost3_ptr[l])) continue;
                if (cost3_ptr[l] >= best) continue;
                const double z3 = z3_ptr[l];
                const double right_slope = slope3_ptr[l];
                if (!(z3 > 0.0) || !std::isfinite(right_slope)) continue;

                double c0, c1, c2, c3;
                if (!c4_coefficients(
                        z2, z3, left_slope, right_slope, ds1,
                        c0, c1, c2, c3)) {
                    continue;
                }
                ++checked_local;

                const double edge2 = c4_time(
                    c0, c1, c2, c3, ds1,
                    qeta, qw, quad_eta.numel()
                );
                if (!std::isfinite(edge2)) continue;
                if (edge2 + cost3_ptr[l] >= best) continue;

                const int64_t screen1_stride_17 =
                    eta1.numel() > 33
                        ? std::max<int64_t>(1, (eta1.numel() - 1) / 16)
                        : 1;
                const int64_t screen1_stride_65 =
                    eta1.numel() > 129
                        ? std::max<int64_t>(1, (eta1.numel() - 1) / 64)
                        : 1;
                if (screen1_stride_17 > 1 &&
                    !check_c4_samples(
                        c0, c1, c2, c3, ds1,
                        eta1_ptr, eta1.numel(),
                        e1_qs, e1_qss, e1_qsss, e1_m, e1_c, e1_g,
                        dm1, dc1, dg1, analytic_rate,
                        dof, vel, acc, tau_lim, jerk_lim, rate_lim,
                        tolerance, screen1_stride_17, false)) {
                    continue;
                }
                if (screen1_stride_65 > 1 &&
                    screen1_stride_65 < screen1_stride_17 &&
                    !check_c4_samples(
                        c0, c1, c2, c3, ds1,
                        eta1_ptr, eta1.numel(),
                        e1_qs, e1_qss, e1_qsss, e1_m, e1_c, e1_g,
                        dm1, dc1, dg1, analytic_rate,
                        dof, vel, acc, tau_lim, jerk_lim, rate_lim,
                        tolerance, screen1_stride_65, false)) {
                    continue;
                }
                if (!check_c4_samples(
                        c0, c1, c2, c3, ds1,
                        eta1_ptr, eta1.numel(),
                        e1_qs, e1_qss, e1_qsss, e1_m, e1_c, e1_g,
                        dm1, dc1, dg1, analytic_rate,
                        dof, vel, acc, tau_lim, jerk_lim, rate_lim,
                        tolerance, 1, true)) {
                    continue;
                }
                ++feasible_local;
                const double total = edge2 + cost3_ptr[l];
                if (total < best) {
                    best = total;
                    best_l = l;
                }
            }

            checked_ptr[j] = checked_local;
            feasible_ptr[j] = feasible_local;
            if (best_l >= 0) {
                out_cost_ptr[j] = best;
                out_succ_ptr[j] = best_l;
                out_slope_ptr[j] = left_slope;
            }
        }
    });

    return {
        out_cost, out_successor, out_slope, out_first_cost, checked, feasible
    };
}


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> terminal_stage(
    torch::Tensor current_z,
    double z_end,
    double ds,
    torch::Tensor eta,
    torch::Tensor edge_qs,
    torch::Tensor edge_qss,
    torch::Tensor edge_qsss,
    torch::Tensor edge_m,
    torch::Tensor edge_c,
    torch::Tensor edge_gravity,
    torch::Tensor velocity,
    torch::Tensor acceleration,
    torch::Tensor torque,
    torch::Tensor jerk,
    torch::Tensor torque_rate,
    double tolerance,
    torch::Tensor edge_dm,
    torch::Tensor edge_dc,
    torch::Tensor edge_dg,
    bool analytic_rate
) {
    if (analytic_rate) {
        require_2d_double_cpu(edge_dm, "edge_dm");
        require_2d_double_cpu(edge_dc, "edge_dc");
        require_2d_double_cpu(edge_dg, "edge_dg");
        TORCH_CHECK(
            edge_dm.sizes() == edge_m.sizes() &&
            edge_dc.sizes() == edge_m.sizes() &&
            edge_dg.sizes() == edge_m.sizes(),
            "analytic rate coefficient tensor shape mismatch"
        );
    }
    require_1d_double_cpu(current_z, "current_z");
    require_1d_double_cpu(eta, "eta");
    require_1d_double_cpu(velocity, "velocity");
    require_1d_double_cpu(acceleration, "acceleration");
    require_1d_double_cpu(torque, "torque");
    require_1d_double_cpu(jerk, "jerk");
    require_1d_double_cpu(torque_rate, "torque_rate");
    require_2d_double_cpu(edge_qs, "edge_qs");
    require_2d_double_cpu(edge_qss, "edge_qss");
    require_2d_double_cpu(edge_qsss, "edge_qsss");
    require_2d_double_cpu(edge_m, "edge_m");
    require_2d_double_cpu(edge_c, "edge_c");
    require_2d_double_cpu(edge_gravity, "edge_gravity");

    TORCH_CHECK(ds > 0.0 && std::isfinite(ds), "ds must be positive");
    const int64_t dof = velocity.numel();
    TORCH_CHECK(dof <= 64, "Kaserer DP3 native kernel supports at most 64 DoF");
    TORCH_CHECK(
        acceleration.numel() == dof && torque.numel() == dof &&
        jerk.numel() == dof && torque_rate.numel() == dof,
        "limit size mismatch"
    );
    TORCH_CHECK(
        edge_qs.size(0) == eta.numel() && edge_qs.size(1) == dof,
        "edge geometry shape mismatch"
    );
    TORCH_CHECK(
        edge_qss.sizes() == edge_qs.sizes() &&
        edge_qsss.sizes() == edge_qs.sizes() &&
        edge_m.sizes() == edge_qs.sizes() &&
        edge_c.sizes() == edge_qs.sizes() &&
        edge_gravity.sizes() == edge_qs.sizes(),
        "edge geometry arrays must have identical shape"
    );

    const int64_t rows = current_z.numel();
    auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto out_cost = torch::full(
        {rows}, std::numeric_limits<double>::infinity(), f64
    );
    auto out_slope = torch::full(
        {rows}, std::numeric_limits<double>::quiet_NaN(), f64
    );
    auto checked = torch::zeros({rows}, i64);
    auto feasible = torch::zeros({rows}, i64);

    const double* z_ptr = current_z.data_ptr<double>();
    const double* eta_ptr = eta.data_ptr<double>();
    const double* eqs = edge_qs.data_ptr<double>();
    const double* eqss = edge_qss.data_ptr<double>();
    const double* eqsss = edge_qsss.data_ptr<double>();
    const double* em = edge_m.data_ptr<double>();
    const double* ec = edge_c.data_ptr<double>();
    const double* eg = edge_gravity.data_ptr<double>();
    const double* vel = velocity.data_ptr<double>();
    const double* acc = acceleration.data_ptr<double>();
    const double* tau_lim = torque.data_ptr<double>();
    const double* jerk_lim = jerk.data_ptr<double>();
    const double* rate_lim = torque_rate.data_ptr<double>();
    const double* dmp = analytic_rate ? edge_dm.data_ptr<double>() : nullptr;
    const double* dcp = analytic_rate ? edge_dc.data_ptr<double>() : nullptr;
    const double* dgp = analytic_rate ? edge_dg.data_ptr<double>() : nullptr;
    double* cost_ptr = out_cost.data_ptr<double>();
    double* slope_ptr = out_slope.data_ptr<double>();
    int64_t* checked_ptr = checked.data_ptr<int64_t>();
    int64_t* feasible_ptr = feasible.data_ptr<int64_t>();

    at::parallel_for(1, rows, 1, [&](int64_t begin, int64_t end) {
        for (int64_t j = begin; j < end; ++j) {
            const double z0 = z_ptr[j];
            if (!(z0 > 0.0) || !std::isfinite(z0)) continue;
            checked_ptr[j] = 1;

            const int64_t stride17 =
                eta.numel() > 33
                    ? std::max<int64_t>(1, (eta.numel() - 1) / 16)
                    : 1;
            if (stride17 > 1 &&
                !check_c2_samples(
                    z0, z_end, ds,
                    eta_ptr, eta.numel(),
                    eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                    dof, vel, acc, tau_lim, jerk_lim, rate_lim,
                    tolerance, stride17, false)) {
                continue;
            }

            const int64_t stride65 =
                eta.numel() > 129
                    ? std::max<int64_t>(1, (eta.numel() - 1) / 64)
                    : 1;
            if (stride65 > 1 && stride65 < stride17 &&
                !check_c2_samples(
                    z0, z_end, ds,
                    eta_ptr, eta.numel(),
                    eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                    dof, vel, acc, tau_lim, jerk_lim, rate_lim,
                    tolerance, stride65, false)) {
                continue;
            }

            if (!check_c2_samples(
                    z0, z_end, ds,
                    eta_ptr, eta.numel(),
                    eqs, eqss, eqsss, em, ec, eg,
                        dmp, dcp, dgp, analytic_rate,
                    dof, vel, acc, tau_lim, jerk_lim, rate_lim,
                    tolerance, 1, true)) {
                continue;
            }

            const double edge = c2_time(z0, z_end, ds);
            if (!std::isfinite(edge)) continue;
            cost_ptr[j] = edge;
            slope_ptr[j] = (z_end - z0) / ds;
            feasible_ptr[j] = 1;
        }
    });

    return {out_cost, out_slope, checked, feasible};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("search_ranges", &search_ranges, "Paper Kaserer DP3 successor ranges");
    m.def("c3_stage", &c3_stage, "Paper Kaserer DP3 C3 stage");
    m.def("terminal_stage", &terminal_stage, "Paper Kaserer DP3 terminal C2 stage");
    m.def("start_stage", &start_stage, "Paper Kaserer DP3 C2/C4 start stage");
}
