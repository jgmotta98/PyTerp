#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <omp.h>

#include "nanoflann.hpp"

namespace py = pybind11;

struct NumpyAdaptor {
    const py::detail::unchecked_reference<double, 2> &data;
    const size_t n_rows;

    NumpyAdaptor(const py::detail::unchecked_reference<double, 2> &d) : data(d), n_rows(d.shape(0)) {}

    inline size_t kdtree_get_point_count() const { return n_rows; }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return data(idx, dim); }

    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};


py::array_t<double> cpp_interpolate(
    py::array_t<double, py::array::c_style | py::array::forcecast> source_pts,
    py::array_t<double, py::array::c_style | py::array::forcecast> source_vals,
    py::array_t<double, py::array::c_style | py::array::forcecast> target_pts,
    int k_neighbors,
    double power)
{
    if (source_pts.ndim() != 2 || source_pts.shape(1) != 3)
        throw std::runtime_error("Source points must be an Nx3 array.");
    if (target_pts.ndim() != 2 || target_pts.shape(1) != 3)
        throw std::runtime_error("Target points must be an Mx3 array.");
    if (source_vals.ndim() != 1 || source_vals.shape(0) != source_pts.shape(0))
        throw std::runtime_error("Source values must be a 1D array matching the number of source points.");

    auto source_pts_ptr = source_pts.unchecked<2>();
    auto source_vals_ptr = source_vals.unchecked<1>();
    auto target_pts_ptr = target_pts.unchecked<2>();

    const size_t n_sources = source_pts.shape(0);
    const size_t n_targets = target_pts.shape(0);

    NumpyAdaptor source_cloud(source_pts_ptr);
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, NumpyAdaptor>,
        NumpyAdaptor,
        3
    > kd_tree_t;

    kd_tree_t index(3, source_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    py::array_t<double> interpolated_vals(n_targets);
    auto interpolated_vals_ptr = interpolated_vals.mutable_unchecked<1>();

    #pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < n_targets; ++i) {
        double query_pt[3] = {target_pts_ptr(i, 0), target_pts_ptr(i, 1), target_pts_ptr(i, 2)};

        std::vector<unsigned int> ret_index(k_neighbors);
        std::vector<double> out_dist_sqr(k_neighbors);
        index.knnSearch(&query_pt[0], k_neighbors, &ret_index[0], &out_dist_sqr[0]);

        double total_weight = 0.0;
        double weighted_sum = 0.0;

        for (size_t j = 0; j < k_neighbors; ++j) {
            const unsigned int neighbor_idx = ret_index[j];
            const double dist = std::sqrt(out_dist_sqr[j]);

            if (dist < 1e-9) {
                weighted_sum = source_vals_ptr(neighbor_idx);
                total_weight = 1.0;
                break;
            }

            const double weight = 1.0 / std::pow(dist, power);
            weighted_sum += source_vals_ptr(neighbor_idx) * weight;
            total_weight += weight;
        }

        if (total_weight > 1e-9) {
            interpolated_vals_ptr(i) = weighted_sum / total_weight;
        } else {
            interpolated_vals_ptr(i) = 0.0;
        }
    }

    return interpolated_vals;
}

py::array_t<double> cpp_classify(
    py::array_t<double, py::array::c_style | py::array::forcecast> source_pts,
    py::array_t<double, py::array::c_style | py::array::forcecast> source_vals,
    py::array_t<double, py::array::c_style | py::array::forcecast> target_pts,
    int k_neighbors,
    double power)
{
    if (source_pts.ndim() != 2 || source_pts.shape(1) != 3)
        throw std::runtime_error("Source points must be an Nx3 array.");
    if (target_pts.ndim() != 2 || target_pts.shape(1) != 3)
        throw std::runtime_error("Target points must be an Mx3 array.");
    if (source_vals.ndim() != 1 || source_vals.shape(0) != source_pts.shape(0))
        throw std::runtime_error("Source values must be a 1D array matching the number of source points.");

    auto source_pts_ptr = source_pts.unchecked<2>();
    auto source_vals_ptr = source_vals.unchecked<1>();
    auto target_pts_ptr = target_pts.unchecked<2>();
    
    const size_t n_targets = target_pts.shape(0);

    NumpyAdaptor source_cloud(source_pts_ptr);
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, NumpyAdaptor>,
        NumpyAdaptor,
        3
    > kd_tree_t;

    kd_tree_t index(3, source_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    py::array_t<double> classified_vals(n_targets);
    auto classified_vals_ptr = classified_vals.mutable_unchecked<1>();

    #pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < n_targets; ++i) {
        double query_pt[3] = {target_pts_ptr(i, 0), target_pts_ptr(i, 1), target_pts_ptr(i, 2)};

        std::vector<unsigned int> ret_index(k_neighbors);
        std::vector<double> out_dist_sqr(k_neighbors);
        index.knnSearch(&query_pt[0], k_neighbors, &ret_index[0], &out_dist_sqr[0]);

        double weight_class_1 = 0.0;
        double weight_class_0 = 0.0;
        
        bool exact_match = false;
        double exact_match_value = 0.0;

        for (size_t j = 0; j < k_neighbors; ++j) {
            const unsigned int neighbor_idx = ret_index[j];
            const double dist = std::sqrt(out_dist_sqr[j]);

            if (dist < 1e-9) {
                exact_match_value = source_vals_ptr(neighbor_idx);
                exact_match = true;
                break;
            }

            const double weight = 1.0 / std::pow(dist, power);
            
            if (source_vals_ptr(neighbor_idx) > 0.5) {
                weight_class_1 += weight;
            } else {
                weight_class_0 += weight;
            }
        }

        if (exact_match) {
            classified_vals_ptr(i) = (exact_match_value > 0.5) ? 1.0 : 0.0;
        } else {
            classified_vals_ptr(i) = (weight_class_1 > weight_class_0) ? 1.0 : 0.0;
        }
    }

    return classified_vals;
}

PYBIND11_MODULE(pyterp, m) {
    m.doc() = "A high-performance parallel interpolator using C++, OpenMP, and pybind11";

    m.def("interpolate", &cpp_interpolate, "Interpolates scattered 3D data onto target points using parallel k-NN IDW",
          py::arg("source_points"),
          py::arg("source_values"),
          py::arg("target_points"),
          py::arg("k_neighbors") = 8,
          py::arg("power") = 2.0);

    m.def("classify", &cpp_classify, "Classifies scattered 3D data using parallel weighted k-NN. Returns binary values (0.0 or 1.0).",
          py::arg("source_points"),
          py::arg("source_values"),
          py::arg("target_points"),
          py::arg("k_neighbors") = 8,
          py::arg("power") = 2.0);
}