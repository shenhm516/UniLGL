#ifndef PCD_UTILS_HPP
#define PCD_UTILS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <random>

namespace py = pybind11;
// Define a RowMajor matrix type to match Numpy's default memory layout, avoiding copies
using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

bool is_valid_pcd(const py::array_t<float>& pcd);
py::array_t<float> pcd_to_feat_mat(const py::array_t<float>& pcd, int voxel_num, float voxel_size, const py::array_t<float>& ground_centroid, const py::array_t<float>& ground_normal);
py::tuple get_points_in_feat_mat(const py::array_t<float>& feat_mat, int x, int y, int range);
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> get_3d_matches_with_2d_matches_impl(
    const py::array_t<float>& pcd_A,
    const py::array_t<float>& pcd_B,
    const py::array_t<int>& matches_2d_A,
    const py::array_t<int>& matches_2d_B,
    int voxel_num,
    float voxel_size,
    int search_range,
    int x_max_ind_a,
    int x_max_ind_b,
    const py::array_t<float>& ground_centroid_A,
    const py::array_t<float>& ground_normal_A,
    const py::array_t<float>& ground_centroid_B,
    const py::array_t<float>& ground_normal_B,
    int point_option
);
py::tuple get_3d_matches_with_2d_matches(
    const py::array_t<float>& pcd_A,
    const py::array_t<float>& pcd_B,
    const py::array_t<int>& matches_2d_A,
    const py::array_t<int>& matches_2d_B,
    int voxel_num,
    float voxel_size,
    int search_range,
    int x_max_ind_a,
    int x_max_ind_b,
    const py::array_t<float>& ground_centroid_A,
    const py::array_t<float>& ground_normal_A,
    const py::array_t<float>& ground_centroid_B,
    const py::array_t<float>& ground_normal_B,
    int point_option
);
std::pair<py::array_t<uint8_t>, py::array_t<uint8_t>> getBEV(
    const Eigen::Ref<const RowMatrixXf>& all_points,
    float pc_range, 
    float res, 
    int blur_size, 
    int median_size, 
    float high_intensity_threshold
);



std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> estimate_rigid_transform(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);



std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> gnc_tls_impl(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, int maxIter, double threshold);
py::tuple localization_gnc_3d_with_all_2d_matches(
    const py::array_t<float>& pcd_A,
    const py::array_t<float>& pcd_B,
    const py::array_t<int>& matches_2d_A_lidar,
    const py::array_t<int>& matches_2d_B_lidar,
    const py::array_t<int>& matches_2d_A_intensity,
    const py::array_t<int>& matches_2d_B_intensity,
    int voxel_num,
    float voxel_size,
    int search_range,
    int x_max_ind_a,
    int x_max_ind_b,
    int gnc_max_iter,
    float gnc_threshold,
    const py::array_t<float>& ground_centroid_A,
    const py::array_t<float>& ground_normal_A,
    const py::array_t<float>& ground_centroid_B,
    const py::array_t<float>& ground_normal_B,
    float gnc_inlier_ratio,
    bool profiling
);


RowMatrixXf voxel_filter(Eigen::Ref<const RowMatrixXf>& points, float leaf_size);
#endif // PCD_UTILS_HPP