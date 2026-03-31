#include "pcd_utils.hpp"

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> estimate_rigid_transform(
    const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    // Compute centroids
    Eigen::Vector3d centroidA = A.colwise().mean();
    Eigen::Vector3d centroidB = B.colwise().mean();

    // Center point clouds
    Eigen::MatrixXd A_centered = A.rowwise() - centroidA.transpose();
    Eigen::MatrixXd B_centered = B.rowwise() - centroidB.transpose();

    // Compute covariance matrix
    Eigen::Matrix3d H = A_centered.transpose() * B_centered;

    // SVD decomposition
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Vector3d singular_values = svd.singularValues();

    // Compute rotation matrix
    Eigen::Matrix3d R = V * U.transpose();

    // Ensure proper rotation (det(R) = 1)
    if (R.determinant() < 0) {
        V.col(2) = -V.col(2);
        R = V * U.transpose();
    }

    // Compute translation
    Eigen::Vector3d t = centroidB - R * centroidA;

    // Compute scaling (optional, set to 1 for rigid transform)
    double scale = 1.0;

    return std::make_tuple(R, t, Eigen::Vector3d(singular_values(0), singular_values(1), singular_values(2)));
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> 
gnc_tls_impl(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, int maxIter = 100, double threshold = 0.4) {
    
    int n = A.rows(); // number of points
    int dim = A.cols(); // dimension (2 or 3)
    
    // Validate dimensions
    if (dim != B.cols()) {
        throw std::invalid_argument("Point clouds A and B must have the same dimension");
    }
    // if (dim != 2 && dim != 3) {
    //     throw std::invalid_argument("Only 2D and 3D are supported");
    // }
    
    double thre_sqrt = threshold * threshold;
    
    // Initialize rotation and translation using dynamic matrices
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(dim, dim);
    Eigen::VectorXd t = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd omg = Eigen::VectorXd::Ones(n);
    double mu = 0.0;

    Eigen::MatrixXd RA = A * R.transpose();
    Eigen::MatrixXd residuals_mat = RA.rowwise() + t.transpose() - B;
    Eigen::VectorXd res = residuals_mat.rowwise().norm();
    Eigen::VectorXd res2 = res.array().square();
    
    for (int i = 0; i < maxIter; ++i) {
        if (i == 0) {
            double max_res = res.maxCoeff();
            mu = thre_sqrt / (2 * max_res * max_res - thre_sqrt);
        } else {
            mu *= 1.4;
            
            // Update weights based on residuals
            for (int j = 0; j < n; ++j) {
                if (res2(j) > (mu + 1) / mu * thre_sqrt) {
                    omg(j) = 0.0;
                } else if (res2(j) < mu / (mu + 1) * thre_sqrt) {
                    omg(j) = 1.0;
                } else {
                    omg(j) = threshold / res(j) * std::sqrt(mu * (mu + 1)) - mu;
                }
            }
        }
        
        // Compute weighted means
        double omg_sum = omg.sum();
        Eigen::VectorXd A_mean = (A.transpose() * omg) / omg_sum;
        Eigen::VectorXd B_mean = (B.transpose() * omg) / omg_sum;
        
        // Center the point clouds
        Eigen::MatrixXd X = A.rowwise() - A_mean.transpose();
        Eigen::MatrixXd Y = B.rowwise() - B_mean.transpose();
        
        // Compute weighted covariance matrix W = X^T * diag(omg) * Y
        Eigen::MatrixXd weighted_Y = Y;
        for (int j = 0; j < n; ++j) {
            weighted_Y.row(j) *= omg(j);
        }
        Eigen::MatrixXd W = X.transpose() * weighted_Y;
        
        // SVD decomposition
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();
        
        R = V * U.transpose();
        
        // Ensure proper rotation (det(R) = 1)
        if (R.determinant() < 0) {
            V.col(dim - 1) = -V.col(dim - 1);
            R = V * U.transpose();
        }
        
        // Update translation
        t = B_mean - R * A_mean;
        
        // Compute residuals: (R*A^T)^T + t - B
        RA = A * R.transpose();
        residuals_mat = RA.rowwise() + t.transpose() - B;
        res = residuals_mat.rowwise().norm();
        res2 = res.array().square();
        
        // Check convergence
        double weighted_res = omg.dot(res2);
        if (weighted_res < 1e-2 || i == maxIter - 1) {
            break;
        }
    }
    
    return std::make_tuple(R, t, omg);
}

py::tuple localization_gnc_3d_with_all_2d_matches(
    const py::array_t<float>& pcd_A,
    const py::array_t<float>& pcd_B,
    const py::array_t<int>& matches_2d_A_lidar,
    const py::array_t<int>& matches_2d_B_lidar,
    const py::array_t<int>& matches_2d_A_intensity,
    const py::array_t<int>& matches_2d_B_intensity,
    int voxel_num = 201,
    float voxel_size = 0.4,
    int search_range = 1,
    int x_max_ind_a = 100,
    int x_max_ind_b = 100,
    int gnc_max_iter = 100,
    float gnc_threshold = 0.4,
    const py::array_t<float>& ground_centroid_A = py::array_t<float>({3}),
    const py::array_t<float>& ground_normal_A = py::array_t<float>({3}),
    const py::array_t<float>& ground_centroid_B = py::array_t<float>({3}),
    const py::array_t<float>& ground_normal_B = py::array_t<float>({3}),
    float gnc_inlier_ratio = 0.5f,
    bool profiling = false
){
    auto time_start = std::chrono::system_clock::now();
    auto [matches_3d_A_lidar, matches_3d_B_lidar] = get_3d_matches_with_2d_matches_impl(pcd_A, pcd_B, matches_2d_A_lidar, matches_2d_B_lidar, voxel_num, voxel_size, search_range, x_max_ind_a, x_max_ind_b, ground_centroid_A, ground_normal_A, ground_centroid_B, ground_normal_B, 0);
    auto [matches_3d_A_intensity, matches_3d_B_intensity] = get_3d_matches_with_2d_matches_impl(pcd_A, pcd_B, matches_2d_A_intensity, matches_2d_B_intensity, voxel_num, voxel_size, search_range, x_max_ind_a, x_max_ind_b, ground_centroid_A, ground_normal_A, ground_centroid_B, ground_normal_B, 1);
    
    Eigen::MatrixXd matches_3d_A(matches_3d_A_lidar.rows() + matches_3d_A_intensity.rows(), 3);
    Eigen::MatrixXd matches_3d_B(matches_3d_B_lidar.rows() + matches_3d_B_intensity.rows(), 3);
    matches_3d_A << matches_3d_A_lidar,
                    matches_3d_A_intensity;
    matches_3d_B << matches_3d_B_lidar,
                    matches_3d_B_intensity;
    
    auto time_match3d = std::chrono::system_clock::now();
    double duration_match = std::chrono::duration_cast<std::chrono::microseconds>(time_match3d - time_start).count() / 1e6;
    if (matches_3d_A.rows() < 3) {
        if (profiling) {
            return py::make_tuple(
                Eigen::Matrix3d::Zero(), // Rotation matrix
                Eigen::Vector3d::Zero(),     // Translation vector
                std::vector<int>(),           // Empty inliers
                std::vector<Eigen::Vector3d>(), // Empty 3D matches A
                std::vector<Eigen::Vector3d>(),  // Empty 3D matches B
                (double)0.0,
                duration_match,               // Duration of matching
                (double)0.0                   // Duration of solving (not applicable here)
            );
        } else {
            return py::make_tuple(
                Eigen::Matrix3d::Zero(), // Rotation matrix
                Eigen::Vector3d::Zero(),     // Translation vector
                std::vector<int>(),           // Empty inliers
                std::vector<Eigen::Vector3d>(), // Empty 3D matches A
                std::vector<Eigen::Vector3d>(),  // Empty 3D matches B
                (double)0.0
            );
        }
    }

    // Eigen::MatrixXd match_points_A(matches_3d_A.size(), 3);
    // Eigen::MatrixXd match_points_B(matches_3d_B.size(), 3);
    // for (size_t i = 0; i < matches_3d_A.size(); ++i) {
    //     match_points_A.row(i) = matches_3d_A[i];
    //     match_points_B.row(i) = matches_3d_B[i];
    // }
    auto [R, t, confidence] = gnc_tls_impl(matches_3d_A, matches_3d_B, gnc_max_iter, gnc_threshold);
    std::vector<int> inliers;
    std::vector<Eigen::Vector3d> inlier_points_A;
    std::vector<Eigen::Vector3d> inlier_points_B;
    for (size_t i = 0; i < matches_3d_A.rows(); ++i) {
        if (confidence[i] > gnc_inlier_ratio) {
            inliers.push_back(i);
            inlier_points_A.push_back(matches_3d_A.row(i));
            inlier_points_B.push_back(matches_3d_B.row(i));
        }
    }
    double inliers_ratio = 0.0;
    if (matches_3d_A.size() > 0){
        inliers_ratio = (double)inliers.size() / (double)matches_3d_A.rows();
    }
    auto time_solve = std::chrono::system_clock::now();
    if (profiling) {
        double duration_solve = std::chrono::duration_cast<std::chrono::microseconds>(time_solve - time_match3d).count() / 1e6;
        return py::make_tuple(R, t, inliers, inlier_points_A, inlier_points_B, inliers_ratio, duration_match, duration_solve);
    }
    return py::make_tuple(R, t, inliers, inlier_points_A, inlier_points_B, inliers_ratio);
    // return py
}