#include "pcd_utils.hpp"


bool is_valid_pcd(const py::array_t<float>& pcd) {
    py::buffer_info pcd_info = pcd.request();
    if (pcd_info.ndim != 2 || pcd_info.shape[1] != 4) {
        return false; // Invalid shape
    }
    if (!(py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_ == (pcd.flags() & py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_))) {
        return false; // Not C-contiguous
    }
    return true; // Valid point cloud
}

py::array_t<float> pcd_to_feat_mat(
    const py::array_t<float>& pcd,
    int voxel_num,
    float voxel_size,
    const py::array_t<float>& ground_centroid,
    const py::array_t<float>& ground_normal) {

    py::buffer_info pcd_info = pcd.request();
    if (!is_valid_pcd(pcd)) {
        throw std::runtime_error("Invalid point cloud data. Expected shape (N, 4) and C-contiguous.");
    }

    // Get ground plane centroid and normal
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    float nx = 0.0f, ny = 0.0f, nz = 1.0f;
    float norm = 1.0f;

    if (ground_centroid.size() >= 3) {
        auto centroid_ptr = ground_centroid.unchecked<1>();
        cx = centroid_ptr(0);
        cy = centroid_ptr(1);
        cz = centroid_ptr(2);
    }
    if (ground_normal.size() >= 3) {
        auto normal_ptr = ground_normal.unchecked<1>();
        nx = normal_ptr(0);
        ny = normal_ptr(1);
        nz = normal_ptr(2);
        norm = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (norm < 1e-6f) {
            norm = 1.0f;  // Avoid division by zero
        }
    }

    // Ensure normal points upward (z > 0)
    if (nz < 0) {
        nx = -nx; ny = -ny; nz = -nz;
    }

    auto feat_mat = py::array_t<float>({voxel_num, voxel_num, 9});
    int point_num = pcd_info.shape[0];
    auto pcd_ptr = pcd.unchecked<2>();
    auto feat_mat_ptr = feat_mat.mutable_unchecked<3>();

    for(int i = 0; i < voxel_num; ++i) {
        for (int j = 0; j < voxel_num; ++j) {
            for (int k = 0; k < 9; ++k) {
                feat_mat_ptr(i, j, k) = -10000; // Initialize feature matrix
            }
            feat_mat_ptr(i, j, 4) = 10000; // Initialize min_height_diff with a large value
        }
    }

    float target_height = 0.5f; // Target height above ground (1m)

    for(int i = 0; i < point_num; ++i) {
        float x = pcd_ptr(i, 0);
        float y = pcd_ptr(i, 1);
        float z = pcd_ptr(i, 2);
        float intensity = pcd_ptr(i, 3);

        // Signed distance to ground plane: n · (p - c) / ||n||
        // Positive = above ground, Negative = below ground
        float signed_dist = (nx*(x - cx) + ny*(y - cy) + nz*(z - cz)) / norm;

        int xcs = voxel_num / 2 - x / voxel_size;
        int ycs = voxel_num / 2 - y / voxel_size;
        if (xcs < 0 || xcs >= voxel_num || ycs < 0 || ycs >= voxel_num)
            continue; // Skip points outside the voxel grid

        // Only consider points above ground (signed_dist > 0)
        // Select point closest to target height (1m above ground)
        // if (signed_dist > 0) {
        float height_diff = std::abs(signed_dist - target_height);
        if (height_diff < feat_mat_ptr(xcs, ycs, 4)) {
            feat_mat_ptr(xcs, ycs, 0) = x;
            feat_mat_ptr(xcs, ycs, 1) = y;
            feat_mat_ptr(xcs, ycs, 2) = z;
            feat_mat_ptr(xcs, ycs, 3) = intensity;
            feat_mat_ptr(xcs, ycs, 4) = height_diff; // Store height difference
        }
        // }
        if (intensity > feat_mat_ptr(xcs, ycs, 8)) {
            feat_mat_ptr(xcs, ycs, 5) = x;
            feat_mat_ptr(xcs, ycs, 6) = y;
            feat_mat_ptr(xcs, ycs, 7) = z;
            feat_mat_ptr(xcs, ycs, 8) = intensity; // Update intensity
        }
    }
    return feat_mat;
}

py::tuple get_points_in_feat_mat(const py::array_t<float>& feat_mat, int x, int y, int range){
    py::buffer_info feat_mat_info = feat_mat.request();
    if (feat_mat_info.ndim != 3 || feat_mat_info.shape[2] != 9) {
        throw std::runtime_error("Invalid feature matrix shape. Expected shape (N, N, 9).");
    }

    auto feat_mat_ptr = feat_mat.unchecked<3>();
    Eigen::Vector3d target_height_point;
    Eigen::Vector3d max_intensity_point;
    float min_height_diff = 10000.0f; // Initialize with a large value
    float max_intensity = -10000.0f; // Initialize with a small value
    for (int i = -range; i <= range; ++i) {
        for (int j = -range; j <= range; ++j) {
            int x_idx = x + i;
            int y_idx = y + j;
            if (x_idx < 0 || x_idx >= feat_mat_info.shape[0] || y_idx < 0 || y_idx >= feat_mat_info.shape[1]) {
                continue; // Skip out-of-bounds indices
            }
            // Find point closest to target height (min height_diff, stored in feat_mat[4])
            if (feat_mat_ptr(x_idx, y_idx, 4) < min_height_diff) {
                min_height_diff = feat_mat_ptr(x_idx, y_idx, 4);
                target_height_point = {feat_mat_ptr(x_idx, y_idx, 0),
                                       feat_mat_ptr(x_idx, y_idx, 1),
                                       feat_mat_ptr(x_idx, y_idx, 2)};
            }
            if (feat_mat_ptr(x_idx, y_idx, 8) > max_intensity) {
                max_intensity = feat_mat_ptr(x_idx, y_idx, 8);
                max_intensity_point = {feat_mat_ptr(x_idx, y_idx, 5),
                                       feat_mat_ptr(x_idx, y_idx, 6),
                                       feat_mat_ptr(x_idx, y_idx, 7)};
            }
        }
    }
    return  py::make_tuple(
        target_height_point,
        max_intensity_point
    );
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> get_3d_matches_with_2d_matches_impl(
    const py::array_t<float>& pcd_A,
    const py::array_t<float>& pcd_B,
    const py::array_t<int>& matches_2d_A,
    const py::array_t<int>& matches_2d_B,
    int voxel_num = 201,
    float voxel_size = 0.4,
    int search_range = 1,
    int x_max_ind_a = 100,
    int x_max_ind_b = 100,
    const py::array_t<float>& ground_centroid_A = py::array_t<float>({3}),
    const py::array_t<float>& ground_normal_A = py::array_t<float>({3}),
    const py::array_t<float>& ground_centroid_B = py::array_t<float>({3}),
    const py::array_t<float>& ground_normal_B = py::array_t<float>({3}),
    int point_option = 0   // 0: min dist_to_ground, 1: max intensity
) {
    auto feat_mat_A = pcd_to_feat_mat(pcd_A, voxel_num, voxel_size, ground_centroid_A, ground_normal_A);
    auto feat_mat_B = pcd_to_feat_mat(pcd_B, voxel_num, voxel_size, ground_centroid_B, ground_normal_B);

    auto match_2d_A_ptr = matches_2d_A.unchecked<2>();
    auto match_2d_B_ptr = matches_2d_B.unchecked<2>();

    std::vector<Eigen::Vector3d> matches_3d_A;
    std::vector<Eigen::Vector3d> matches_3d_B;
    float coord_max = voxel_num * voxel_size / 2.0f;
    for(int i = 0; i < matches_2d_A.shape(0); ++i) {
        int A_x = match_2d_A_ptr(i, 0);
        int A_y = match_2d_A_ptr(i, 1);
        int B_x = match_2d_B_ptr(i, 0);
        int B_y = match_2d_B_ptr(i, 1);
        if (A_x < 0 || A_x >= voxel_num || A_y < 0 || A_y >= voxel_num ||
            B_x < 0 || B_x >= voxel_num || B_y < 0 || B_y >= voxel_num) {
            throw std::runtime_error("Match indices out of bounds in feature matrix.");
        }
        auto A_points = get_points_in_feat_mat(feat_mat_A, A_x, A_y, search_range);
        auto B_points = get_points_in_feat_mat(feat_mat_B, B_x, B_y, search_range);
        auto A_point_chs = A_points[point_option].cast<Eigen::Vector3d>();
        auto B_point_chs = B_points[point_option].cast<Eigen::Vector3d>();
        if (std::abs(A_point_chs[0]) > coord_max || std::abs(A_point_chs[1]) > coord_max ||
            std::abs(B_point_chs[0]) > coord_max || std::abs(B_point_chs[1]) > coord_max) {
            continue; // Skip points outside the coordinate range
        }
        A_point_chs[0] += (x_max_ind_a - 100) * voxel_size;
        B_point_chs[0] += (x_max_ind_b - 100) * voxel_size;
        matches_3d_A.push_back(A_point_chs);
        matches_3d_B.push_back(B_point_chs);
    }
    // convert to Eigen matrices
    Eigen::MatrixXd match_points_A(matches_3d_A.size(), 3);
    Eigen::MatrixXd match_points_B(matches_3d_B.size(), 3);
    for (size_t i = 0; i < matches_3d_A.size(); ++i) {
        match_points_A.row(i) = matches_3d_A[i];
        match_points_B.row(i) = matches_3d_B[i];
    }
    return std::make_tuple(match_points_A, match_points_B);
}

py::tuple get_3d_matches_with_2d_matches(
    const py::array_t<float>& pcd_A,
    const py::array_t<float>& pcd_B,
    const py::array_t<int>& matches_2d_A,
    const py::array_t<int>& matches_2d_B,
    int voxel_num = 201,
    float voxel_size = 0.4,
    int search_range = 1,
    int x_max_ind_a = 100,
    int x_max_ind_b = 100,
    const py::array_t<float>& ground_centroid_A = py::array_t<float>({3}),
    const py::array_t<float>& ground_normal_A = py::array_t<float>({3}),
    const py::array_t<float>& ground_centroid_B = py::array_t<float>({3}),
    const py::array_t<float>& ground_normal_B = py::array_t<float>({3}),
    int point_option = 0
) {
    auto [matches_3d_A, matches_3d_B] = get_3d_matches_with_2d_matches_impl(
        pcd_A, pcd_B, matches_2d_A, matches_2d_B,
        voxel_num, voxel_size, search_range, x_max_ind_a, x_max_ind_b,
        ground_centroid_A, ground_normal_A, ground_centroid_B, ground_normal_B, point_option);
    return py::make_tuple(matches_3d_A, matches_3d_B);
}

std::pair<py::array_t<uint8_t>, py::array_t<uint8_t>> getBEV(
    const Eigen::Ref<const RowMatrixXf>& all_points, // Input is directly an Eigen reference
    float pc_range = 40,
    float res = 0.4f,
    int blur_size = 25,
    int median_size = 3,
    float high_intensity_threshold = 0.995f
) {
    // Check input dimensions
    if (all_points.cols() < 3) {
        throw std::runtime_error("Input point cloud must have at least 3 columns (x, y, z, [intensity])");
    }

    int num_points = all_points.rows();
    bool has_intensity = all_points.cols() >= 4;

    // 1. Initialize grid parameters
    float x_min = -pc_range;
    float y_min = -pc_range;
    float x_max = pc_range;
    float y_max = pc_range;

    int x_min_ind = static_cast<int>(std::floor(x_min / res));
    int x_max_ind = static_cast<int>(std::floor(x_max / res));
    int y_min_ind = static_cast<int>(std::floor(y_min / res));
    int y_max_ind = static_cast<int>(std::floor(y_max / res));

    int x_num = x_max_ind - x_min_ind + 1;
    int y_num = y_max_ind - y_min_ind + 1;

    // 2. Prepare images (using OpenCV Mat)
    // Corresponding Python: np.zeros((y_num, x_num), dtype=np.uint8)
    cv::Mat mat_global_image = cv::Mat::zeros(y_num, x_num, CV_8UC1);
    cv::Mat mat_intensity_max = cv::Mat::zeros(y_num, x_num, CV_32FC1);

    // Auxiliary matrices for Mean calculation
    cv::Mat mat_intensity_sum = cv::Mat::zeros(y_num, x_num, CV_32FC1);
    cv::Mat mat_intensity_count = cv::Mat::zeros(y_num, x_num, CV_32FC1);

    // 3. Fill grid (core loop)
    // Directly access Eigen data
    for (int i = 0; i < num_points; ++i) {
        float px = all_points(i, 0);
        float py = all_points(i, 1);

        // Coordinate mapping logic (keep consistent with Python: x_ind uses p[1], y_ind uses p[0])
        int x_ind = x_max_ind - static_cast<int>(std::floor(py / res));
        int y_ind = y_max_ind - static_cast<int>(std::floor(px / res));

        // Boundary check
        if (x_ind < 0 || x_ind >= x_num || y_ind < 0 || y_ind >= y_num) continue;

        // Global Image count
        uint8_t& global_val = mat_global_image.at<uint8_t>(y_ind, x_ind);
        if (global_val < 10) {
            global_val += 1;
        }

        if (has_intensity) {
            // float intensity = all_points(i, 3);
            float intensity = std::min(all_points(i, 3), 255.0f);
            // uint8_t u_intensity = static_cast<uint8_t>(intensity); // Assume Python side guarantees range or truncate here

            // Max Intensity
            float& max_val = mat_intensity_max.at<float>(y_ind, x_ind);
            if (intensity > max_val) {
                max_val = intensity;
            }

            // Mean Accumulation
            mat_intensity_sum.at<float>(y_ind, x_ind) += intensity;
            mat_intensity_count.at<float>(y_ind, x_ind) += 1.0f;
        }
    }

    cv::Mat mat_intensity_mean = mat_intensity_sum / (mat_intensity_count + 1e-6f); // Avoid division by zero
    
    
    // cv::Mat valid_mask = mat_intensity_count > 0;
    // mat_intensity_sum.copyTo(mat_intensity_mean, valid_mask);
    // cv::divide(mat_intensity_mean, mat_intensity_count, mat_intensity_mean);
    

    // Blur (25, 25)
    cv::blur(mat_intensity_mean, mat_intensity_mean, cv::Size(blur_size, blur_size));

    // Calculate Ratio: Max / (Mean + 1)
    // cv::Mat mat_intensity_image;
    // cv::Mat mat_max_float;
    // mat_intensity_max.convertTo(mat_max_float, CV_32F);
    // cv::divide(mat_max_float, mat_intensity_mean + 1.0f, mat_intensity_image);
    cv::Mat mat_intensity_image = mat_intensity_max / (mat_intensity_mean + 1.0f);

    std::vector<float> valid_pixels;
    valid_pixels.reserve(y_num * x_num / 10); 
    
    if (mat_intensity_image.isContinuous()) {
        float* data_ptr = (float*)mat_intensity_image.datastart;
        int total_elements = mat_intensity_image.rows * mat_intensity_image.cols;
        for (int p = 0; p < total_elements; ++p) {
            if (data_ptr[p] > 0.01f) { // Remove blank background
                valid_pixels.push_back(data_ptr[p]);
            }
        }
    } else {
        for (int r = 0; r < mat_intensity_image.rows; ++r) {
            for (int c = 0; c < mat_intensity_image.cols; ++c) {
                float val = mat_intensity_image.at<float>(r, c);
                if (val > 0.01f) valid_pixels.push_back(val);
            }
        }
    }
    
    float threshold = 0.0f;
    if (!valid_pixels.empty()) {
        size_t target_idx = static_cast<size_t>(valid_pixels.size() * high_intensity_threshold);
        target_idx = std::min(target_idx, valid_pixels.size() - 1); // Safety check
        std::nth_element(valid_pixels.begin(), valid_pixels.begin() + target_idx, valid_pixels.end());
        threshold = valid_pixels[target_idx];
    }

    // // Find 99.5% percentile (Percentile)
    // std::vector<float> pixels;
    // pixels.reserve(y_num * x_num);
    // if (mat_intensity_image.isContinuous()) {
    //     pixels.assign((float*)mat_intensity_image.datastart, (float*)mat_intensity_image.dataend);
    // } else {
    //     for (int r = 0; r < mat_intensity_image.rows; ++r) {
    //         for (int c = 0; c < mat_intensity_image.cols; ++c) {
    //             pixels.push_back(mat_intensity_image.at<float>(r, c));
    //         }
    //     }
    // }
    
    // float threshold = 0.0f;
    // if (!pixels.empty()) {
    //     // Use nth_element to find Top 0.5% (i.e., 99.5% position)
    //     size_t target_idx = static_cast<size_t>(pixels.size() * high_intensity_threshold);
    //     std::nth_element(pixels.begin(), pixels.begin() + target_idx, pixels.end());
    //     threshold = pixels[target_idx];
    // }

    // Mask and median filtering
    cv::Mat high_light_mask = mat_intensity_image > threshold;
    cv::Mat median_img;
    cv::medianBlur(mat_intensity_image, median_img, median_size);
    median_img.copyTo(mat_intensity_image, high_light_mask); // Replace highlight regions

    // Intensity normalization: (val - min) / (max - min) * 255
    double min_val, max_val;
    cv::minMaxLoc(mat_intensity_image, &min_val, &max_val);
    if (max_val > min_val) {
        mat_intensity_image = (mat_intensity_image - min_val) / (max_val - min_val) * 255.0f;
    } else {
        mat_intensity_image = cv::Mat::zeros(mat_intensity_image.size(), CV_32F);
    }

    // 5. Post-processing - Global Image
    // Python: mat_global_image[mat_global_image<=1] = 0
    cv::threshold(mat_global_image, mat_global_image, 1, 0, cv::THRESH_TOZERO);

    // Python: mat_global_image = mat_global_image * 10
    // Note: This may overflow uint8, but Python code may allow overflow or max 255
    // For safety, convert to float for normalization
    cv::Mat global_float;
    mat_global_image.convertTo(global_float, CV_32F);
    global_float = global_float * 10.0f;

    // Python: mat_global_image[np.where(mat_global_image>255)]=255
    // Python: mat_global_image = mat_global_image/np.max(mat_global_image)*255
    cv::minMaxLoc(global_float, &min_val, &max_val);
    
    if (max_val > 0) {
        global_float = global_float / max_val * 255.0f;
    }

    // 6. Convert back to uint8 and output
    cv::Mat result_global, result_intensity;
    global_float.convertTo(result_global, CV_8U);
    mat_intensity_image.convertTo(result_intensity, CV_8U);

    return std::make_pair(
        py::array_t<uint8_t>({result_global.rows, result_global.cols}, result_global.data),
        py::array_t<uint8_t>({result_intensity.rows, result_intensity.cols}, result_intensity.data)
    );
}