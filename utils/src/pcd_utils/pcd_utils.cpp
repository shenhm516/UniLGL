#include "pcd_utils.hpp"




py::tuple filter_matches(
    const py::array_t<float>& faiss_D, 
    const py::array_t<int>& faiss_I,
    const py::tuple& query_indices, 
    const py::tuple& db_indices,
    float threshold,
    bool ratio_filter = false,
    float ratio_filter_threshold = 0.8f,
    bool filter_gnc = false,
    float filter_gnc_threshold = 1.0f
) {
    py::buffer_info D_info = faiss_D.request();
    py::buffer_info I_info = faiss_I.request();

    if (D_info.ndim != 2 || I_info.ndim != 2 || D_info.shape[0] != I_info.shape[0]) {
        throw std::runtime_error("Invalid shape for distance or index arrays.");
    }
    auto D_ptr = faiss_D.unchecked<2>();
    auto I_ptr = faiss_I.unchecked<2>();

    auto query_x_indices = query_indices[0].cast<py::array_t<int64_t>>();
    auto query_y_indices = query_indices[1].cast<py::array_t<int64_t>>();
    auto db_x_indices = db_indices[0].cast<py::array_t<int64_t>>();
    auto db_y_indices = db_indices[1].cast<py::array_t<int64_t>>();

    auto query_x_ptr = query_x_indices.unchecked<1>();
    auto query_y_ptr = query_y_indices.unchecked<1>();
    auto db_x_ptr = db_x_indices.unchecked<1>();
    auto db_y_ptr = db_y_indices.unchecked<1>();

    std::vector<Eigen::Vector2i> query_match_indices;
    std::vector<Eigen::Vector2i> db_match_indices;
    for(int query_local_idx = 0; query_local_idx < I_info.shape[0]; ++query_local_idx) {
        int db_local_match_idx = I_ptr(query_local_idx, 0);
        if (D_ptr(query_local_idx, 0) > threshold) {
            continue; // Skip matches with distance greater than threshold
        }
        if (ratio_filter && I_info.shape[1] > 1) {
            float ratio = D_ptr(query_local_idx, 0) / D_ptr(query_local_idx, 1);
            if (ratio > ratio_filter_threshold) {
                continue; // Skip matches that do not satisfy the ratio test
            }
        }
        Eigen::Vector2i query_point(
            query_x_ptr(query_local_idx),
            query_y_ptr(query_local_idx)
        );
        Eigen::Vector2i db_point(
            db_x_ptr(db_local_match_idx),
            db_y_ptr(db_local_match_idx)
        );
        query_match_indices.push_back(query_point);
        db_match_indices.push_back(db_point);
    }
    
    if (filter_gnc){
        Eigen::MatrixXd match_mat_A(query_match_indices.size(), 2);
        Eigen::MatrixXd match_mat_B(db_match_indices.size(), 2);
        for (size_t i = 0; i < query_match_indices.size(); ++i) {
            match_mat_A.row(i) = query_match_indices[i].cast<double>();
            match_mat_B.row(i) = db_match_indices[i].cast<double>();
        }
        auto [R, t, weights] = gnc_tls_impl(match_mat_A, match_mat_B, 100, filter_gnc_threshold);
        query_match_indices.clear();
        db_match_indices.clear();
        for (size_t i = 0; i < weights.size(); ++i) {
            if (weights(i) > 0.5f) { // Keep matches with weight greater than 0.5
                query_match_indices.push_back(match_mat_A.row(i).cast<int>());
                db_match_indices.push_back(match_mat_B.row(i).cast<int>());
            }
        }
    }


    auto query_match_array = py::array_t<int64_t>({(int)query_match_indices.size(),2});
    auto db_match_array = py::array_t<int64_t>({(int)db_match_indices.size(),2});
    auto query_match_ptr = query_match_array.mutable_unchecked<2>();
    auto db_match_ptr = db_match_array.mutable_unchecked<2>();
    for (size_t i = 0; i < query_match_indices.size(); ++i) {
        query_match_ptr(i, 0) = query_match_indices[i](0);
        query_match_ptr(i, 1) = query_match_indices[i](1);
        db_match_ptr(i, 0) = db_match_indices[i](0);
        db_match_ptr(i, 1) = db_match_indices[i](1);
    }
    return py::make_tuple(query_match_array, db_match_array);
}

py::tuple precision_recall_curve(const py::array_t<int>& labels, const py::array_t<float>& scores, int downsample = 1) {
    auto scores_info = scores.request();
    auto labels_info = labels.request();
    if (scores_info.ndim != 1 || labels_info.ndim != 1) {
        throw std::runtime_error("Scores and labels must be 1D arrays.");
    }
    if (scores_info.shape[0] != labels_info.shape[0]) {
        throw std::runtime_error("Scores and labels must have the same length.");
    }
    auto scores_ptr = scores.unchecked<1>();
    auto labels_ptr = labels.unchecked<1>();

    // Sort scores and corresponding labels
    std::vector<std::pair<float, int>> score_label_pairs;
    int total_positive = 0;
    for (size_t i = 0; i < scores_info.shape[0]; i += downsample) {
        score_label_pairs.emplace_back(std::make_pair(scores_ptr(i), labels_ptr(i)));
        if (labels_ptr(i) == 1) {
            total_positive++;
        }
        // std::cout << "Score: " << scores_ptr(i) << ", Label: " << labels_ptr(i) << std::endl;
    }
    // printf("Total positive labels: %d\n", total_positive);
    std::sort(score_label_pairs.begin(), score_label_pairs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });

    auto precision_array = py::array_t<float>(score_label_pairs.size());
    auto recall_array = py::array_t<float>(score_label_pairs.size());
    auto threshold_array = py::array_t<float>(score_label_pairs.size());
    auto precision_ptr = precision_array.mutable_unchecked<1>();
    auto recall_ptr = recall_array.mutable_unchecked<1>();
    auto thresholds_ptr = threshold_array.mutable_unchecked<1>();

    int true_positive = 0;
    int false_positive = 0;

    for (size_t i = 0; i < score_label_pairs.size(); ++i) {
        float score = score_label_pairs[i].first;
        int label = score_label_pairs[i].second;

        if (label == 1) {
            true_positive++;
        } else {
            false_positive++;
        }

        double precision = 1.0f; // Default precision if no positive predictions
        if (true_positive + false_positive > 0) {
            precision = true_positive / (double)(true_positive + false_positive);
        }
        double recall = true_positive / (double)total_positive;

        precision_ptr(i) = (float)precision;
        recall_ptr(i) = (float)recall;
        thresholds_ptr(i) = score;
        // printf("Score: %.4f, Precision: %.4f, Recall: %.4f\n", score, precision, recall);
        // exit(0);
    }

    return py::make_tuple(precision_array, recall_array, threshold_array);
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "Point Cloud Data Utilities";
    m.def("pcd_to_feat_mat", &pcd_to_feat_mat, "Convert point cloud data to feature matrix",
          py::arg("pcd"), py::arg("voxel_num"), py::arg("voxel_size"), py::arg("ground_centroid"), py::arg("ground_normal"));
    m.def("get_points_in_feat_mat", &get_points_in_feat_mat, "Get points in feature matrix",
          py::arg("feat_mat"), py::arg("x"), py::arg("y"), py::arg("range"));
    m.def("estimate_rigid_transform", &estimate_rigid_transform, "Estimate rigid transformation between two point clouds",
          py::arg("A"), py::arg("B"));
    m.def("get_3d_matches_with_2d_matches", &get_3d_matches_with_2d_matches, "Get 3D registration matches from 2D matches",
          py::arg("pcd_A"), py::arg("pcd_B"),
          py::arg("matches_2d_A"), py::arg("matches_2d_B"),
          py::arg("voxel_num") = 201,
          py::arg("voxel_size") = 0.4,
          py::arg("search_range") = 1,
          py::arg("x_max_ind_a") = 100,
          py::arg("x_max_ind_b") = 100,
          py::arg("ground_centroid_A") = py::array_t<float>({3}),
          py::arg("ground_normal_A") = py::array_t<float>({3}),
          py::arg("ground_centroid_B") = py::array_t<float>({3}),
          py::arg("ground_normal_B") = py::array_t<float>({3}),
          py::arg("point_option") = 0);
    m.def("localization_gnc_3d_with_all_2d_matches", &localization_gnc_3d_with_all_2d_matches, "Localize 3D point cloud with all 2D matches using GNC",
          py::arg("pcd_A"), py::arg("pcd_B"),
          py::arg("matches_2d_A_lidar"), py::arg("matches_2d_B_lidar"),
          py::arg("matches_2d_A_intensity"), py::arg("matches_2d_B_intensity"),
          py::arg("voxel_num") = 201,
          py::arg("voxel_size") = 0.4,
          py::arg("search_range") = 1,
          py::arg("x_max_ind_a") = 100,
          py::arg("x_max_ind_b") = 100,
          py::arg("gnc_max_iter") = 100,
          py::arg("gnc_threshold") = 0.4,
          py::arg("ground_centroid_A") = py::array_t<float>({3}),
          py::arg("ground_normal_A") = py::array_t<float>({3}),
          py::arg("ground_centroid_B") = py::array_t<float>({3}),
          py::arg("ground_normal_B") = py::array_t<float>({3}),
          py::arg("gnc_inlier_ratio") = 0.5f,
          py::arg("profiling") = false);
    m.def("filter_matches", &filter_matches, "Filter matches based on distance threshold",
          py::arg("faiss_D"), 
          py::arg("faiss_I"),
          py::arg("query_indices"), 
          py::arg("db_indices"),
          py::arg("threshold") = 0.4,
          py::arg("ratio_filter") = false,
          py::arg("ratio_filter_threshold") = 0.8f,
          py::arg("filter_gnc") = false,
          py::arg("filter_gnc_threshold") = 0.4f);
    m.def("precision_recall_curve", &precision_recall_curve, "Compute precision-recall curve",
          py::arg("labels"), py::arg("scores"), py::arg("downsample") = 1);
    m.def("getBEV", &getBEV, "Generate BEV maps from pointcloud",
          py::arg("all_points"), 
          py::arg("pc_range") = 40.0f, 
          py::arg("res") = 0.4f, 
          py::arg("blur_size") = 25, 
          py::arg("median_size") = 5, 
          py::arg("high_intensity_threshold") = 0.995f);
    m.def("voxel_filter", &voxel_filter, "Voxel filter for point cloud",
          py::arg("point_cloud"), 
          py::arg("leaf_size") = 0.4);
}