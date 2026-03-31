#include "pcd_utils.hpp"
#include <vector>
#include <unordered_map>
#include <cmath>

// Voxel coordinate structure
struct VoxelKey {
    int x, y, z;

    bool operator==(const VoxelKey &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Hash function
struct VoxelHash {
    std::size_t operator()(const VoxelKey& k) const {
        // Use bitwise mixed hash to reduce collisions
        using std::size_t;
        using std::hash;
        size_t res = 17;
        res = res * 31 + hash<int>()(k.x);
        res = res * 31 + hash<int>()(k.y);
        res = res * 31 + hash<int>()(k.z);
        return res;
    }
};

// Accumulator structure
struct VoxelAccumulator {
    Eigen::VectorXf sum; // Store sum of all dimensions (x, y, z, intensity, ...)
    int count = 0;
};

// Voxel filter function
RowMatrixXf voxel_filter(Eigen::Ref<const RowMatrixXf>& points, float leaf_size) {
    // Check dimensions
    if (points.rows() == 0) return RowMatrixXf(0, points.cols());
    if (points.cols() < 3) {
        throw std::runtime_error("Input points must have at least 3 columns (x,y,z)");
    }

    int n_points = points.rows();
    int n_dims = points.cols();

    // 1. Find bounding box minimum (simulate Python: points - min)
    // Although mathematically voxel_idx = floor(p/leaf) is sufficient, we calculate min to be consistent with Python logic (positivize)
    // If not strictly requiring bit-level consistency with Python output order/logic, this step can be skipped.
    // Here we keep the subtract min operation for numerical stability.
    Eigen::Vector3f min_bound = points.leftCols(3).colwise().minCoeff();

    // 2. Build hash table
    std::unordered_map<VoxelKey, VoxelAccumulator, VoxelHash> grid;
    // Reserve space, assuming downsampled points are 1/2 of original (heuristic optimization)
    grid.reserve(n_points / 2);

    // 3. Iterate through point cloud (Zero-Copy Read)
    for (int i = 0; i < n_points; ++i) {
        // Calculate voxel index
        int vx = static_cast<int>(std::floor((points(i, 0) - min_bound[0]) / leaf_size));
        int vy = static_cast<int>(std::floor((points(i, 1) - min_bound[1]) / leaf_size));
        int vz = static_cast<int>(std::floor((points(i, 2) - min_bound[2]) / leaf_size));

        VoxelKey key{vx, vy, vz};
        auto& acc = grid[key]; // Find or insert

        if (acc.count == 0) {
            acc.sum = points.row(i); // First insertion, initialize sum
        } else {
            acc.sum += points.row(i); // Accumulate
        }
        acc.count++;
    }

    // 4. Generate result
    // Output matrix must allocate new memory since point count changed
    RowMatrixXf filtered_points(grid.size(), n_dims);

    int row_idx = 0;
    for (auto& kv : grid) {
        // Calculate mean: sum / count
        filtered_points.row(row_idx) = kv.second.sum / static_cast<float>(kv.second.count);
        row_idx++;
    }

    return filtered_points;
}
