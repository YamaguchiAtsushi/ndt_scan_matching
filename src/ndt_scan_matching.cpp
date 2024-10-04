#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

class NDTScanMatcher {
public:
    NDTScanMatcher() {
        scan_subscriber_ = nh_.subscribe("scan", 1, &NDTScanMatcher::scanCallback, this);
        pose_publisher_ = nh_.advertise<geometry_msgs::Pose>("ndt_pose", 1);
        marker_publisher_ = nh_.advertise<visualization_msgs::Marker>("ndt_marker", 1);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber scan_subscriber_;
    ros::Publisher pose_publisher_;
    ros::Publisher marker_publisher_;

    // グリッドサイズを定義
    const double grid_size = 1.0; // 1メートルごとのグリッド


    // NDTのためにグリッドを構築し、平均と共分散行列を計算する関数
    void buildNDTGrid(const std::vector<Eigen::Vector2d>& points,
                    std::map<std::pair<int, int>, std::vector<Eigen::Vector2d>>& grid_cells,
                    std::map<std::pair<int, int>, Eigen::Vector2d>& grid_means,
                    std::map<std::pair<int, int>, Eigen::Matrix2d>& grid_covariances) {
        
        // グリッドセルごとに点群を分類
        for (const auto& point : points) {
            int grid_x = static_cast<int>(point(0) / grid_size);
            int grid_y = static_cast<int>(point(1) / grid_size);
            grid_cells[{grid_x, grid_y}].push_back(point);
        }

        // 各グリッドセルごとに平均と分散を計算
        for (auto& cell : grid_cells) {
            const std::vector<Eigen::Vector2d>& cell_points = cell.second;
            int num_points = cell_points.size();

            // 平均の計算
            Eigen::Vector2d mean = Eigen::Vector2d::Zero();
            for (const auto& point : cell_points) {
                mean += point;
            }
            mean /= num_points;
            grid_means[cell.first] = mean;

            // 共分散行列の計算
            Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
            for (const auto& point : cell_points) {
                Eigen::Vector2d diff = point - mean;
                covariance += diff * diff.transpose();
            }
            covariance /= num_points - 1;
            grid_covariances[cell.first] = covariance;
        }
    }

    // NDTマッチングを実行する関数
    //Eigen::Vector2d : double型の2次元のvector
    //Eigen::Matrix3d : double型の3×3の行列
    void performNDT(const std::vector<Eigen::Vector2d>& input_points, 
                    const std::map<std::pair<int, int>, Eigen::Vector2d>& grid_means, 
                    const std::map<std::pair<int, int>, Eigen::Matrix2d>& grid_covariances) {
        
        Eigen::Matrix3d transformation = Eigen::Matrix3d::Identity();
        const int max_iterations = 30;
        const double epsilon = 1e-4;

        for (int iter = 0; iter < max_iterations; ++iter) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            int corresponding_points_count = 0;

            for (const auto& source_point : input_points) {
                Eigen::Vector3d source_h = Eigen::Vector3d(source_point(0), source_point(1), 1.0);
                Eigen::Vector2d query(source_h(0), source_h(1));

                // 点をグリッドにマッピング
                int grid_x = static_cast<int>(query(0) / grid_size);
                int grid_y = static_cast<int>(query(1) / grid_size);
                std::pair<int, int> grid_cell = {grid_x, grid_y};

                // グリッドが存在するかチェック
                if (grid_means.count(grid_cell) > 0 && grid_covariances.count(grid_cell) > 0) {
                    const Eigen::Vector2d& mean = grid_means.at(grid_cell);
                    const Eigen::Matrix2d& covariance = grid_covariances.at(grid_cell);
                    
                    Eigen::Vector2d error = query - mean;

                    // 共分散行列の逆行列を使用して誤差を計算
                    Eigen::Matrix2d inv_covariance = covariance.inverse();
                    double mahalanobis_distance = error.transpose() * inv_covariance * error;

                    if (mahalanobis_distance < 3.0) { // 距離の閾値（適切に調整）
                        Eigen::Matrix2d J = -inv_covariance;
                        H += J.transpose() * J;
                        b += J.transpose() * error;
                        corresponding_points_count++;
                    }
                }
            }

            if (corresponding_points_count > 0) {
                Eigen::Vector2d delta = H.colPivHouseholderQr().solve(-b);
                transformation *= expMap(delta);
                transformation(2, 2) = 1.0;

                if (delta.norm() < epsilon) {
                    ROS_INFO("NDT scan matching has converged");
                    break;
                }
            }
        }

        // 最終的な結果のポーズを公開
        geometry_msgs::Pose pose;
        pose.position.x = transformation(0, 2);
        pose.position.y = transformation(1, 2);
        pose.orientation.w = 1.0;

        // 入力点を変換してマーカーを作成
        std::vector<Eigen::Vector2d> transformed_points;
        for (const auto& point : input_points) {
            Eigen::Vector3d transformed_point = transformation * Eigen::Vector3d(point(0), point(1), 1.0);
            transformed_points.push_back(Eigen::Vector2d(transformed_point(0), transformed_point(1)));
        }

        publishMarker(pose, input_points, transformed_points); // マーカーをパブリッシュ
        pose_publisher_.publish(pose);
    }

    Eigen::Matrix3d expMap(const Eigen::Vector2d& v) {
        double theta = atan2(v(1), v(0)); // y/xから角度を計算
        Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
        double c = cos(theta);
        double s = sin(theta);
        T(0, 0) = c;
        T(0, 1) = -s;
        T(1, 0) = s;
        T(1, 1) = c;
        T(0, 2) = v(0);
        T(1, 2) = v(1);
        return T;
    }

    void publishMarker(const geometry_msgs::Pose& pose, 
                       const std::vector<Eigen::Vector2d>& input_points, 
                       const std::vector<Eigen::Vector2d>& transformed_points) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "laser"; // RVizでのフレームを指定
        marker.header.stamp = ros::Time::now();
        marker.ns = "ndt_results";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_LIST; // 線分として表示

        // マーカーの色
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        // マーカーのスケール
        marker.scale.x = 0.01; // 線の太さ

        // 入力点群をマーカーに追加
        for (const auto& point : input_points) {
            geometry_msgs::Point p;
            p.x = point(0);
            p.y = point(1);
            p.z = 0;
            marker.points.push_back(p);
        }

        // マッチング結果の点群をマーカーに追加
        for (const auto& point : transformed_points) {
            geometry_msgs::Point p;
            p.x = point(0);
            p.y = point(1);
            p.z = 0;
            marker.points.push_back(p);
        }

        marker_publisher_.publish(marker);
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan) {
        std::vector<Eigen::Vector2d> input_points;

        for (size_t i = 0; i < scan->ranges.size(); ++i) {
            float angle = scan->angle_min + i * scan->angle_increment;
            float range = scan->ranges[i];

            if (range < scan->range_max) {
                Eigen::Vector2d point;
                point(0) = range * cos(angle);
                point(1) = range * sin(angle);
                input_points.push_back(point);
            }
        }

        // ターゲット点の設定
        std::vector<Eigen::Vector2d> target_points = {
            Eigen::Vector2d(1.0, 0.0),
            Eigen::Vector2d(0.0, 1.0),
            Eigen::Vector2d(-1.0, 0.0),
            Eigen::Vector2d(0.0, -1.0)
        };

        // グリッド内の平均と共分散を計算するための準備
        std::map<std::pair<int, int>, Eigen::Matrix<double, 2, 1>> mean_map;
        std::map<std::pair<int, int>, Eigen::Matrix<double, 2, 2>> covariance_map;

        // グリッドのサイズや分割数を決定するための変数
        const int grid_size = 10; // 例: 10x10のグリッド
        const double cell_size = 0.5; // セルのサイズ

        // 入力点をグリッドに振り分ける
        for (const auto& point : input_points) {
            int grid_x = static_cast<int>(point(0) / cell_size);
            int grid_y = static_cast<int>(point(1) / cell_size);
            std::pair<int, int> grid_key = std::make_pair(grid_x, grid_y);

            // 平均と共分散の更新
            if (mean_map.find(grid_key) == mean_map.end()) {
                mean_map[grid_key] = Eigen::Matrix<double, 2, 1>::Zero();
                covariance_map[grid_key] = Eigen::Matrix<double, 2, 2>::Zero();
            }

            mean_map[grid_key] += point; // 平均に加算
            covariance_map[grid_key](0, 0) += point(0) * point(0);
            covariance_map[grid_key](0, 1) += point(0) * point(1);
            covariance_map[grid_key](1, 0) += point(1) * point(0);
            covariance_map[grid_key](1, 1) += point(1) * point(1);
        }

        // NDTマッチングを実行
        performNDT(input_points, mean_map, covariance_map);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ndt_scan_matching");
    NDTScanMatcher ndt_scan_matcher;

    ros::spin();
    return 0;
}
