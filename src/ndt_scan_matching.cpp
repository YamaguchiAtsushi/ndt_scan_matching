#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Dense>
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

    // NDTマッチングを実行する関数
    void performNDT(const std::vector<Eigen::Vector2d>& input_points, const std::vector<Eigen::Vector2d>& target_points) {
        Eigen::Matrix3d transformation = Eigen::Matrix3d::Identity();
        const int max_iterations = 30;
        const double epsilon = 1e-4;

        // マッチング結果を格納するためのベクトル
        std::vector<Eigen::Vector2d> transformed_points;

        for (int iter = 0; iter < max_iterations; ++iter) {
            // 更新のための行列とベクトルを初期化
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero(); // 2x2行列に変更
            Eigen::Vector2d b = Eigen::Vector2d::Zero(); // 2次元ベクトルに変更
            int corresponding_points_count = 0;

            // KDTreeでの最近傍探索
            for (const auto& source_point : input_points) { // input_pointsを使用
                Eigen::Vector3d source_h = transformation * Eigen::Vector3d(source_point(0), source_point(1), 1.0);
                Eigen::Vector2d query(source_h(0), source_h(1));

                double min_dist = std::numeric_limits<double>::max();
                Eigen::Vector2d closest_target;

                for (const auto& target_point : target_points) {
                    double dist = (query - target_point).norm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_target = target_point;
                    }
                }

                if (min_dist < 3.0) { // 最大距離の閾値
                    Eigen::Vector2d error = closest_target - query;

                    // ヘッセ行列Hとベクトルbの更新
                    Eigen::Matrix2d J; // ジャコビ行列
                    J << -1, 0,
                         0, -1;
                    H += J.transpose() * J;
                    b += J.transpose() * error;
                    corresponding_points_count++;
                }
            }

            // 更新量の計算
            if (corresponding_points_count > 0) {
                Eigen::Vector2d delta = H.colPivHouseholderQr().solve(-b); // 2次元ベクトルの更新量
                transformation *= expMap(delta); // expMapも修正が必要です
                transformation(2, 2) = 1.0; // z成分は変更しない

                if (delta.norm() < epsilon) {
                    ROS_INFO("NDT scan matching has converged");
                    break;
                }
            }
        }

        // 最終的なポーズを公開
        geometry_msgs::Pose pose;
        pose.position.x = transformation(0, 2);
        pose.position.y = transformation(1, 2);
        pose.orientation.w = 1.0; // 回転なしの単純な例
        pose_publisher_.publish(pose);

        // マッチング結果の点群を計算
        for (const auto& point : input_points) { // input_pointsを使用
            Eigen::Vector3d transformed_point = transformation * Eigen::Vector3d(point(0), point(1), 1.0);
            transformed_points.push_back(Eigen::Vector2d(transformed_point(0), transformed_point(1)));
        }

        // RVizにマーカーを表示
        publishMarker(pose, input_points, transformed_points);
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

        // 例: 固定のターゲットポイントを設定する
        std::vector<Eigen::Vector2d> target_points = {
            Eigen::Vector2d(1.0, 0.0),
            Eigen::Vector2d(0.0, 1.0),
            Eigen::Vector2d(-1.0, 0.0),
            Eigen::Vector2d(0.0, -1.0)
        };

        // NDTマッチングを実行
        performNDT(input_points, target_points);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ndt_scan_matching");
    NDTScanMatcher ndt_scan_matcher;

    ros::spin();
    return 0;
}
