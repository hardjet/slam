#include <iostream>
#include <chrono>
#include <numeric>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#include "util/image_util.h"
#include "openvslam/config.h"
#include "openvslam/data/bow_database.h"
#include "openvslam/data/frame.h"
#include "openvslam/util/image_converter.h"
#include "openvslam/feature/orb_extractor.h"
#include "openvslam/match/area.h"


using namespace std;
using namespace cv;

void opencv_match(cv::Mat& img_1, cv::Mat& img_2)
{

    //-- 初始化
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;


    Ptr<FeatureDetector> detector = ORB::create(2000);
    Ptr<DescriptorExtractor> descriptor = ORB::create(2000);

    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matches );


    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"opencv costs time: "<<time_used.count() <<" seconds."<<endl;


    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 15.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

    //-- 第五步:绘制匹配结果
    Mat im_kpts;
    drawKeypoints(img_1, keypoints_1, im_kpts);
    imshow ( "opencv keypts", im_kpts );

    Mat img_goodmatch;
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch );
    resize(img_goodmatch, img_goodmatch, Size(img_1.cols*1.2, img_1.rows));
    imshow ( "cv优化后匹配点对", img_goodmatch );

}


void match(const std::shared_ptr<openvslam::config>& cfg, const std::string& image_dir_path)
{
    using namespace openvslam;
    // 获取图像
    image_sequence sequence(image_dir_path, 10);
    const auto frames = sequence.get_frames();


    auto extractor = new feature::orb_extractor(cfg->orb_params_);
    extractor->set_max_num_keypoints(extractor->get_max_num_keypoints());

    auto camera = cfg->camera_;
    auto img1 = cv::imread(frames.at(0).img_path_, cv::IMREAD_UNCHANGED);
    util::convert_to_grayscale(img1, camera->color_order_);
    auto img2 = cv::imread(frames.at(1).img_path_, cv::IMREAD_UNCHANGED);
    util::convert_to_grayscale(img2, camera->color_order_);

    auto init_frm_ = data::frame(img1, 0., extractor, nullptr, camera, cfg->true_depth_thr_, cv::Mat{});
    auto curr_frm = data::frame(img2, 0., extractor, nullptr, camera, cfg->true_depth_thr_, cv::Mat{});

    // ============= BruteForce-Hamming 匹配方法
    Ptr<DescriptorMatcher> matcher1  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher1->match ( init_frm_.descriptors_, curr_frm.descriptors_, matches );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < init_frm_.descriptors_.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < init_frm_.descriptors_.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 15.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;

    drawMatches ( img1, init_frm_.keypts_, img2, curr_frm.keypts_, good_matches, img_goodmatch );
    resize(img_goodmatch, img_goodmatch, Size(img1.cols*1.2, img1.rows));
    imshow ( "BruteForce-Hamming", img_goodmatch );

    // ==============================

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // initialize the previously matched coordinates
    std::vector<cv::Point2f> prev_matched_coords_;
    prev_matched_coords_.resize(init_frm_.undist_keypts_.size());
    for (unsigned int i = 0; i < init_frm_.undist_keypts_.size(); ++i) {
        prev_matched_coords_.at(i) = init_frm_.undist_keypts_.at(i).pt;
    }

    //! initial matching indices (index: idx of initial frame, value: idx of current frame)
    std::vector<int> init_matches_;
    match::area matcher(0.9, true);
    auto num_matches = matcher.match_in_consistent_area(init_frm_, curr_frm, prev_matched_coords_, init_matches_, 100);
    std::cout << "num_matches:" << num_matches << std::endl;

    //! matching between reference and current frames
    std::vector<std::pair<int, int>> ref_cur_matches_;
    ref_cur_matches_.reserve(curr_frm.undist_keypts_.size());
    for (unsigned int ref_idx = 0; ref_idx < init_matches_.size(); ++ref_idx) {
        const auto cur_idx = init_matches_.at(ref_idx);
        if (0 <= cur_idx) {
            ref_cur_matches_.emplace_back(std::make_pair(ref_idx, cur_idx));
        }
    }

    good_matches.clear();
    for (unsigned int i = 0; i < ref_cur_matches_.size(); i++ )
    {
        cv::DMatch match(ref_cur_matches_[i].first, ref_cur_matches_[i].second, 0, 10);
        good_matches.push_back(match);
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"opencvslam costs time: "<<time_used.count() <<" seconds."<<endl;

    Mat im_kpts;
    drawKeypoints(img1, init_frm_.keypts_, im_kpts);
    imshow ( "openvslam keypts", im_kpts );

    cv::drawMatches ( img1, init_frm_.keypts_, img2, curr_frm.keypts_, good_matches, img_goodmatch );
    resize(img_goodmatch, img_goodmatch, Size(img1.cols*1.2, img1.rows));
    cv::imshow("openvslam", img_goodmatch);
    cv::waitKey(0);

    opencv_match(img1, img2);
    cv::waitKey(0);

    std::cout << "end~" << std::endl;
}


int main(int argc, char* argv[])
{

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto img_dir_path = op.add<popl::Value<std::string>>("i", "img-dir", "directory path which contains images");
    auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!img_dir_path->is_set() || !config_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(config_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << *cfg << std::endl;
    match(cfg, img_dir_path->value());


    return EXIT_SUCCESS;

}
