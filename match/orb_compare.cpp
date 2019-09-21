#include <iostream>
#include <chrono>
#include <numeric>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#include "openvslam/config.h"
#include "openvslam/data/bow_database.h"
#include "openvslam/data/frame.h"
#include "openvslam/util/image_converter.h"
#include "openvslam/feature/orb_extractor.h"
#include "openvslam/match/area.h"

using namespace std;
using namespace cv;


void orb_compare(const std::shared_ptr<openvslam::config>& im_cfg,
                 const std::shared_ptr<openvslam::config>& rs_cfg)
{
    using namespace openvslam;

    auto extractor_im = new feature::orb_extractor(im_cfg->orb_params_);
    auto extractor_rs = new feature::orb_extractor(rs_cfg->orb_params_);


    auto img_im= cv::imread("/home/anson/work/vslam/vslam/data/match/indemend/right.jpg",
            cv::IMREAD_UNCHANGED);

    auto img_rs = cv::imread("/home/anson/work/vslam/vslam/data/match/realsense/right.jpg",
            cv::IMREAD_UNCHANGED);


    auto camera_im = im_cfg->camera_;
    auto camera_rs = rs_cfg->camera_;

    auto frm_im = data::frame(img_im, 0., extractor_im, nullptr, camera_im, im_cfg->true_depth_thr_, cv::Mat{});
    auto frm_rs = data::frame(img_rs, 0., extractor_rs, nullptr, camera_rs, rs_cfg->true_depth_thr_, cv::Mat{});

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // initialize the previously matched coordinates
    std::vector<cv::Point2f> prev_matched_coords_;
    prev_matched_coords_.resize(frm_im.undist_keypts_.size());
    for (unsigned int i = 0; i < frm_im.undist_keypts_.size(); ++i) {
        prev_matched_coords_.at(i) = frm_im.undist_keypts_.at(i).pt;
    }

    //! initial matching indices (index: idx of initial frame, value: idx of current frame)
    std::vector<int> init_matches_;
    match::area matcher(0.9, true);
    auto num_matches = matcher.match_in_consistent_area(frm_im, frm_rs, prev_matched_coords_, init_matches_, 100);
    std::cout << "num_matches:" << num_matches << std::endl;

    //! matching between reference and current frames
    std::vector<std::pair<int, int>> ref_cur_matches_;
    ref_cur_matches_.reserve(frm_rs.undist_keypts_.size());
    for (unsigned int ref_idx = 0; ref_idx < init_matches_.size(); ++ref_idx) {
        const auto cur_idx = init_matches_.at(ref_idx);
        if (0 <= cur_idx) {
            ref_cur_matches_.emplace_back(std::make_pair(ref_idx, cur_idx));
        }
    }

    std::vector< cv::DMatch > good_matches;
    for (unsigned int i = 0; i < ref_cur_matches_.size(); i++ )
    {
        cv::DMatch match(ref_cur_matches_[i].first, ref_cur_matches_[i].second, 0, 10);
        good_matches.push_back(match);
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"opencvslam costs time: "<<time_used.count() <<" seconds."<<endl;

    Mat img_goodmatch;
    cv::drawMatches ( img_im, frm_im.keypts_, img_rs, frm_rs.keypts_, good_matches, img_goodmatch );
    cv::resize(img_goodmatch, img_goodmatch, Size(int(2560*0.7), int(720*0.7)));
    cv::imshow("match", img_goodmatch);
    cv::waitKey(0);

    std::cout << "end~" << std::endl;

}


int main()
{
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif

    std::string indemind_config_file_path = "/home/anson/work/vslam/vslam/data/match/indemend/Indemind_right.yaml";
    std::string realsense_config_file_path = "/home/anson/work/vslam/vslam/data/match/realsense/D435i_ir_r.yaml";

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");

    spdlog::set_level(spdlog::level::debug);

    // load configuration
    std::shared_ptr<openvslam::config> indemind_cfg;
    try {
        indemind_cfg = std::make_shared<openvslam::config>(indemind_config_file_path);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::shared_ptr<openvslam::config> realsense_cfg;
    try {
        realsense_cfg = std::make_shared<openvslam::config>(realsense_config_file_path);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }


    orb_compare(indemind_cfg, realsense_cfg);

    return EXIT_SUCCESS;

}
