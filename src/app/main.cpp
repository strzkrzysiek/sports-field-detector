// Copyright 2023 Krzysztof Wrobel

#include <exception>
#include <string>

#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "app/app_utlis.h"
#include "line_model_detection/line_model.h"
#include "line_model_detection/line_model_defs.h"
#include "line_model_detection/line_model_detector.h"

namespace he = hawkeye;
namespace po = boost::program_options;

int runApp(const std::string& imgpath, const std::string& outpath, uint width, uint height, bool visualize, bool debug) {
  if (debug) visualize = true;
  
  // READING IMAGE ///////////////////////////////////////////////////////////////

  cv::Mat input_image = he::readImage(imgpath, width, height);
  if (input_image.empty()) {
    return -1;
  }

  if (visualize) {
    cv::imshow("Image", input_image);
    cv::waitKey(1);
  }

  // PROCESSING IMAGE ////////////////////////////////////////////////////////////

  he::LineModel tennis_court_model = he::defineTennisCourtModel();
  he::Mat3 camera_matrix = he::fakeCameraMatrix(input_image.size());

  he::LineModelDetector detector(tennis_court_model);
  auto result = detector.detect(input_image, camera_matrix, debug);

  // PRESENTING RESULTS //////////////////////////////////////////////////////////

  if (!outpath.empty() && result.model2camera_image_homography) {
    generateOutputCsvFile(outpath,
                          result.model2camera_image_homography.value(),
                          input_image.size(),
                          tennis_court_model);
  }

  if (visualize) {
    if (result.model2camera_image_homography) {
      cv::imshow("Image", result.visualization);
    }

    for (const auto& [window_name, display_image] : result.debug_images) {
      cv::imshow(window_name, display_image);
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
  }

  return 0;
}

int main(int argc, char* argv[]) {
  try {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "Court Lines App";

    // PROGRAM ARGUMENTS ///////////////////////////////////////////////////////////
    
    std::string imgpath;
    std::string outpath;
    uint width, height;

    po::options_description po_desc("Usage");
    po_desc.add_options()
      ("help", "Produce help message")
      ("imgpath,p", po::value<std::string>(&imgpath)->required(), "Path to the raw image file")
      ("outpath,o", po::value<std::string>(&outpath), "Path to the output CSV file (optional)")
      ("width,w", po::value<uint>(&width)->required(), "Image width")
      ("height,h", po::value<uint>(&height)->required(), "Image height")
      ("visualize,v", "Show visualized input and output")
      ("debug,d", "Show debug images (implies --visualize)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, po_desc), vm);

    if (argc <= 1 || vm.count("help")) {
      LOG(INFO) << "\n" << po_desc;
      return 0;
    }

    try {
      po::notify(vm);
    } catch(const po::error& e) {
      LOG(ERROR) << "Program options error: " << e.what() << '\n' << po_desc;
      return -1;
    }

    return runApp(imgpath, outpath, width, height, vm.count("visualize"), vm.count("debug"));

  } catch(const std::exception& e) {
    LOG(ERROR) << "Error: " << e.what();
    return -1;
  } catch(...) {
    LOG(ERROR) << "Unknown error";
    return -1;
  }

  return 0;
}
