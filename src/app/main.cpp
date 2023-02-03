#include <exception>
#include <fstream>
#include <string>

#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "line_model_detection/line_model.h"
#include "line_model_detection/line_model_defs.h"
#include "line_model_detection/line_model_detector.h"

namespace he = hawkeye;
namespace po = boost::program_options;

cv::Mat readImage(const std::string& imgpath, uint width, uint height) {
  LOG(INFO) << "Opening image: " << imgpath;

  std::ifstream ifs(imgpath, std::ios::in | std::ios::binary);
  if (!ifs) {
    LOG(ERROR) << "Could not open the file for reading: " << imgpath;
    return cv::Mat();
  }

  ifs.seekg(0, ifs.end);
  uint file_size = ifs.tellg();
  ifs.seekg(0, ifs.beg);

  if (file_size != width * height) {
    LOG(ERROR) << "Incorrect file size: " << file_size << ", expected: " << width * height;
    return cv::Mat();
  }

  cv::Mat image(height, width, CV_8U);
  ifs.read(image.ptr<char>(), width * height);

  ifs.close();

  return image;
}


int main(int argc, char* argv[]) {
  try {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "Court Lines App";

    // PROGRAM ARGUMENTS ///////////////////////////////////////////////////////////
    
    std::string imgpath;
    uint width, height;

    po::options_description po_desc("Usage");
    po_desc.add_options()
      ("help", "Produce help message")
      ("imgpath,p", po::value<std::string>(&imgpath)->required(), "Path to the raw image file")
      ("width,w", po::value<uint>(&width)->required(), "Image width")
      ("height,h", po::value<uint>(&height)->required(), "Image height");

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
    
    // READING IMAGE ///////////////////////////////////////////////////////////////

    cv::Mat input_image = readImage(imgpath, width, height);
    if (input_image.empty()) {
      return -1;
    }

    // PROCESSING IMAGE ////////////////////////////////////////////////////////////

    cv::imshow("Input image", input_image);

    he::LineModel tennis_court_model = he::defineTennisCourtModel();

    he::LineModelDetector detector;
    auto result = detector.detect(input_image);

    for (const auto& [window_name, display_image] : result) {
      cv::imshow(window_name, display_image);
    }

    cv::imshow("Model image", tennis_court_model.getImage());
    cv::waitKey(0);
    cv::destroyAllWindows();

  } catch(const std::exception& e) {
    LOG(ERROR) << "Error: " << e.what();
    return -1;
  } catch(...) {
    LOG(ERROR) << "Unknown error";
    return -1;
  }

  return 0;
}
