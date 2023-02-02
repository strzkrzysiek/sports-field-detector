#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "line_model_detection/line_model.h"
#include "line_model_detection/line_model_defs.h"

using namespace hawkeye;


int main(int, char* argv[]) {
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Court Lines App";

  LineModel tennis_court_model = defineTennisCourtModel();

  cv::imshow("Model image", tennis_court_model.getImage());
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}
