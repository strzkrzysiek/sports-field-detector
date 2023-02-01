#include <glog/logging.h>

int main(int, char* argv[]) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "Court Lines App";

    return 0;
}
