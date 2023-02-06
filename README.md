# sports-field-detector
Detecting predefined sports fields in given images

## Building

The project has been developed under MacOS with necessary libraries installed with `brew`. However, it should also work on Linux operating systems. Windows is not supported.

### Required libraries
* glog (https://github.com/google/glog)
* boost (https://www.boost.org/)
* OpenCV (https://opencv.org/)
* Eigen (https://eigen.tuxfamily.org/)
* Ceres Solver (http://ceres-solver.org/)

### CMake build system

The project uses CMake (https://cmake.org) as build system. To build go into the project directory and type:
```
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```
Make sure to build in the release configuration, otherwise the resulting executable is extremely slow.

## Running

The following options are available in the CLI:
```
  --help                Produce help message
  -s [ --sport ] arg    Choose sport from predefined ones [tennis|volleyball]
  -p [ --imgpath ] arg  Path to the raw image file
  -o [ --outpath ] arg  Path to the output CSV file (optional)
  -w [ --width ] arg    Image width
  -h [ --height ] arg   Image height
  -v [ --visualize ]    Show visualized input and output
  -d [ --debug ]        Show debug images (implies --visualize)
  ```
  
As an input image provide a raw grayscale image with one-byte values per pixel in row-major order. You are required to provide the width and height of the input image.

Example command:
```
./build/app/sports_field_detector_app -s volleyball -p ./examples/volleyball1_950x713.raw  -w 950 -h 713 -o ./detectedlines.csv --debug
```

## Algorithm
The algorithm has three main steps:
* Line pixel extraction [src/line_model_detection/line_pixel_extractor.h](src/line_model_detection/line_pixel_extractor.h)
  - extracting pixels that might be a line.
  - composed of several `Filter`s.
* Line detection [src/line_model_detection/line_detector.h](src/line_model_detection/line_detector.h)
  - detecting lines having extracted pixels as an input.
  - composed of several `Step`s.
* Homography estimation [src/line_model_detection/homography_estimator.h](src/line_model_detection/homography_estimator.h)
  - brute-force matching of the detected and model lines.
  - `Test`ing and `Scoring` the results to find the best estimation.

See the `LineModelDetector::detect()` method in [src/line_model_detection/line_model_detector.cpp](src/line_model_detection/line_model_detector.cpp) to have a quick view on all the parts of the algorithm.

See my playground in Jupyter notebook [prototyping/Court Lines.ipynb](<prototyping/Court Lines.ipynb>) to have some insight into the details.

## Examples

### Tennis

![Tennis detection example](tennis_detection_example.png?raw=true "Tennis detection example")

### Volleyball

![Volleyball detection example](volleyball_detection_example.png?raw=true "Volleyball detection example")
