#include "line_model_detection/line_model_defs.h"

namespace hawkeye {

LineModel defineTennisCourtModel() {
  // One pixel is one inch
  // A line is 2in wide
  // Add one feet margin from each side

  LineModel model(12 * (36 + 2), 12 * (78 + 2), 2);

  model.addLine("Baseline",
                 ImagePoint(   1*12 + 0.5,  1*12 + 0.5),
                 ImagePoint(  37*12 - 1.5,  1*12 + 0.5));

  model.addLine("Baseline",
                 ImagePoint(   1*12 + 0.5, 79*12 - 1.5),
                 ImagePoint(  37*12 - 1.5, 79*12 - 1.5));
                 
  model.addLine("Service line",
                 ImagePoint( 5.5*12 + 0.5, 19*12 - 0.5),
                 ImagePoint(32.5*12 - 1.5, 19*12 - 0.5));

  model.addLine("Service line",
                 ImagePoint( 5.5*12 + 0.5, 61*12 - 0.5),
                 ImagePoint(32.5*12 - 1.5, 61*12 - 0.5));

  model.addLine("Doubles sideline",
                 ImagePoint(   1*12 + 0.5,  1*12 + 0.5),
                 ImagePoint(   1*12 + 0.5, 79*12 - 1.5));

  model.addLine("Doubles sideline",
                 ImagePoint(  37*12 - 1.5,  1*12 + 0.5),
                 ImagePoint(  37*12 - 1.5, 79*12 - 1.5));

  model.addLine("Singles sideline",
                 ImagePoint( 5.5*12 + 0.5,  1*12 + 0.5),
                 ImagePoint( 5.5*12 + 0.5, 79*12 - 1.5));

  model.addLine("Singles sideline",
                 ImagePoint(32.5*12 - 1.5,  1*12 + 0.5),
                 ImagePoint(32.5*12 - 1.5, 79*12 - 1.5));

  model.addLine("Center service line",
                 ImagePoint(  19*12 - 0.5, 19*12 - 0.5),
                 ImagePoint(  19*12 - 0.5, 61*12 - 0.5));

  model.commit();

  return model; 
}

} // namespace hawkeye
