#include <memory>

#include "armor_detector/detector.hpp"
#include "armor_detector/detector_base.hpp"
#include "armor_detector/yolo_detector.hpp"

namespace rm_auto_aim
{

struct DetectorParams
{
  std::string type;
  Detector::DetectorParams params;
  YoloDetector::YoloParams yolo_params;
};

class DetectorFactory
{
 public:
  std::unique_ptr<DetectorBase> Create(DetectorParams params)
  {
    if (params.type == "traditional")
    {
      return std::make_unique<Detector>(params.params);
    }

#ifdef ARMOR_DETECTOR_HAS_OPENVINO
    if (params.type == "yolo")
    {
      return std::make_unique<YoloDetector>(params.yolo_params);
    }
#else
    if (params.type == "yolo")
    {
      throw std::runtime_error(
          "yolo detector is requested, but this package was built without OpenVINO "
          "support");
    }
#endif

    throw std::runtime_error("unknown detector type: " + params.type);
  }
};

inline std::unique_ptr<DetectorFactory> detector_factory;

}  // namespace rm_auto_aim