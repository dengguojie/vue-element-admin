#include "sampling_kernels.h"
#include <algorithm>
#include "log.h"
#include "status.h"
using namespace std;

namespace aicpu {
SamplingKernelType SamplingKernelTypeFromString(std::string str) {
  if (str == "lanczos1") return Lanczos1Kernel;
  if (str == "lanczos3") return Lanczos3Kernel;
  if (str == "lanczos5") return Lanczos5Kernel;
  if (str == "gaussian") return GaussianKernel;
  if (str == "box") return BoxKernel;
  if (str == "triangle") return TriangleKernel;
  if (str == "keyscubic") return KeysCubicKernel;
  if (str == "mitchellcubic") return MitchellCubicKernel;
  return SamplingKernelTypeEnd;
}
}  // namespace aicpu