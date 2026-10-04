#include "batch_plan.h"

namespace tesseract_decoder::batch_detail {
thread_local BatchPlanState* active_batch_plan = nullptr;
thread_local BatchPlanFrame* active_batch_frame = nullptr;
}  // namespace tesseract_decoder::batch_detail
