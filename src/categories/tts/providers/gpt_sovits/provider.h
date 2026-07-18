#pragma once

#include "providers/tts_provider.h"

namespace tts::gpt_sovits_provider {

std::shared_ptr<ITTSModel> load_model(
    const ModelConfig& config,
    const RuntimeContext& runtime
);

} // namespace tts::gpt_sovits_provider
