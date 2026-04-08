/// NEXUS — CoreML Draft Model for Neural Engine (Objective-C++ implementation).
///
/// Wraps Apple's CoreML MLModel API to run a small draft model on the ANE,
/// freeing the GPU for the main (large) model during speculative decoding.
///
/// Key: MLComputeUnitsCPUAndNeuralEngine routes computation to the ANE,
/// which runs at very low power and doesn't compete with Metal GPU work.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "draft_model.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace nexus::compute {

// ─── Implementation ─────────────────────────────────────────────────────────

struct DraftModel::Impl {
    DraftModelConfig config;

#ifdef NEXUS_HAS_COREML
    MLModel* coreml_model = nil;
#endif

    bool coreml_active = false;
    bool ready = false;

    // Fallback: simple greedy sampling from logits
    int32_t sample_greedy(const std::vector<float>& logits) {
        return static_cast<int32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    int32_t sample_token(const std::vector<float>& logits, const SamplingParams& params) {
        if (params.temperature <= 0.0f) return sample_greedy(logits);

        // Temperature + top-k sampling for draft tokens
        int vocab = static_cast<int>(logits.size());
        std::vector<float> probs(vocab);
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (int i = 0; i < vocab; i++) {
            probs[i] = expf((logits[i] - max_l) / params.temperature);
            sum += probs[i];
        }
        for (int i = 0; i < vocab; i++) probs[i] /= sum;

        std::mt19937 rng(std::random_device{}());
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return static_cast<int32_t>(dist(rng));
    }
};

// ─── Constructor / Destructor ───────────────────────────────────────────────

DraftModel::DraftModel(const DraftModelConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;

#ifdef NEXUS_HAS_COREML
    if (!config.coreml_model_path.empty()) {
        @autoreleasepool {
            NSError* error = nil;

            // Configure for Neural Engine
            MLModelConfiguration* mlconfig = [[MLModelConfiguration alloc] init];
            if (config.use_ane) {
                mlconfig.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
                fprintf(stderr, "[nexus] DraftModel: routing to CPU + Neural Engine\n");
            } else {
                mlconfig.computeUnits = MLComputeUnitsCPUOnly;
                fprintf(stderr, "[nexus] DraftModel: routing to CPU only\n");
            }

            // Load compiled model
            NSURL* modelURL = [NSURL fileURLWithPath:
                [NSString stringWithUTF8String:config.coreml_model_path.c_str()]];

            impl_->coreml_model = [MLModel modelWithContentsOfURL:modelURL
                                                   configuration:mlconfig
                                                           error:&error];

            if (impl_->coreml_model) {
                impl_->coreml_active = true;
                impl_->ready = true;
                fprintf(stderr, "[nexus] DraftModel: CoreML model loaded from %s\n",
                        config.coreml_model_path.c_str());
            } else {
                fprintf(stderr, "[nexus] DraftModel: CoreML load failed: %s\n",
                        error ? [[error localizedDescription] UTF8String] : "unknown");
                fprintf(stderr, "[nexus] DraftModel: falling back to CPU draft model\n");
            }
        }
    }
#endif

    if (!impl_->coreml_active) {
        // Fallback: we'll use a simplified approach
        // In a full implementation, this would load a small NEXUS transformer
        // For now, mark as ready with fallback mode
        impl_->ready = true;
        fprintf(stderr, "[nexus] DraftModel: using fallback mode (no CoreML)\n");
    }
}

DraftModel::~DraftModel() {
#ifdef NEXUS_HAS_COREML
    @autoreleasepool {
        impl_->coreml_model = nil;
    }
#endif
}

DraftModel::DraftModel(DraftModel&&) noexcept = default;
DraftModel& DraftModel::operator=(DraftModel&&) noexcept = default;

// ─── Queries ────────────────────────────────────────────────────────────────

bool DraftModel::is_coreml_active() const { return impl_->coreml_active; }
bool DraftModel::is_ready() const { return impl_->ready; }
int DraftModel::vocab_size() const { return impl_->config.vocab_size; }

// ─── Prediction ─────────────────────────────────────────────────────────────

std::vector<float> DraftModel::predict(const std::vector<int32_t>& input_tokens, int seq_len) {
    std::vector<float> logits(impl_->config.vocab_size, 0.0f);

#ifdef NEXUS_HAS_COREML
    if (impl_->coreml_active && impl_->coreml_model) {
        @autoreleasepool {
            NSError* error = nil;

            // Create input MLMultiArray for token IDs
            NSArray<NSNumber*>* shape = @[@1, @((int)input_tokens.size())];
            MLMultiArray* input_array = [[MLMultiArray alloc]
                initWithShape:shape
                     dataType:MLMultiArrayDataTypeInt32
                        error:&error];

            if (!input_array) {
                fprintf(stderr, "[nexus] DraftModel: failed to create input array\n");
                return logits;
            }

            // Fill input tokens
            for (size_t i = 0; i < input_tokens.size(); i++) {
                input_array[[NSArray arrayWithObjects:@0, @((int)i), nil]] =
                    @(input_tokens[i]);
            }

            // Create feature provider
            NSDictionary* input_dict = @{@"input_ids": input_array};
            MLDictionaryFeatureProvider* provider =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict
                                                                 error:&error];

            if (!provider) {
                fprintf(stderr, "[nexus] DraftModel: failed to create feature provider\n");
                return logits;
            }

            // Run prediction
            id<MLFeatureProvider> output = [impl_->coreml_model predictionFromFeatures:provider
                                                                                 error:&error];

            if (!output) {
                fprintf(stderr, "[nexus] DraftModel: prediction failed: %s\n",
                        error ? [[error localizedDescription] UTF8String] : "unknown");
                return logits;
            }

            // Extract logits from output
            MLMultiArray* output_logits = [output featureValueForName:@"logits"].multiArrayValue;
            if (output_logits) {
                int vocab = std::min(impl_->config.vocab_size,
                                     static_cast<int>(output_logits.count));
                for (int i = 0; i < vocab; i++) {
                    logits[i] = output_logits[i].floatValue;
                }
            }
        }
        return logits;
    }
#endif

    // Fallback: return uniform logits (draft model not available)
    // A real fallback would run a small transformer on CPU
    float uniform = 1.0f / impl_->config.vocab_size;
    std::fill(logits.begin(), logits.end(), uniform);
    return logits;
}

// ─── Draft Generation ───────────────────────────────────────────────────────

std::vector<int32_t> DraftModel::generate_draft(const std::vector<int32_t>& prompt,
                                                  int n_tokens,
                                                  const SamplingParams& params) {
    std::vector<int32_t> draft_tokens;
    draft_tokens.reserve(n_tokens);

    // Build context from prompt
    std::vector<int32_t> context = prompt;
    int seq_len = static_cast<int>(prompt.size());

    for (int i = 0; i < n_tokens; i++) {
        // Get logits for next token
        auto logits = predict(context, seq_len);

        // Sample (typically greedy for draft tokens)
        SamplingParams draft_params = params;
        draft_params.temperature = 0.0f;  // Greedy for speed
        int32_t next = impl_->sample_token(logits, draft_params);

        draft_tokens.push_back(next);
        context.push_back(next);
        seq_len++;
    }

    return draft_tokens;
}

void DraftModel::reset() {
    // Reset any internal KV cache state
    // CoreML models are stateless per prediction, so nothing to do for CoreML path
    // Fallback transformer would need cache reset here
}

}  // namespace nexus::compute
