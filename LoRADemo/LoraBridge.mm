#import "LoraBridge.h"

#import <CLiteRTLM/engine.h>

#include <string>

@implementation LoraBridge {
    LiteRtLmEngine *_engine;
    std::string _err;
}

- (instancetype)init {
    if ((self = [super init])) {
        _engine = nullptr;
    }
    return self;
}

- (void)dealloc {
    if (_engine) litert_lm_engine_delete(_engine);
}

- (NSString *)lastError {
    return _err.empty() ? @"" : [NSString stringWithUTF8String:_err.c_str()];
}

- (BOOL)loadBaseModel:(NSString *)modelPath
              backend:(NSString *)backend
             cacheDir:(NSString *)cacheDir {
    _err.clear();
    LiteRtLmEngineSettings *settings = litert_lm_engine_settings_create(
        modelPath.UTF8String, backend.UTF8String, /*vision=*/nullptr,
        /*audio=*/nullptr);
    if (!settings) {
        _err = "litert_lm_engine_settings_create failed";
        return NO;
    }
    if (cacheDir.length > 0) {
        litert_lm_engine_settings_set_cache_dir(settings, cacheDir.UTF8String);
    }
    _engine = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);
    if (!_engine) {
        _err = "litert_lm_engine_create failed (check model path / backend)";
        return NO;
    }
    return YES;
}

- (NSString *)generate:(NSString *)prompt
              loraPath:(NSString *)loraPath
             maxTokens:(int)maxTokens {
    _err.clear();
    if (!_engine) { _err = "engine not loaded"; return nil; }

    LiteRtLmSessionConfig *cfg = litert_lm_session_config_create();
    if (!cfg) { _err = "session_config_create failed"; return nil; }
    litert_lm_session_config_set_max_output_tokens(cfg, maxTokens);

    if (loraPath.length > 0) {
        // This is the dynamic-LoRA entry point. Returns 0 on success.
        int rc = litert_lm_session_config_set_lora_path(cfg, loraPath.UTF8String);
        if (rc != 0) {
            _err = std::string("set_lora_path failed (rc=") +
                   std::to_string(rc) + ") for " + loraPath.UTF8String;
            litert_lm_session_config_delete(cfg);
            return nil;
        }
    }

    LiteRtLmSession *session = litert_lm_engine_create_session(_engine, cfg);
    litert_lm_session_config_delete(cfg);
    if (!session) {
        // A failure HERE when loraPath != nil is the signal that the engine
        // rejected the adapter (e.g. the old "Lora is not supported." path).
        _err = "create_session failed (LoRA load may have been rejected)";
        return nil;
    }

    std::string promptStr = prompt.UTF8String;
    LiteRtLmInputData *input = litert_lm_input_data_create(
        kLiteRtLmInputDataTypeText, promptStr.data(), promptStr.size());
    const LiteRtLmInputData *inputs[1] = {input};

    LiteRtLmResponses *resp =
        litert_lm_session_generate_content(session, inputs, 1);

    NSString *result = nil;
    if (resp) {
        const char *text = litert_lm_responses_get_response_text_at(resp, 0);
        if (text) result = [NSString stringWithUTF8String:text];
        litert_lm_responses_delete(resp);
    } else {
        _err = "generate_content returned null";
    }

    litert_lm_input_data_delete(input);
    litert_lm_session_delete(session);
    return result;
}

@end
