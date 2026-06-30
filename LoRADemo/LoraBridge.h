#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Minimal Obj-C++ wrapper over the LiteRT-LM C API (CLiteRTLM-LoRA.xcframework)
/// that demonstrates dynamic LoRA adapter swapping.
///
/// Design: one Engine holds the base model resident. Each generate call opens a
/// fresh Session whose SessionConfig optionally points at a LoRA adapter file
/// (`litert_lm_session_config_set_lora_path`). Switching adapters = switching
/// sessions; the base model is never reloaded.
@interface LoraBridge : NSObject

/// Loads the base .litertlm and warms the engine.
/// backend: @"gpu" (Metal) or @"cpu".  cacheDir: writable XNNPACK cache path.
/// Returns NO on failure (see -lastError).
- (BOOL)loadBaseModel:(NSString *)modelPath
              backend:(NSString *)backend
             cacheDir:(nullable NSString *)cacheDir
    NS_SWIFT_NAME(loadBase(modelPath:backend:cacheDir:));

/// Runs `prompt` in a fresh session. If `loraPath` is non-nil the adapter at
/// that path is activated for this generation only; pass nil for the base model.
/// Blocking. Returns the generated text, or nil on error.
- (nullable NSString *)generate:(NSString *)prompt
                       loraPath:(nullable NSString *)loraPath
                      maxTokens:(int)maxTokens
    NS_SWIFT_NAME(generate(prompt:loraPath:maxTokens:));

@property (nonatomic, readonly, copy) NSString *lastError;

@end

NS_ASSUME_NONNULL_END
