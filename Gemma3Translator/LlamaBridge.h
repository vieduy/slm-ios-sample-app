#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Thin Obj-C wrapper around llama.cpp's C API (llama.h), exposing the *same*
/// surface as GemmaBridge so the Swift side can treat either engine
/// identically (see the InferenceEngine protocol). Loads a GGUF model
/// (e.g. Bonsai-1.7B Q1_0, a qwen3-arch 1-bit model), applies the model's
/// embedded chat template, and streams UTF-8 chunks back per decoded token.
@interface LlamaBridge : NSObject

- (instancetype)init;

/// Load a .gguf model and create a llama context.
/// backend:  @"cpu" (n_gpu_layers=0) or @"gpu" (offload all layers to Metal).
///           NOTE: Q1_0 currently has a CPU kernel in mainline llama.cpp; a
///           Metal kernel may be absent, so @"gpu" can fail at load/decode for
///           1-bit models — callers should be ready to fall back to @"cpu".
/// cacheDir: unused by llama.cpp (no external kernel cache); accepted for
///           signature parity with GemmaBridge. Pass nil.
/// Returns NO on failure (-lastError for details).
- (BOOL)loadWithModelPath:(NSString *)modelPath
                  backend:(NSString *)backend
                 cacheDir:(nullable NSString *)cacheDir
                benchmark:(BOOL)benchmark
    NS_SWIFT_NAME(load(modelPath:backend:cacheDir:benchmark:));

/// Engine-measured prefill/decode throughput from the most recent generation
/// (llama_perf_context). Keys match GemmaBridge:
/// "ttftMs", "prefillTps", "decodeTps", "prefillTokens", "decodeTokens".
/// Returns nil if benchmark instrumentation wasn't enabled at load.
- (nullable NSDictionary<NSString *, NSNumber *> *)lastBenchmark;

/// Blocking generation. Returns the full reply, or nil on error.
- (nullable NSString *)generatePrompt:(NSString *)prompt
                            maxTokens:(int)maxTokens
                          temperature:(float)temperature
                                 topK:(int)topK
                                 topP:(float)topP
    NS_SWIFT_NAME(generate(prompt:maxTokens:temperature:topK:topP:));

/// Streaming generation. `onChunk` fires on the calling thread per decoded
/// token; the final invocation has done=YES with an empty chunk.
- (void)streamPrompt:(NSString *)prompt
           maxTokens:(int)maxTokens
         temperature:(float)temperature
                topK:(int)topK
                topP:(float)topP
             onChunk:(void (^)(NSString *chunk, BOOL done))onChunk
    NS_SWIFT_NAME(stream(prompt:maxTokens:temperature:topK:topP:onChunk:));

/// Clear the KV cache so the next generation starts fresh. (Each generate/
/// stream call already clears state, so this is mainly for parity.)
- (void)resetSession;

#pragma mark - Dynamic LoRA

/// Load a LoRA adapter (.gguf, produced by convert_lora_to_gguf.py against the
/// SAME base architecture) and keep it resident, keyed by `identifier`. The
/// adapter is tied to the loaded model, so this must be re-called after every
/// engine (re)load. Idempotent per identifier (a second load replaces the first).
/// Returns NO on failure — e.g. the adapter's arch/tokenizer doesn't match the
/// base (see -lastError).
- (BOOL)loadAdapterAtPath:(NSString *)path
               identifier:(NSString *)identifier
    NS_SWIFT_NAME(loadAdapter(path:identifier:));

/// Activate (or blend) a previously-loaded adapter on the live context, taking
/// effect on the next decode — no model reload. Pass `identifier` = nil/empty to
/// revert to the pure base model. `scale` is the LoRA strength (1.0 = as trained,
/// 0.0 = off). Returns NO if `identifier` was never loaded (see -lastError).
- (BOOL)setActiveAdapter:(nullable NSString *)identifier
                   scale:(float)scale
    NS_SWIFT_NAME(setActiveAdapter(_:scale:));

@property (nonatomic, readonly, copy) NSString *lastError;

@end

NS_ASSUME_NONNULL_END
