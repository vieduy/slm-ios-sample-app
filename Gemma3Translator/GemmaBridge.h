#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Thin Obj-C wrapper around gemma::GemmaEngine (which itself wraps
/// LiteRT-LM's C engine API). LiteRT-LM owns tokenization, Jinja chat
/// templating, KV cache, and sampling — the Swift side just hands in a
/// raw prompt and reads UTF-8 chunks back.
@interface GemmaBridge : NSObject

- (instancetype)init;

/// Load the .litertlm bundle and warm the LiteRT-LM engine.
/// backend:  @"cpu" or @"gpu".
/// cacheDir: writable dir for the XNNPACK kernel cache. On iOS pass the
///           app's Caches/ path - the bundle itself is read-only so without
///           this every cold launch redoes ~1-2 s of kernel selection work.
///           Pass nil/empty to use LiteRT-LM's default (next to the model).
/// Returns NO on failure (-lastError for details).
- (BOOL)loadWithModelPath:(NSString *)modelPath
                  backend:(NSString *)backend
                 cacheDir:(nullable NSString *)cacheDir
                benchmark:(BOOL)benchmark
    NS_SWIFT_NAME(load(modelPath:backend:cacheDir:benchmark:));

/// Engine-measured prefill/decode throughput from the most recent generation.
/// Keys: "ttftMs", "prefillTps", "decodeTps", "prefillTokens", "decodeTokens".
/// Returns nil if benchmark instrumentation wasn't enabled at load.
- (nullable NSDictionary<NSString *, NSNumber *> *)lastBenchmark;

/// Blocking generation. Returns the full reply, or nil on error.
- (nullable NSString *)generatePrompt:(NSString *)prompt
                            maxTokens:(int)maxTokens
                          temperature:(float)temperature
                                 topK:(int)topK
                                 topP:(float)topP
    NS_SWIFT_NAME(generate(prompt:maxTokens:temperature:topK:topP:));

/// Streaming generation. `onChunk` fires on the same thread the engine
/// callback runs on (not necessarily the main queue) — marshal to main
/// before touching UI. The final invocation has done=YES (and may carry
/// an empty chunk).
- (void)streamPrompt:(NSString *)prompt
           maxTokens:(int)maxTokens
         temperature:(float)temperature
                topK:(int)topK
                topP:(float)topP
             onChunk:(void (^)(NSString *chunk, BOOL done))onChunk
    NS_SWIFT_NAME(stream(prompt:maxTokens:temperature:topK:topP:onChunk:));

/// Drop multi-turn conversation history. Call this before each
/// independent translation so the engine sees a fresh session.
- (void)resetSession;

@property (nonatomic, readonly, copy) NSString *lastError;

@end

NS_ASSUME_NONNULL_END
