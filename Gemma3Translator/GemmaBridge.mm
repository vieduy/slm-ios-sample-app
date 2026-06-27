#import "GemmaBridge.h"

#include <memory>

#include "gemma_engine.h"

@implementation GemmaBridge {
    std::unique_ptr<gemma::GemmaEngine> _eng;
}

- (instancetype)init {
    if ((self = [super init])) {
        _eng = std::make_unique<gemma::GemmaEngine>();
    }
    return self;
}

- (BOOL)loadWithModelPath:(NSString *)modelPath
                  backend:(NSString *)backend
                 cacheDir:(NSString *)cacheDir
                benchmark:(BOOL)benchmark {
    std::string cd = cacheDir ? std::string([cacheDir UTF8String]) : std::string{};
    bool ok = _eng->Init([modelPath UTF8String], [backend UTF8String], cd,
                         benchmark ? true : false);
    return ok ? YES : NO;
}

- (NSDictionary<NSString *, NSNumber *> *)lastBenchmark {
    gemma::GemmaEngine::BenchStats s;
    if (!_eng->last_benchmark(s)) return nil;
    return @{
        @"ttftMs":        @(s.ttft_s * 1000.0),
        @"prefillTps":    @(s.prefill_tps),
        @"decodeTps":     @(s.decode_tps),
        @"prefillTokens": @(s.prefill_tokens),
        @"decodeTokens":  @(s.decode_tokens),
    };
}

static gemma::GenerationConfig MakeCfg(int n, float t, int k, float p) {
    gemma::GenerationConfig c;
    c.max_new_tokens = n;
    c.temperature    = t;
    c.top_k          = k;
    c.top_p          = p;
    return c;
}

- (nullable NSString *)generatePrompt:(NSString *)prompt
                            maxTokens:(int)maxTokens
                          temperature:(float)temperature
                                 topK:(int)topK
                                 topP:(float)topP {
    auto cfg = MakeCfg(maxTokens, temperature, topK, topP);
    std::string out = _eng->Generate([prompt UTF8String], cfg);
    if (!_eng->last_error().empty()) {
        return nil;
    }
    return [NSString stringWithUTF8String:out.c_str()];
}

- (void)streamPrompt:(NSString *)prompt
           maxTokens:(int)maxTokens
         temperature:(float)temperature
                topK:(int)topK
                topP:(float)topP
             onChunk:(void (^)(NSString *, BOOL))onChunk {
    auto cfg = MakeCfg(maxTokens, temperature, topK, topP);
    _eng->GenerateStream(
        [prompt UTF8String], cfg,
        [onChunk](const std::string &chunk, bool done) {
            NSString *s = [NSString stringWithUTF8String:chunk.c_str()];
            onChunk(s ?: @"", done ? YES : NO);
        });
}

- (void)resetSession {
    _eng->ResetSession();
}

- (NSString *)lastError {
    const std::string &e = _eng->last_error();
    return e.empty() ? @"" : [NSString stringWithUTF8String:e.c_str()];
}

@end
