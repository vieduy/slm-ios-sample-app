#import "LlamaBridge.h"

#include <algorithm>
#include <string>
#include <thread>
#include <vector>
#include <sys/sysctl.h>

#if __has_include(<llama/llama.h>)
#import <llama/llama.h>          // Xcode: resolved from the embedded llama.xcframework
#import <llama/ggml-backend.h>   // ggml_backend_dev_* (CPU-only device selection)
#else
#include "llama.h"               // standalone syntax check: -I .../Headers
#include "ggml-backend.h"
#endif

// One-time global backend init (ggml/Metal registration).
static void EnsureBackend() {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        llama_backend_init();
    });
}

// Probe the device's CPU ISA features so we know whether enabling hardware SDOT
// (-march=...+dotprod) in the llama.cpp build would help or SIGILL-crash here.
static int SysctlInt(const char *name) {
    int v = 0; size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, nullptr, 0) != 0) return -1;
    return v;
}
static void LogCpuFeatures() {
    fprintf(stderr,
        "[[CPUINFO]] DotProd=%d I8MM=%d FP16=%d BF16=%d cores=%d perf_cores=%d\n",
        SysctlInt("hw.optional.arm.FEAT_DotProd"),
        SysctlInt("hw.optional.arm.FEAT_I8MM"),
        SysctlInt("hw.optional.arm.FEAT_FP16"),
        SysctlInt("hw.optional.arm.FEAT_BF16"),
        SysctlInt("hw.logicalcpu"),
        SysctlInt("hw.perflevel0.logicalcpu"));
}

@implementation LlamaBridge {
    llama_model       *_model;
    llama_context     *_ctx;
    const llama_vocab *_vocab;
    std::string        _chatTmpl;   // model's embedded (qwen3) chat template
    std::string        _lastError;
    bool               _benchmark;
    bool               _isThinking; // qwen3/Bonsai-style reasoning model
    int                _nThreads;
}

- (instancetype)init {
    if ((self = [super init])) {
        _model = nullptr;
        _ctx   = nullptr;
        _vocab = nullptr;
        _benchmark = false;
        _isThinking = false;
        // Decode is compute-bound on the generic Q1_0 ARM kernel and scales with
        // cores (measured on A12: 2 thr = 4.7 tok/s < 4 thr = 5.5), so use all of
        // them. Cap at 8 to avoid oversubscription on bigger SoCs.
        unsigned hw = std::thread::hardware_concurrency();
        _nThreads = (int)std::max(1u, std::min(8u, hw ? hw : 4u));
    }
    return self;
}

- (void)dealloc {
    if (_ctx)   llama_free(_ctx);
    if (_model) llama_model_free(_model);
}

- (BOOL)loadWithModelPath:(NSString *)modelPath
                  backend:(NSString *)backend
                 cacheDir:(NSString *)cacheDir
                benchmark:(BOOL)benchmark {
    (void)cacheDir;  // llama.cpp has no external kernel cache.
    EnsureBackend();
    LogCpuFeatures();
    _benchmark = benchmark;
    _lastError.clear();

    const bool wantGpu = [backend isEqualToString:@"gpu"];
    const bool canGpu  = llama_supports_gpu_offload();
    // Force the GPU attempt when requested (don't silently gate on canGpu): if
    // Metal truly can't load, the model load fails and the VM falls back to CPU
    // *visibly*. This makes "GPU" actually try Metal so we can measure it.
    const bool useGpu  = wantGpu;
    fprintf(stderr, "[[LLAMA]] reqBackend=%s wantGpu=%d canGpu=%d useGpu=%d\n",
            backend.UTF8String, wantGpu, canGpu, useGpu);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = useGpu ? 999 : 0;   // 999 = offload all transformer layers
    mp.use_mmap     = true;               // mmap weights (keeps phys_footprint low)

    // CPU mode: use CPU + accelerator (BLAS/Accelerate) devices, but exclude the
    // GPU (Metal). On the A12, Metal offload bounced decode ops between backends
    // (~395 graph splits/token) and forced a one-time ~14 s Metal shader compile,
    // yet BLAS meaningfully speeds up the batched *prefill* matmul. So keep BLAS,
    // drop Metal. (devs must stay alive through llama_model_load_from_file below.)
    std::vector<ggml_backend_dev_t> devs;
    if (!useGpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            auto t = ggml_backend_dev_type(d);
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU)
                continue;  // skip Metal
            devs.push_back(d);
        }
        devs.push_back(nullptr);  // NULL-terminate
        mp.devices = devs.data();
    }

    _model = llama_model_load_from_file([modelPath UTF8String], mp);
    if (!_model) {
        _lastError = "llama_model_load_from_file failed";
        return NO;
    }

    llama_context_params cp = llama_context_default_params();
    // iPhone XS has 4 GB RAM; the full 32k context KV cache (~3.7 GB for this
    // qwen3-1.7B) won't fit. Cap to 1024 — KV cache ~112 MiB.
    cp.n_ctx           = 1024;
    cp.n_batch         = 512;
    cp.n_threads       = _nThreads;   // decode  (compute-bound → use all cores)
    cp.n_threads_batch = _nThreads;   // prefill (BLAS handles the heavy matmul)
    cp.no_perf         = false;     // enable prefill/decode timing for lastBenchmark

    _ctx = llama_init_from_model(_model, cp);
    if (!_ctx) {
        _lastError = "llama_init_from_model failed";
        llama_model_free(_model);
        _model = nullptr;
        return NO;
    }

    _vocab = llama_model_get_vocab(_model);
    const char *tmpl = llama_model_chat_template(_model, nullptr);
    if (tmpl) _chatTmpl = tmpl;
    // Reasoning models (Qwen3/Bonsai) wrap answers in <think>...</think>. Their
    // jinja template ends the generation prompt with an *empty* think block to
    // skip reasoning, but llama_chat_apply_template (no jinja) drops it — so the
    // model reasons anyway. Detect such models so [self tokenize:] can re-add it.
    _isThinking = _chatTmpl.find("<think>") != std::string::npos;
    return YES;
}

// Wrap a single user turn in the model's chat template, then tokenize.
- (std::vector<llama_token>)tokenize:(NSString *)prompt {
    std::string user = prompt.UTF8String ?: "";
    std::string formatted;

    if (!_chatTmpl.empty()) {
        llama_chat_message msg{ "user", user.c_str() };
        std::vector<char> buf(user.size() * 2 + 512);
        int n = llama_chat_apply_template(_chatTmpl.c_str(), &msg, 1,
                                          /*add_ass=*/true,
                                          buf.data(), (int)buf.size());
        if (n > (int)buf.size()) {                 // buffer too small — grow once
            buf.resize(n);
            n = llama_chat_apply_template(_chatTmpl.c_str(), &msg, 1, true,
                                          buf.data(), (int)buf.size());
        }
        if (n > 0) formatted.assign(buf.data(), (size_t)n);
    }
    if (formatted.empty()) formatted = user;        // no template → raw prompt

    // Disable reasoning: replicate the template's no-think behavior by closing
    // an empty think block right after the assistant header. Idempotent — skip
    // if a </think> is already present (e.g. llama.cpp added it).
    if (_isThinking && formatted.find("</think>") == std::string::npos) {
        formatted += "<think>\n\n</think>\n\n";
    }

    int need = -llama_tokenize(_vocab, formatted.c_str(), (int)formatted.size(),
                               nullptr, 0, /*add_special=*/true,
                               /*parse_special=*/true);
    std::vector<llama_token> toks(need > 0 ? need : 0);
    int n = llama_tokenize(_vocab, formatted.c_str(), (int)formatted.size(),
                           toks.data(), (int)toks.size(), true, true);
    if (n < 0) toks.clear(); else toks.resize((size_t)n);
    return toks;
}

// Shared decode loop for both blocking and streaming generation.
- (std::string)run:(NSString *)prompt
         maxTokens:(int)maxTokens
              temp:(float)temp
              topK:(int)topK
              topP:(float)topP
           onChunk:(void (^ _Nullable)(const std::string &, bool))onChunk {
    if (!_ctx) { _lastError = "engine not loaded"; return ""; }

    // Stateless per call: clear KV and reset perf so lastBenchmark reflects
    // only this generation (mirrors GemmaEngine's per-call conversation).
    llama_memory_clear(llama_get_memory(_ctx), /*data=*/true);
    llama_perf_context_reset(_ctx);
    _lastError.clear();

    std::vector<llama_token> prompt_tokens = [self tokenize:prompt];
    if (prompt_tokens.empty()) {
        _lastError = "tokenize failed";
        if (onChunk) onChunk("", true);
        return "";
    }

    // Sampler chain: greedy when temp<=0, else top_k -> top_p -> temp -> dist.
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler *smpl = llama_sampler_chain_init(sp);
    if (temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (topK > 0)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topK));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    std::string full;
    char piece[256];
    llama_token cur = 0;
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(),
                                            (int)prompt_tokens.size());
    int generated = 0;
    while (generated < maxTokens) {
        int rc = llama_decode(_ctx, batch);
        if (rc != 0) {
            _lastError = "llama_decode failed (rc=" + std::to_string(rc) + ")";
            break;
        }
        cur = llama_sampler_sample(smpl, _ctx, -1);
        if (llama_vocab_is_eog(_vocab, cur)) break;

        int np = llama_token_to_piece(_vocab, cur, piece, sizeof(piece),
                                      /*lstrip=*/0, /*special=*/false);
        if (np > 0) {
            std::string chunk(piece, (size_t)np);
            full += chunk;
            if (onChunk) onChunk(chunk, false);
        }
        ++generated;
        batch = llama_batch_get_one(&cur, 1);   // &cur stays valid next decode
    }

    if (onChunk) onChunk("", true);
    llama_sampler_free(smpl);
    return full;
}

- (NSDictionary<NSString *, NSNumber *> *)lastBenchmark {
    if (!_ctx || !_benchmark) return nil;
    llama_perf_context_data d = llama_perf_context(_ctx);
    double prefillTps = d.t_p_eval_ms > 0 ? d.n_p_eval / (d.t_p_eval_ms / 1000.0) : 0;
    double decodeTps  = d.t_eval_ms   > 0 ? d.n_eval   / (d.t_eval_ms   / 1000.0) : 0;
    return @{
        @"ttftMs":        @(d.t_p_eval_ms),          // prefill time ≈ time-to-first-token
        @"prefillTps":    @(prefillTps),
        @"decodeTps":     @(decodeTps),
        @"prefillTokens": @(d.n_p_eval),
        @"decodeTokens":  @(d.n_eval),
    };
}

- (NSString *)generatePrompt:(NSString *)prompt
                   maxTokens:(int)maxTokens
                 temperature:(float)temperature
                        topK:(int)topK
                        topP:(float)topP {
    std::string out = [self run:prompt maxTokens:maxTokens temp:temperature
                           topK:topK topP:topP onChunk:nil];
    if (!_lastError.empty()) return nil;
    return [NSString stringWithUTF8String:out.c_str()];
}

- (void)streamPrompt:(NSString *)prompt
           maxTokens:(int)maxTokens
         temperature:(float)temperature
                topK:(int)topK
                topP:(float)topP
             onChunk:(void (^)(NSString *, BOOL))onChunk {
    [self run:prompt maxTokens:maxTokens temp:temperature topK:topK topP:topP
      onChunk:[onChunk](const std::string &chunk, bool done) {
          NSString *s = [NSString stringWithUTF8String:chunk.c_str()];
          onChunk(s ?: @"", done ? YES : NO);
      }];
}

- (void)resetSession {
    if (_ctx) llama_memory_clear(llama_get_memory(_ctx), /*data=*/true);
}

- (NSString *)lastError {
    return _lastError.empty() ? @"" : [NSString stringWithUTF8String:_lastError.c_str()];
}

@end
