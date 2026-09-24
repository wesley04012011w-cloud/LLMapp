#include <jni.h>
#include <android/log.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <atomic>
#include <algorithm>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <llama.h>

static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static llama_sampler *g_sampler = nullptr;
static std::vector<llama_token> g_cached_tokens;
static std::vector<std::pair<std::string, std::string>> g_history;
static std::mutex g_mutex;
static std::atomic_bool g_stop{false};
static bool g_backend_ready=false;
static float g_temperature=0.7f;
static float g_top_p=0.95f;
static int g_top_k=40;
static float g_min_p=0.05f;
static int g_max_tokens=2048;
static std::string g_system_prompt;
static bool g_use_jinja=true;
static std::mutex g_log_mutex;
static std::string g_current_log;

static void append_file(const char *path,const char *data,size_t len){
    if(!path||!*path||!data||!len)return;
    int fd=open(path,O_WRONLY|O_CREAT|O_APPEND,0600);
    if(fd<0)return;
    size_t off=0;
    while(off<len){ssize_t n=write(fd,data+off,len-off);if(n<=0)break;off+=(size_t)n;}
    close(fd);
}
static void native_log(const char *fmt,...){
    char buf[2048];
    va_list ap;va_start(ap,fmt);
    int n=vsnprintf(buf,sizeof(buf),fmt,ap);
    va_end(ap);
    if(n<0)return;
    size_t len=(size_t)std::min(n,(int)sizeof(buf)-1);
    __android_log_write(ANDROID_LOG_ERROR,"LLMapp",buf);
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if(!g_current_log.empty()){append_file(g_current_log.c_str(),buf,len);append_file(g_current_log.c_str(),"\n",1);}
}
static void llama_log_bridge(enum ggml_log_level level,const char *text,void *){
    if(!text)return;
    if(level>=GGML_LOG_LEVEL_WARN){
        native_log("[llama] %s",text);
    }
}

static void crash_handler(int sig){
    if(!g_current_log.empty()){
        const char *msg="\n========== NATIVE CRASH ==========\\nsignal received; process crashed in native code.\\n";
        append_file(g_current_log.c_str(),msg,strlen(msg));
    }
    signal(sig,SIG_DFL);raise(sig);
}
static void install_crash_handlers(){
    static std::once_flag once;
    std::call_once(once,[]{signal(SIGSEGV,crash_handler);signal(SIGABRT,crash_handler);signal(SIGBUS,crash_handler);signal(SIGILL,crash_handler);signal(SIGFPE,crash_handler);});
}

static void clear_engine() {
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_cached_tokens.clear();
    g_history.clear();
}

static bool rebuild_sampler() {
    if (!g_model) return false;
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sp);
    if (!g_sampler) return false;
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(g_top_k));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(g_top_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_min_p(g_min_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(g_temperature));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &s, bool add_special) {
    int n = llama_tokenize(vocab, s.data(), (int)s.size(), nullptr, 0, add_special, true);
    if (n < 0) n = -n;
    if (n <= 0) return {};
    std::vector<llama_token> tokens((size_t)n);
    if (llama_tokenize(vocab, s.data(), (int)s.size(), tokens.data(), n, add_special, true) < 0) return {};
    return tokens;
}

static std::string format_chat() {
    const char *tmpl = g_use_jinja ? llama_model_chat_template(g_model, nullptr) : nullptr;

    // Some recent GGUF templates (notably Qwen3.5) are full Jinja templates.
    // The low-level C helper may reject them even though the model carries a
    // valid template in metadata.  Fall back to their standard im_start form
    // instead of sending a plain "User:/Assistant:" prompt to the model.
    if (tmpl && std::strstr(tmpl, "<|im_start|>")) {
        std::string fallback;
        fallback.reserve(1024);
        if (!g_system_prompt.empty()) {
            fallback += "<|im_start|>system\n";
            fallback += g_system_prompt;
            fallback += "<|im_end|>\n";
        }
        for (const auto &m : g_history) {
            fallback += "<|im_start|>";
            fallback += m.first;
            fallback += "\n";
            fallback += m.second;
            fallback += "<|im_end|>\n";
        }
        fallback += "<|im_start|>assistant\n";
        if (std::strstr(tmpl, "<think>")) {
            fallback += "<think>\n";
        }
        return fallback;
    }

    std::string fallback;
    fallback.reserve(1024);
    for (const auto &m : g_history) {
        fallback += m.first == "user" ? "User: " : "Assistant: ";
        fallback += m.second;
        fallback += "\n";
    }
    fallback += "Assistant: ";

    if (!tmpl) return fallback;

    std::vector<llama_chat_message> msgs;
    msgs.reserve(g_history.size() + (g_system_prompt.empty() ? 0 : 1));
    if (!g_system_prompt.empty()) msgs.push_back({"system", g_system_prompt.c_str()});
    for (auto &m : g_history) msgs.push_back({m.first.c_str(), m.second.c_str()});

    int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, nullptr, 0);
    if (n < 0) return fallback;
    std::string out((size_t)n, '\0');
    if (llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, out.data(), out.size()) < 0) return fallback;
    return out;
}

// Converts arbitrary UTF-8 token bytes to UTF-16 safely. llama.cpp token pieces
// can contain incomplete UTF-8 sequences, which must never be passed to NewStringUTF.
static jstring utf8_to_jstring(JNIEnv *env, const char *data, size_t len) {
    std::vector<jchar> out;
    out.reserve(len);
    size_t i = 0;
    while (i < len) {
        const unsigned char c = (unsigned char)data[i];
        uint32_t cp = 0;
        size_t need = 0;
        if (c < 0x80) { cp = c; need = 1; }
        else if (c >= 0xC2 && c <= 0xDF) { cp = c & 0x1F; need = 2; }
        else if (c >= 0xE0 && c <= 0xEF) { cp = c & 0x0F; need = 3; }
        else if (c >= 0xF0 && c <= 0xF4) { cp = c & 0x07; need = 4; }
        else { cp = 0xFFFD; need = 1; }

        if (need > 1) {
            if (i + need > len) { cp = 0xFFFD; need = 1; }
            else {
                bool valid = true;
                for (size_t j = 1; j < need; ++j) {
                    unsigned char cc = (unsigned char)data[i + j];
                    if ((cc & 0xC0) != 0x80) { valid = false; break; }
                }
                if (valid) {
                    if (need == 2) cp = (cp << 6) | ((unsigned char)data[i + 1] & 0x3F);
                    else if (need == 3) cp = (cp << 12) | (((unsigned char)data[i + 1] & 0x3F) << 6) | ((unsigned char)data[i + 2] & 0x3F);
                    else cp = (cp << 18) | (((unsigned char)data[i + 1] & 0x3F) << 12) |
                              (((unsigned char)data[i + 2] & 0x3F) << 6) | ((unsigned char)data[i + 3] & 0x3F);
                    if ((need == 2 && cp < 0x80) || (need == 3 && cp < 0x800) ||
                        (need == 4 && (cp < 0x10000 || cp > 0x10FFFF)) ||
                        (cp >= 0xD800 && cp <= 0xDFFF)) valid = false;
                }
                if (!valid) { cp = 0xFFFD; need = 1; }
            }
        }

        if (cp <= 0xFFFF) out.push_back((jchar)cp);
        else {
            cp -= 0x10000;
            out.push_back((jchar)(0xD800 | (cp >> 10)));
            out.push_back((jchar)(0xDC00 | (cp & 0x3FF)));
        }
        i += need;
    }
    return env->NewString(out.data(), (jsize)out.size());
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmapp_Engine_initLogs(JNIEnv* env,jobject,jstring jdir){
    if(!jdir)return;
    const char *p=env->GetStringUTFChars(jdir,nullptr); if(!p)return;
    {std::lock_guard<std::mutex> lock(g_log_mutex); g_current_log=std::string(p)+"/startup.txt";}
    env->ReleaseStringUTFChars(jdir,p);
    install_crash_handlers();
    llama_log_set(llama_log_bridge,nullptr);
    if(!g_backend_ready){
        llama_backend_init();
        g_backend_ready=true;
        native_log("[startup] llama backend initialized");
    }
    native_log("[startup] native logger initialized");
}
extern "C" JNIEXPORT void JNICALL
Java_com_llmapp_Engine_startEntryLog(JNIEnv* env,jobject,jstring jpath){
    if(!jpath)return;
    const char *p=env->GetStringUTFChars(jpath,nullptr); if(!p)return;
    {std::lock_guard<std::mutex> lock(g_log_mutex); g_current_log=p;}
    env->ReleaseStringUTFChars(jpath,p);
    native_log("========== NEW ENTRY ==========");
}
extern "C" JNIEXPORT void JNICALL
Java_com_llmapp_Engine_log(JNIEnv* env,jobject,jstring jmsg){
    if(!jmsg)return;
    const char *p=env->GetStringUTFChars(jmsg,nullptr); if(!p)return;
    native_log("%s",p); env->ReleaseStringUTFChars(jmsg,p);
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmapp_Engine_setGenerationSettings(JNIEnv *env, jobject, jfloat temperature, jfloat topP, jint topK, jfloat minP, jint maxTokens, jstring jsystem, jboolean useJinja) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_temperature = std::max(0.0f, std::min(2.0f, (float)temperature));
    g_top_p = std::max(0.0f, std::min(1.0f, (float)topP));
    g_top_k = std::max(0, std::min(100, (int)topK));
    g_min_p = std::max(0.0f, std::min(1.0f, (float)minP));
    g_max_tokens = std::max(128, std::min(4096, (int)maxTokens));
    g_use_jinja = useJinja;
    g_system_prompt.clear();
    if (jsystem) {
        const char *p = env->GetStringUTFChars(jsystem, nullptr);
        if (p) {
            g_system_prompt = p;
            env->ReleaseStringUTFChars(jsystem, p);
        }
    }
    if (g_model && g_ctx && !rebuild_sampler()) {
        native_log("[settings] sampler rebuild failed");
    }
    native_log("[settings] temp=%.3f top_p=%.3f top_k=%d min_p=%.3f max_tokens=%d jinja=%d system_bytes=%zu",
               g_temperature, g_top_p, g_top_k, g_min_p, g_max_tokens, g_use_jinja ? 1 : 0, g_system_prompt.size());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_llmapp_Engine_loadModel(JNIEnv *env, jobject, jstring jpath) {
    std::lock_guard<std::mutex> lock(g_mutex);
    native_log("[load] begin");
    clear_engine();
    if (!g_backend_ready) {
        native_log("[load] backend was not initialized");
        return JNI_FALSE;
    }
    if (!jpath) {
        native_log("[load] null model path");
        return JNI_FALSE;
    }

    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) {
        native_log("[load] GetStringUTFChars failed");
        return JNI_FALSE;
    }
    native_log("[load] model path=%s", path);

    llama_model_params mp = llama_model_default_params();
    g_model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!g_model) {
        native_log("[load] llama_model_load_from_file returned null");
        return JNI_FALSE;
    }
    native_log("[load] model loaded");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned threads = std::max(1u, std::min(hw ? hw : 1u, 8u));
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    native_log("[load] context n_ctx=%u n_batch=%u n_ubatch=%u threads=%u",cp.n_ctx,cp.n_batch,cp.n_ubatch,threads);
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) {
        native_log("[load] llama_init_from_model returned null");
        clear_engine();
        return JNI_FALSE;
    }
    native_log("[load] context initialized");

    if (!rebuild_sampler()) {
        native_log("[load] sampler initialization failed");
        clear_engine();
        return JNI_FALSE;
    }

    // Warm up the freshly-created context once.  This activates the model's
    // compute paths before the first real prompt, then clears the temporary
    // memory so the first user turn starts from a clean context.
    const llama_vocab *warm_vocab = llama_model_get_vocab(g_model);
    const llama_token warm_token = warm_vocab ? llama_vocab_bos(warm_vocab) : LLAMA_TOKEN_NULL;
    if (warm_vocab && warm_token != LLAMA_TOKEN_NULL) {
        llama_batch warm = llama_batch_init(1, 0, 1);
        if (warm.token && warm.pos && warm.n_seq_id && warm.seq_id && warm.logits) {
            warm.n_tokens = 1;
            warm.token[0] = warm_token;
            warm.pos[0] = 0;
            warm.n_seq_id[0] = 1;
            warm.seq_id[0][0] = 0;
            warm.logits[0] = false;
            llama_set_warmup(g_ctx, true);
            const int warm_rc = llama_decode(g_ctx, warm);
            llama_set_warmup(g_ctx, false);
            llama_synchronize(g_ctx);
            llama_batch_free(warm);
            llama_memory_clear(llama_get_memory(g_ctx), true);
            native_log("[load] warmup rc=%d; context cleared", warm_rc);
            if (warm_rc != 0) {
                native_log("[load] warmup decode failed");
                clear_engine();
                return JNI_FALSE;
            }
        } else {
            llama_batch_free(warm);
            native_log("[load] warmup batch allocation failed");
            clear_engine();
            return JNI_FALSE;
        }
    } else {
        native_log("[load] no BOS token available; skipping warmup");
    }

    native_log("[load] sampler initialized; model ready");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_llmapp_Engine_generate(JNIEnv *env, jobject, jstring jprompt, jobject callback) {
    std::lock_guard<std::mutex> lock(g_mutex);
    native_log("[generate] begin");
    if (!g_model || !g_ctx || !g_sampler || !jprompt || !callback) {
        native_log("[generate] invalid engine/callback state");
        return JNI_FALSE;
    }

    g_stop = false;
    const char *p = env->GetStringUTFChars(jprompt, nullptr);
    if (!p) return JNI_FALSE;
    std::string user(p);
    env->ReleaseStringUTFChars(jprompt, p);
    native_log("[generate] prompt bytes=%zu history_before=%zu",user.size(),g_history.size());
    g_history.push_back({"user", user});

    const std::string formatted = format_chat();
    native_log("[generate] formatted prompt bytes=%zu",formatted.size());
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    const auto tokens = tokenize(vocab, formatted, true);
    native_log("[generate] prompt tokens=%zu cached=%zu",tokens.size(),g_cached_tokens.size());
    if (tokens.empty()) {
        native_log("[generate] tokenization returned empty");
        g_history.pop_back();
        return JNI_FALSE;
    }

    size_t common = 0;
    while (common < g_cached_tokens.size() && common < tokens.size() && g_cached_tokens[common] == tokens[common]) ++common;
    native_log("[generate] common prefix=%zu",common);
    if (common < g_cached_tokens.size()) {
        llama_memory_seq_rm(llama_get_memory(g_ctx), 0, (llama_pos)common, -1);
        g_cached_tokens.resize(common);
    }

    llama_sampler_reset(g_sampler);
    llama_batch batch = llama_batch_init(512, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        g_history.pop_back();
        return JNI_FALSE;
    }

    for (size_t i = common; i < tokens.size(); ++i) {
        const int j = batch.n_tokens++;
        batch.token[j] = tokens[i];
        batch.pos[j] = (llama_pos)i;
        batch.n_seq_id[j] = 1;
        batch.seq_id[j][0] = 0;
        batch.logits[j] = (i + 1 == tokens.size());
        if (batch.n_tokens == 512 || i + 1 == tokens.size()) {
            native_log("[generate] decode prompt batch=%d",batch.n_tokens);
            if (llama_decode(g_ctx, batch) != 0) {
                native_log("[generate] prompt decode failed");
                llama_batch_free(batch);
                g_history.pop_back();
                return JNI_FALSE;
            }
            batch.n_tokens = 0;
        }
    }
    llama_batch_free(batch);
    g_cached_tokens = tokens;

    native_log("[generate] prompt decode complete");
    jclass cb_cls = env->GetObjectClass(callback);
    if (!cb_cls) { g_history.pop_back(); return JNI_FALSE; }
    jmethodID on_token = env->GetMethodID(cb_cls, "onToken", "(Ljava/lang/String;)V");
    env->DeleteLocalRef(cb_cls);
    if (!on_token || env->ExceptionCheck()) {
        native_log("[generate] callback method lookup failed");
        env->ExceptionClear();
        g_history.pop_back();
        return JNI_FALSE;
    }

    // Qwen3.5-style templates can place the opening <think> tag in the
    // generation prompt itself, so that tag never appears in sampled output.
    // Notify the UI about the already-open reasoning section without changing
    // the model input or generated text.
    size_t tail = formatted.size();
    while (tail > 0) {
        const unsigned char c = (unsigned char)formatted[tail - 1];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') --tail;
        else break;
    }
    const bool thinking_prefilled = tail >= 7 &&
        formatted.compare(tail - 7, 7, "<think>") == 0;
    if (thinking_prefilled) {
        jstring js = env->NewStringUTF("<think>");
        if (js) {
            env->CallVoidMethod(callback, on_token, js);
            env->DeleteLocalRef(js);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                g_stop = true;
                g_history.pop_back();
                return JNI_FALSE;
            }
        }
        native_log("[generate] reasoning prefilled by chat template");
    }

    std::string answer;
    answer.reserve(256);

    native_log("[generate] sampling loop start");
    for (int step = 0; step < g_max_tokens && !g_stop; ++step) {
        const llama_token tok = llama_sampler_sample(g_sampler, g_ctx, -1);
        if(step==0) native_log("[generate] first token sampled=%d",tok);
        if (llama_vocab_is_eog(vocab, tok)) {
            native_log("[generate] EOG at step=%d",step);
            break;
        }
        llama_sampler_accept(g_sampler, tok);

        char piece[16384];
        int n = llama_token_to_piece(vocab, tok, piece, (int)sizeof(piece), 0, true);
        if (n < 0) {
            std::vector<char> larger((size_t)(-n) + 1);
            n = llama_token_to_piece(vocab, tok, larger.data(), (int)larger.size(), 0, true);
            if (n > 0) {
                answer.append(larger.data(), (size_t)n);
                jstring js = utf8_to_jstring(env, larger.data(), (size_t)n);
                if (!js) { g_stop = true; break; }
                env->CallVoidMethod(callback, on_token, js);
                env->DeleteLocalRef(js);
            }
        } else if (n > 0) {
            answer.append(piece, (size_t)n);
            jstring js = utf8_to_jstring(env, piece, (size_t)n);
            if (!js) { g_stop = true; break; }
            env->CallVoidMethod(callback, on_token, js);
            env->DeleteLocalRef(js);
        }

        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            g_stop = true;
            break;
        }

        if (step>0 && step%100==0) native_log("[generate] generated steps=%d answer_bytes=%zu cache=%zu",step,answer.size(),g_cached_tokens.size());
        if (g_cached_tokens.size() >= 4095) {
            g_stop = true;
            break;
        }

        llama_batch next = llama_batch_init(1, 0, 1);
        if (!next.token || !next.pos || !next.n_seq_id || !next.seq_id || !next.logits) {
            llama_batch_free(next);
            g_stop = true;
            break;
        }
        next.n_tokens = 1;
        next.token[0] = tok;
        next.pos[0] = (llama_pos)g_cached_tokens.size();
        next.n_seq_id[0] = 1;
        next.seq_id[0][0] = 0;
        next.logits[0] = true;
        const int decode_rc = llama_decode(g_ctx, next);
        llama_batch_free(next);
        if (decode_rc != 0) {
            g_stop = true;
            break;
        }
        g_cached_tokens.push_back(tok);
    }

    native_log("[generate] loop finished stop=%d answer_bytes=%zu cache=%zu",g_stop.load(),answer.size(),g_cached_tokens.size());
    if (g_stop) {
        g_history.pop_back();
        return JNI_FALSE;
    }
    g_history.push_back({"assistant", answer});
    native_log("[generate] success answer_bytes=%zu history=%zu",answer.size(),g_history.size());
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmapp_Engine_stop(JNIEnv *, jobject) { g_stop = true; }

extern "C" JNIEXPORT void JNICALL
Java_com_llmapp_Engine_unload(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    clear_engine();
}
