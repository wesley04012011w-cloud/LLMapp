#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <llama.h>

static llama_model * g_model=nullptr;
static llama_context * g_ctx=nullptr;
static llama_sampler * g_sampler=nullptr;
static std::vector<llama_token> g_cached_tokens;
static std::vector<std::pair<std::string,std::string>> g_history;
static std::mutex g_mutex;
static std::atomic_bool g_stop{false};

static void clear_engine(){
    if(g_sampler){llama_sampler_free(g_sampler);g_sampler=nullptr;}
    if(g_ctx){llama_free(g_ctx);g_ctx=nullptr;}
    if(g_model){llama_model_free(g_model);g_model=nullptr;}
    g_cached_tokens.clear();
    g_history.clear();
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab,const std::string &s,bool add_special){
    int n=llama_tokenize(vocab,s.data(),(int)s.size(),nullptr,0,add_special,true);
    if(n<0)n=-n;
    if(n<=0)return {};
    std::vector<llama_token> tokens(n);
    if(llama_tokenize(vocab,s.data(),(int)s.size(),tokens.data(),n,add_special,true)<0)return {};
    return tokens;
}

static std::string format_chat(){
    std::string fallback;
    fallback.reserve(1024);
    for(const auto &m:g_history){
        fallback += m.first=="user" ? "User: " : "Assistant: ";
        fallback += m.second;
        fallback += "\n";
    }
    fallback += "Assistant: ";

    const char * tmpl=llama_model_chat_template(g_model,nullptr);
    if(!tmpl)return fallback;

    std::vector<llama_chat_message> msgs;
    msgs.reserve(g_history.size());
    for(auto &m:g_history)msgs.push_back({m.first.c_str(),m.second.c_str()});

    int n=llama_chat_apply_template(tmpl,msgs.data(),msgs.size(),true,nullptr,0);
    if(n<0)return fallback;
    std::string out((size_t)n,'\0');
    if(llama_chat_apply_template(tmpl,msgs.data(),msgs.size(),true,out.data(),out.size())<0)return fallback;
    return out;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_llmapp_Engine_loadModel(JNIEnv* env,jobject,jstring jpath){
    std::lock_guard<std::mutex> lock(g_mutex);
    clear_engine();
    const char * path=env->GetStringUTFChars(jpath,nullptr);

    llama_model_params mp=llama_model_default_params();
    g_model=llama_model_load_from_file(path,mp);
    env->ReleaseStringUTFChars(jpath,path);
    if(!g_model)return JNI_FALSE;

    llama_context_params cp=llama_context_default_params();
    cp.n_ctx=4096;
    cp.n_batch=512;
    cp.n_ubatch=512;
    const unsigned hw=std::thread::hardware_concurrency();
    cp.n_threads=std::max(1u,hw);
    cp.n_threads_batch=std::max(1u,hw);
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_AUTO;

    g_ctx=llama_init_from_model(g_model,cp);
    if(!g_ctx){clear_engine();return JNI_FALSE;}

    llama_sampler_chain_params sp=llama_sampler_chain_default_params();
    g_sampler=llama_sampler_chain_init(sp);
    llama_sampler_chain_add(g_sampler,llama_sampler_init_top_k(40));
    llama_sampler_chain_add(g_sampler,llama_sampler_init_top_p(0.95f,1));
    llama_sampler_chain_add(g_sampler,llama_sampler_init_temp(0.7f));
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_llmapp_Engine_generate(JNIEnv* env,jobject,jstring jprompt,jobject callback){
    std::lock_guard<std::mutex> lock(g_mutex);
    if(!g_model||!g_ctx||!g_sampler)return JNI_FALSE;

    g_stop=false;
    const char * p=env->GetStringUTFChars(jprompt,nullptr);
    std::string user(p);
    env->ReleaseStringUTFChars(jprompt,p);
    g_history.push_back({"user",user});

    const std::string formatted=format_chat();
    const llama_vocab * vocab=llama_model_get_vocab(g_model);
    const auto tokens=tokenize(vocab,formatted,true);
    if(tokens.empty()){g_history.pop_back();return JNI_FALSE;}

    size_t common=0;
    while(common<g_cached_tokens.size() && common<tokens.size() && g_cached_tokens[common]==tokens[common])common++;
    if(common<g_cached_tokens.size()){
        llama_memory_seq_rm(llama_get_memory(g_ctx),0,(llama_pos)common,-1);
        g_cached_tokens.resize(common);
    }

    llama_sampler_reset(g_sampler);

    llama_batch batch=llama_batch_init(512,0,1);
    for(size_t i=common;i<tokens.size();++i){
        const int j=batch.n_tokens++;
        batch.token[j]=tokens[i];
        batch.pos[j]=(llama_pos)i;
        batch.n_seq_id[j]=1;
        batch.seq_id[j][0]=0;
        batch.logits[j]=(i+1==tokens.size());
        if(batch.n_tokens==512 || i+1==tokens.size()){
            if(llama_decode(g_ctx,batch)!=0){llama_batch_free(batch);g_history.pop_back();return JNI_FALSE;}
            batch.n_tokens=0;
        }
    }
    llama_batch_free(batch);
    g_cached_tokens=tokens;

    jclass cb_cls=env->GetObjectClass(callback);
    jmethodID on_token=env->GetMethodID(cb_cls,"onToken","(Ljava/lang/String;)V");
    std::string answer;
    answer.reserve(256);

    for(int step=0;step<2048 && !g_stop;++step){
        const llama_token tok=llama_sampler_sample(g_sampler,g_ctx,-1);
        if(llama_vocab_is_eog(vocab,tok))break;
        llama_sampler_accept(g_sampler,tok);

        char piece[8192];
        int n=llama_token_to_piece(vocab,tok,piece,(int)sizeof(piece),0,true);
        if(n>0){
            answer.append(piece,n);
            jstring js=env->NewStringUTF(std::string(piece,n).c_str());
            env->CallVoidMethod(callback,on_token,js);
            env->DeleteLocalRef(js);
        }

        llama_batch next=llama_batch_init(1,0,1);
        next.n_tokens=1;
        next.token[0]=tok;
        next.pos[0]=(llama_pos)g_cached_tokens.size();
        next.n_seq_id[0]=1;
        next.seq_id[0][0]=0;
        next.logits[0]=true;
        if(llama_decode(g_ctx,next)!=0){llama_batch_free(next);break;}
        llama_batch_free(next);
        g_cached_tokens.push_back(tok);
    }

    if(g_stop){
        g_history.pop_back();
        return JNI_FALSE;
    }
    g_history.push_back({"assistant",answer});
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL Java_com_llmapp_Engine_stop(JNIEnv*,jobject){g_stop=true;}
extern "C" JNIEXPORT void JNICALL Java_com_llmapp_Engine_unload(JNIEnv*,jobject){std::lock_guard<std::mutex> lock(g_mutex);clear_engine();}
