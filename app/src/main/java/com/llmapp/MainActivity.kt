package com.llmapp

import android.app.ActivityManager
import android.os.Build
import android.os.Bundle
import android.net.Uri
import android.os.Environment
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private data class Message(val user:Boolean,val text:String)

private class Engine {
    init { System.loadLibrary("llmapp") }
    external fun initLogs(directory:String)
    external fun startEntryLog(path:String)
    external fun log(message:String)
    external fun loadModel(path:String):Boolean
    external fun generate(prompt:String,callback:TokenCallback):Boolean
    external fun stop()
    external fun unload()
}
private fun interface TokenCallback { fun onToken(text:String) }

class MainActivity:ComponentActivity(){
    private val engine by lazy { Engine() }
    private lateinit var logDir:File
    private lateinit var logDirPath:String
    private lateinit var sessionLog:File

    override fun onCreate(state:Bundle?){
        super.onCreate(state)
        enableEdgeToEdge()
        window.isNavigationBarContrastEnforced = true
        logDir=File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS),"LLMapp-LOGS").apply{mkdirs()}
        logDirPath=logDir.absolutePath

        val previousSession=File(logDir,"session-active.txt")
        if(previousSession.exists() && previousSession.length()>0){
            val recovered=File(logDir,"crash-recovered-${timestamp()}.txt")
            try{
                if(previousSession.renameTo(recovered)){
                    Toast.makeText(this,"Falha da sessão anterior registrada em:\n${recovered.absolutePath}",Toast.LENGTH_LONG).show()
                }
            }catch(_:Throwable){}
        }

        sessionLog=File(logDir,"session-active.txt")
        try{
            sessionLog.writeText("========== SESSION START ==========\nstarted=${Date()}\n")
        }catch(_:Throwable){}
        Toast.makeText(this,"Logs do LLMapp serão salvos em:\n$logDirPath",Toast.LENGTH_LONG).show()
        engine.initLogs(logDir.absolutePath)
        engine.startEntryLog(sessionLog.absolutePath)
        engine.log("[ui] session started")
        recordPreviousProcessExit()

        val previous=Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler{thread,error->
            try{
                File(logDir,"crash-${timestamp()}.txt").writeText(
                    "========== JAVA/KOTLIN CRASH ==========\n"+
                    "thread=${thread.name}\n"+
                    error.stackTraceToString()
                )
            }catch(_:Throwable){}
            previous?.uncaughtException(thread,error)
        }

        setContent{
            MaterialTheme(colorScheme=darkColorScheme(
                background=Color(0xFF0B0B0C),surface=Color(0xFF151517),
                onBackground=Color(0xFFE8E8E8),onSurface=Color(0xFFE8E8E8)
            )){ChatScreen(engine,logDir)}
        }
    }
    private fun recordPreviousProcessExit(){
    if(Build.VERSION.SDK_INT<30)return
    try{
        val am=getSystemService(ActivityManager::class.java)
        val exits=am.getHistoricalProcessExitReasons(packageName,0,5)
        if(exits.isEmpty())return
        val latest=exits.first()
        val now=System.currentTimeMillis()
        if(now-latest.timestamp<10*60*1000L){
            engine.log("[exit] reason=${latest.reason} status=${latest.status} pss=${latest.pss}KB rss=${latest.rss}KB desc=${latest.description ?: ""}")
            if(latest.reason==android.app.ApplicationExitInfo.REASON_CRASH_NATIVE){
                engine.log("[exit] previous process died from NATIVE CRASH")
            }else if(latest.reason==android.app.ApplicationExitInfo.REASON_CRASH){
                engine.log("[exit] previous process died from JAVA/KOTLIN CRASH")
            }else if(latest.reason==android.app.ApplicationExitInfo.REASON_ANR){
                engine.log("[exit] previous process died from ANR")
            }else if(latest.reason==android.app.ApplicationExitInfo.REASON_LOW_MEMORY){
                engine.log("[exit] previous process was killed by LOW MEMORY")
            }
        }
    }catch(t:Throwable){
        try{engine.log("[exit] diagnostic failed: ${t.stackTraceToString()}")}catch(_:Throwable){}
    }
}

    override fun onDestroy(){
        try{
            engine.log("[ui] activity destroyed")
            if(!isChangingConfigurations){
                engine.log("[ui] clean shutdown")
                sessionLog.appendText("========== CLEAN SHUTDOWN ==========\n")
                sessionLog.delete()
            }
        }catch(_:Throwable){}
        engine.stop()
        engine.unload()
        super.onDestroy()
    }
}

private fun timestamp():String=SimpleDateFormat("yyyyMMdd-HHmmss-SSS",Locale.US).format(Date())



@Composable
private fun ChatScreen(engine:Engine,logDir:File){
    val context=androidx.compose.ui.platform.LocalContext.current
    val scope=rememberCoroutineScope()
    val messages=remember{mutableStateListOf<Message>()}
    var input by remember{mutableStateOf("")}
    var loaded by remember{mutableStateOf(false)}
    var generating by remember{mutableStateOf(false)}
    var current by remember{mutableStateOf("")}
    var thinking by remember{mutableStateOf("")}
    var thinkingActive by remember{mutableStateOf(false)}
    var thinkingDone by remember{mutableStateOf(false)}
    var streamBuffer by remember{mutableStateOf("")}
    val list=rememberLazyListState()

    val picker=rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()){uri:Uri?->
        uri?:return@rememberLauncherForActivityResult
        scope.launch(Dispatchers.IO){
            try{
                val file=File(context.filesDir,"model.gguf")
                engine.log("[ui] copying model")
                context.contentResolver.openInputStream(uri)?.use{ins->file.outputStream().use{outs->ins.copyTo(outs,1024*1024)}}
                    ?:throw IllegalStateException("Could not open selected model")
                engine.log("[ui] model bytes=${file.length()}")
                val ok=engine.loadModel(file.absolutePath)
                launch(Dispatchers.Main){loaded=ok;if(!ok)messages.add(Message(false,"Falha ao carregar o modelo."))}
            }catch(t:Throwable){
                engine.log("[ui] load exception: ${t.stackTraceToString()}")
                launch(Dispatchers.Main){loaded=false;messages.add(Message(false,"Falha ao carregar o modelo."))}
            }
        }
    }

    LaunchedEffect(messages.size,current){if(messages.isNotEmpty())list.animateScrollToItem(messages.lastIndex)}

    Column(Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing).padding(12.dp)){
        LazyColumn(Modifier.weight(1f).fillMaxWidth(),state=list,verticalArrangement=Arrangement.spacedBy(10.dp),contentPadding=PaddingValues(vertical=12.dp)){
            items(messages){m->Row(Modifier.fillMaxWidth(),horizontalArrangement=if(m.user)Arrangement.End else Arrangement.Start){
                Surface(shape=RoundedCornerShape(18.dp),color=if(m.user)Color(0xFF242427)else Color(0xFF151517)){Text(m.text,Modifier.padding(14.dp))}
            }}
            if(thinking.isNotEmpty())item{
                Surface(shape=RoundedCornerShape(18.dp),color=Color(0xFF101012)){
                    Column(Modifier.padding(14.dp)){
                        Text(if(thinkingActive)"Pensando..." else "Pensamento",style=MaterialTheme.typography.labelMedium)
                        Spacer(Modifier.height(6.dp))
                        Text(thinking)
                    }
                }
            }
            if(current.isNotEmpty())item{Surface(shape=RoundedCornerShape(18.dp),color=Color(0xFF151517)){Text(current,Modifier.padding(14.dp))}}
        }
        Row(Modifier.fillMaxWidth(),verticalAlignment=Alignment.CenterVertically,horizontalArrangement=Arrangement.spacedBy(8.dp)){
            OutlinedTextField(value=input,onValueChange={input=it},modifier=Modifier.weight(1f),placeholder={Text("Mensagem...")},maxLines=4,enabled=!generating)
            Button(onClick={
                if(generating){engine.stop();return@Button}
                val prompt=input.trim();if(prompt.isEmpty()||!loaded)return@Button
                input="";messages.add(Message(true,prompt));current="";thinking="";thinkingActive=false;thinkingDone=false;streamBuffer="";generating=true
                scope.launch(Dispatchers.IO){
                    try{
                        engine.log("========== NEW ENTRY ==========")
                        engine.log("[ui] send prompt length=${prompt.length}")
                        val ok=engine.generate(prompt,TokenCallback{token->
                            scope.launch(Dispatchers.Main){
                                streamBuffer += token
                                var again = true
                                while(again){
                                    again = false
                                    if(!thinkingActive && !thinkingDone){
                                        val starts = listOf("<think>", "<|think|>", "<|START_THINKING|>", "[THINK]")
                                        val hit = starts.mapNotNull { tag -> streamBuffer.indexOf(tag).takeIf { it >= 0 }?.let { it to tag } }.minByOrNull { it.first }
                                        if(hit != null){
                                            val (idx, tag) = hit
                                            if(idx > 0) current += streamBuffer.substring(0, idx)
                                            streamBuffer = streamBuffer.substring(idx + tag.length)
                                            thinkingActive = true
                                            again = true
                                        }else if(streamBuffer.length > 64){
                                            current += streamBuffer.dropLast(32)
                                            streamBuffer = streamBuffer.takeLast(32)
                                        }
                                    }else if(thinkingActive){
                                        val ends = listOf("</think>", "<|/think|>", "<|END_THINKING|>", "[/THINK]", "[BEGIN FINAL RESPONSE]")
                                        val hit = ends.mapNotNull { tag -> streamBuffer.indexOf(tag).takeIf { it >= 0 }?.let { it to tag } }.minByOrNull { it.first }
                                        if(hit != null){
                                            val (idx, tag) = hit
                                            thinking += streamBuffer.substring(0, idx)
                                            streamBuffer = streamBuffer.substring(idx + tag.length)
                                            thinkingActive = false
                                            thinkingDone = true
                                            again = true
                                        }else if(streamBuffer.length > 64){
                                            thinking += streamBuffer.dropLast(32)
                                            streamBuffer = streamBuffer.takeLast(32)
                                        }
                                    }else{
                                        current += streamBuffer
                                        streamBuffer = ""
                                    }
                                }
                            }
                        })
                        engine.log("[ui] generate returned=$ok")
                        launch(Dispatchers.Main){
                            current += streamBuffer
                            streamBuffer = ""
                            if(thinkingActive) thinkingActive = false
                            if(current.isNotEmpty())messages.add(Message(false,current))
                            thinking=""; thinkingActive=false; thinkingDone=false
                            current="";generating=false
                            if(!ok)messages.add(Message(false,"Geração interrompida ou falhou."))
                        }
                    }catch(t:Throwable){
                        try{engine.log("[ui] generate exception: ${t.stackTraceToString()}")}catch(_:Throwable){}
                        launch(Dispatchers.Main){current="";generating=false;messages.add(Message(false,"Erro durante a geração."))}
                    }
                }
            },enabled=loaded||generating,modifier=Modifier.height(56.dp)){Text(if(generating)"Pausar" else "Enviar")}
            OutlinedButton(onClick={picker.launch(arrayOf("application/octet-stream","application/x-gguf","*/*"))},enabled=!generating,modifier=Modifier.height(56.dp)){Text("Carregar modelo")}
        }
    }
}
