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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.foundation.shape.RoundedCornerShape
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

private data class Message(val user:Boolean,val text:String,val thinking:String="")

private class Engine {
    init { System.loadLibrary("llmapp") }
    external fun initLogs(directory:String)
    external fun startEntryLog(path:String)
    external fun log(message:String)
    external fun loadModel(path:String):Boolean
    external fun generate(prompt:String,callback:TokenCallback):Boolean
    external fun setGenerationSettings(temperature:Float,topP:Float,topK:Int,minP:Float,maxTokens:Int,systemPrompt:String,useJinja:Boolean)
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
                    Toast.makeText(this,"Falha da sessão anterior registrada em:
${recovered.absolutePath}",Toast.LENGTH_LONG).show()
                }
            }catch(_:Throwable){}
        }

        sessionLog=File(logDir,"session-active.txt")
        try{
            sessionLog.writeText("========== SESSION START ==========
started=${Date()}
")
        }catch(_:Throwable){}
        Toast.makeText(this,"Logs do LLMapp serão salvos em:
$logDirPath",Toast.LENGTH_LONG).show()
        engine.initLogs(logDir.absolutePath)
        engine.startEntryLog(sessionLog.absolutePath)
        engine.log("[ui] session started")
        recordPreviousProcessExit()

        val previous=Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler{thread,error->
            try{
                File(logDir,"crash-${timestamp()}.txt").writeText(
                    "========== JAVA/KOTLIN CRASH ==========
"+
                    "thread=${thread.name}
"+
                    error.stackTraceToString()
                )
            }catch(_:Throwable){}
            previous?.uncaughtException(thread,error)
        }

        setContent{
            MaterialTheme(colorScheme=lightColorScheme(
                background=Color.White,
                surface=Color.White,
                onBackground=Color.Black,
                onSurface=Color.Black,
                primary=Color(0xFF6750A4),
                onPrimary=Color.White
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
                sessionLog.appendText("========== CLEAN SHUTDOWN ==========
")
                sessionLog.delete()
            }
        }catch(_:Throwable){}
        engine.stop()
        engine.unload()
        super.onDestroy()
    }
}

private fun timestamp():String=SimpleDateFormat("yyyyMMdd-HHmmss-SSS",Locale.US).format(Date())



@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatScreen(engine:Engine,logDir:File){ // generation settings sheet is opted into Material3 experimental API
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
    var showSettings by remember{mutableStateOf(false)}
    var temperature by remember{mutableStateOf(0.7f)}
    var topP by remember{mutableStateOf(0.95f)}
    var topK by remember{mutableStateOf(40)}
    var minP by remember{mutableStateOf(0.05f)}
    var maxTokens by remember{mutableStateOf(2048)}
    var systemPrompt by remember{mutableStateOf("")}
    var useJinja by remember{mutableStateOf(true)}
    val list=rememberLazyListState()

    val picker=rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()){uri:Uri?->
        if(uri!=null){
            scope.launch(Dispatchers.IO){
                try{
                    val file=File(context.filesDir,"model.gguf")
                    engine.log("[ui] copying model")
                    context.contentResolver.openInputStream(uri)?.use{ins->file.outputStream().use{outs->ins.copyTo(outs,1024*1024)}}
                        ?:throw IllegalStateException("Could not open selected model")
                    engine.log("[ui] model bytes="+file.length())
                    val ok=engine.loadModel(file.absolutePath)
                    launch(Dispatchers.Main){loaded=ok;if(!ok)messages.add(Message(false,"Falha ao carregar o modelo."))}
                }catch(t:Throwable){
                    engine.log("[ui] load exception: "+t.stackTraceToString())
                    launch(Dispatchers.Main){loaded=false;messages.add(Message(false,"Falha ao carregar o modelo."))}
                }
            }
        }
    }
    LaunchedEffect(generating,current,thinking,messages.size){
        if(generating){
            withFrameNanos{}
            val last=list.layoutInfo.totalItemsCount-1
            if(last>=0){
                // Keep the newest streaming item anchored to the bottom.
                // A large positive offset moves the item upward until its
                // bottom edge reaches the viewport bottom.
                list.scrollToItem(last,1_000_000)
            }
        }
    }

    Surface(Modifier.fillMaxSize(),color=Color.White){
    Column(Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing).padding(horizontal=12.dp,vertical=8.dp)){
        Row(
            Modifier.fillMaxWidth().height(52.dp),
            verticalAlignment=Alignment.CenterVertically
        ){
            Spacer(Modifier.weight(1f))
            IconButton(onClick={showSettings=true},enabled=!generating){
                Icon(Icons.Default.Settings,contentDescription="Configurações de geração",tint=Color.Black)
            }
            FilledTonalIconButton(
                onClick={picker.launch(arrayOf("application/octet-stream","application/x-gguf","*/*"))},
                enabled=!generating
            ){
                Icon(Icons.Default.FolderOpen,contentDescription="Carregar modelo")
            }
        }
        LazyColumn(Modifier.weight(1f).fillMaxWidth(),state=list,verticalArrangement=Arrangement.spacedBy(10.dp),contentPadding=PaddingValues(vertical=12.dp)){
            items(messages){m->
                if(m.user){
                    Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.End){
                        Surface(shape=RoundedCornerShape(18.dp),color=Color(0xFFF0F0F0),contentColor=Color.Black){
                            Text(m.text,Modifier.padding(14.dp))
                        }
                    }
                }else{
                    Column(Modifier.fillMaxWidth(),verticalArrangement=Arrangement.spacedBy(8.dp)){
                        if(m.thinking.isNotEmpty()){
                            Surface(shape=RoundedCornerShape(14.dp),color=Color(0xFFF5F5F5),contentColor=Color.Black){
                                Column(Modifier.padding(horizontal=14.dp,vertical=10.dp)){
                                    Text("THINKING",style=MaterialTheme.typography.labelSmall,color=Color.Black)
                                    Spacer(Modifier.height(4.dp))
                                    Text(m.thinking,color=Color.Black,style=MaterialTheme.typography.bodyMedium)
                                }
                            }
                        }
                        if(m.text.isNotEmpty()){
                            Text(m.text,Modifier.fillMaxWidth().padding(horizontal=2.dp,vertical=2.dp),color=Color.Black,style=MaterialTheme.typography.bodyLarge)
                        }
                    }
                }
            }
            if(thinking.isNotEmpty() || current.isNotEmpty()) item{
                Column(Modifier.fillMaxWidth(),verticalArrangement=Arrangement.spacedBy(8.dp)){
                    if(thinking.isNotEmpty()){
                        Surface(shape=RoundedCornerShape(14.dp),color=Color(0xFFF5F5F5),contentColor=Color.Black){
                            Column(Modifier.padding(horizontal=14.dp,vertical=10.dp)){
                                Text("THINKING",style=MaterialTheme.typography.labelSmall,color=Color.Black)
                                Spacer(Modifier.height(4.dp))
                                Text(thinking,color=Color.Black,style=MaterialTheme.typography.bodyMedium)
                            }
                        }
                    }
                    if(current.isNotEmpty()){
                        Text(current,Modifier.fillMaxWidth().padding(horizontal=2.dp,vertical=2.dp),color=Color.Black,style=MaterialTheme.typography.bodyLarge)
                    }
                }
            }
        }
        val sendEnabled=generating || (loaded && input.isNotBlank())
        TextField(
            value=input,
            onValueChange={input=it},
            modifier=Modifier.fillMaxWidth(),
            placeholder={Text("Mensagem...")},
            maxLines=5,
            enabled=!generating,
            shape=RoundedCornerShape(28.dp),
            colors=TextFieldDefaults.colors(
                focusedContainerColor=Color(0xFFF2F2F2),
                unfocusedContainerColor=Color(0xFFF2F2F2),
                disabledContainerColor=Color(0xFFF2F2F2),
                focusedTextColor=Color.Black,
                unfocusedTextColor=Color.Black,
                disabledTextColor=Color(0xFF777777),
                focusedIndicatorColor=Color.Transparent,
                unfocusedIndicatorColor=Color.Transparent,
                disabledIndicatorColor=Color.Transparent
            ),
            trailingIcon={
                IconButton(
                    onClick={
                        if(generating){
                            engine.stop()
                        }else{
                            val prompt=input.trim()
                            if(prompt.isNotEmpty() && loaded){
                                input=""
                                messages.add(Message(true,prompt))
                                current=""
                                thinking=""
                                thinkingActive=false
                                thinkingDone=false
                                streamBuffer=""
                                generating=true
                                scope.launch(Dispatchers.IO){
                                    try{
                                        engine.log("========== NEW ENTRY ==========")
                                        engine.log("[ui] send prompt length="+prompt.length)
                                        engine.setGenerationSettings(temperature,topP,topK,minP,maxTokens,systemPrompt,useJinja)
                                        val ok=engine.generate(prompt,TokenCallback{token->
                                            scope.launch(Dispatchers.Main){
                                                streamBuffer += token
                                                var again=true
                                                while(again){
                                                    again=false
                                                    if(!thinkingActive && !thinkingDone){
                                                        val starts=listOf("<think>","<|think|>","<|START_THINKING|>","<|channel|>analysis","[THINK]")
                                                        val hit=starts.mapNotNull{tag->streamBuffer.indexOf(tag).takeIf{it>=0}?.let{it to tag}}.minByOrNull{it.first}
                                                        if(hit!=null){
                                                            val(idx,tag)=hit
                                                            if(idx>0)current+=streamBuffer.substring(0,idx)
                                                            streamBuffer=streamBuffer.substring(idx+tag.length)
                                                            thinkingActive=true
                                                            again=true
                                                        }else{
                                                            val keep=starts.maxOfOrNull{tag->
                                                                val max=minOf(tag.length-1,streamBuffer.length)
                                                                (0..max).lastOrNull{n->streamBuffer.endsWith(tag.take(n))}?:0
                                                            }?:0
                                                            if(streamBuffer.length>keep){
                                                                current+=streamBuffer.dropLast(keep)
                                                                streamBuffer=if(keep==0)"" else streamBuffer.takeLast(keep)
                                                            }
                                                        }
                                                    }else if(thinkingActive){
                                                        val ends=listOf("</think>","<|/think|>","<|END_THINKING|>","<|channel|>final","[/THINK]","[BEGIN FINAL RESPONSE]")
                                                        val hit=ends.mapNotNull{tag->streamBuffer.indexOf(tag).takeIf{it>=0}?.let{it to tag}}.minByOrNull{it.first}
                                                        if(hit!=null){
                                                            val(idx,tag)=hit
                                                            thinking+=streamBuffer.substring(0,idx)
                                                            streamBuffer=streamBuffer.substring(idx+tag.length)
                                                            thinkingActive=false
                                                            thinkingDone=true
                                                            again=true
                                                        }else{
                                                            val keep=ends.maxOfOrNull{tag->
                                                                val max=minOf(tag.length-1,streamBuffer.length)
                                                                (0..max).lastOrNull{n->streamBuffer.endsWith(tag.take(n))}?:0
                                                            }?:0
                                                            if(streamBuffer.length>keep){
                                                                thinking+=streamBuffer.dropLast(keep)
                                                                streamBuffer=if(keep==0)"" else streamBuffer.takeLast(keep)
                                                            }
                                                        }
                                                    }else{
                                                        current+=streamBuffer
                                                        streamBuffer=""
                                                    }
                                                }
                                            }
                                        })
                                        engine.log("[ui] generate returned="+ok)
                                        launch(Dispatchers.Main){
                                            if(thinkingActive)thinking+=streamBuffer else current+=streamBuffer
                                            streamBuffer=""
                                            thinkingActive=false
                                            if(current.isNotEmpty() || thinking.isNotEmpty()){
                                                messages.add(Message(false,current,thinking))
                                            }
                                            current=""
                                            thinking=""
                                            generating=false
                                            if(!ok)messages.add(Message(false,"Geração interrompida ou falhou."))
                                        }
                                    }catch(t:Throwable){
                                        try{engine.log("[ui] generate exception: "+t.stackTraceToString())}catch(_:Throwable){}
                                        launch(Dispatchers.Main){
                                            current=""
                                            generating=false
                                            messages.add(Message(false,"Erro durante a geração."))
                                        }
                                    }
                                }
                            }
                        }
                    },
                    enabled=sendEnabled
                ){
                    Icon(
                        imageVector=if(generating)Icons.Default.Stop else Icons.Default.Send,
                        contentDescription=if(generating)"Parar geração" else "Enviar mensagem",
                        tint=Color.Black
                    )
                }
            }
        )
    }

        if(showSettings){
            val sheetState=rememberModalBottomSheetState(skipPartiallyExpanded=true)
            ModalBottomSheet(
                onDismissRequest={showSettings=false},
                sheetState=sheetState
            ){
                Column(
                    Modifier.fillMaxWidth().padding(horizontal=20.dp,vertical=8.dp),
                    verticalArrangement=Arrangement.spacedBy(12.dp)
                ){
                    Text("Configurações de geração",style=MaterialTheme.typography.titleLarge)
                    Text("Temperatura: %.2f".format(Locale.US,temperature))
                    Slider(value=temperature,onValueChange={temperature=it},valueRange=0f..2f)
                    Text("Top P: %.2f".format(Locale.US,topP))
                    Slider(value=topP,onValueChange={topP=it},valueRange=0f..1f)
                    Text("Top K: $topK")
                    Slider(value=topK.toFloat(),onValueChange={topK=it.toInt()},valueRange=0f..100f,steps=99)
                    Text("Min P: %.2f".format(Locale.US,minP))
                    Slider(value=minP,onValueChange={minP=it},valueRange=0f..1f)
                    Text("Máximo de tokens: $maxTokens")
                    Slider(value=maxTokens.toFloat(),onValueChange={maxTokens=(it/128).toInt()*128},valueRange=128f..4096f,steps=31)
                    Row(verticalAlignment=Alignment.CenterVertically){
                        Checkbox(checked=useJinja,onCheckedChange={useJinja=it})
                        Text("Usar template Jinja do modelo")
                    }
                    OutlinedTextField(
                        value=systemPrompt,
                        onValueChange={systemPrompt=it},
                        modifier=Modifier.fillMaxWidth(),
                        label={Text("System prompt")},
                        minLines=3,
                        maxLines=6
                    )
                    Spacer(Modifier.height(8.dp))
                }
            }
        }

    }
}
