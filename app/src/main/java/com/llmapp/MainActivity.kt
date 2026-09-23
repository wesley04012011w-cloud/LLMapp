package com.llmapp

import android.os.Bundle
import android.net.Uri
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
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

private data class Message(val user:Boolean,val text:String)

private class Engine {
    init { System.loadLibrary("llmapp") }
    external fun loadModel(path:String):Boolean
    external fun generate(prompt:String,callback:TokenCallback):Boolean
    external fun stop()
    external fun unload()
}
private fun interface TokenCallback { fun onToken(text:String) }

class MainActivity:ComponentActivity(){
    private val engine by lazy { Engine() }
    override fun onCreate(state:Bundle?){super.onCreate(state);setContent{MaterialTheme(colorScheme=darkColorScheme(background=Color(0xFF0B0B0C),surface=Color(0xFF151517),onBackground=Color(0xFFE8E8E8),onSurface=Color(0xFFE8E8E8))){ChatScreen(engine)}}}
    override fun onDestroy(){engine.stop();engine.unload();super.onDestroy()}
}

@Composable
private fun ChatScreen(engine:Engine){
    val context=androidx.compose.ui.platform.LocalContext.current
    val scope=rememberCoroutineScope()
    val messages=remember{mutableStateListOf<Message>()}
    var input by remember{mutableStateOf("")}; var loaded by remember{mutableStateOf(false)}
    var generating by remember{mutableStateOf(false)}; var current by remember{mutableStateOf("")}
    val list=rememberLazyListState()
    val picker=rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()){uri:Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        scope.launch(Dispatchers.IO){
            val file=File(context.filesDir,"model.gguf")
            context.contentResolver.openInputStream(uri)?.use{ins->file.outputStream().use{outs->ins.copyTo(outs,1024*1024)}}
            val ok=engine.loadModel(file.absolutePath)
            launch(Dispatchers.Main){loaded=ok;if(!ok)messages.add(Message(false,"Falha ao carregar o modelo."))}
        }
    }
    LaunchedEffect(messages.size,current){if(messages.isNotEmpty())list.animateScrollToItem(messages.lastIndex)}
    Column(Modifier.fillMaxSize().padding(12.dp)){
        LazyColumn(Modifier.weight(1f).fillMaxWidth(),state=list,verticalArrangement=Arrangement.spacedBy(10.dp),contentPadding=PaddingValues(vertical=12.dp)){
            items(messages){m->Row(Modifier.fillMaxWidth(),horizontalArrangement=if(m.user)Arrangement.End else Arrangement.Start){Surface(shape=RoundedCornerShape(18.dp),color=if(m.user)Color(0xFF242427) else Color(0xFF151517)){Text(m.text,Modifier.padding(14.dp))}}}
            if(current.isNotEmpty())item{Surface(shape=RoundedCornerShape(18.dp),color=Color(0xFF151517)){Text(current,Modifier.padding(14.dp))}}
        }
        Row(Modifier.fillMaxWidth(),verticalAlignment=Alignment.CenterVertically,horizontalArrangement=Arrangement.spacedBy(8.dp)){
            OutlinedTextField(value=input,onValueChange={input=it},modifier=Modifier.weight(1f),placeholder={Text("Mensagem...")},maxLines=4,enabled=!generating)
            Button(onClick={
                if(generating){engine.stop();return@Button}
                val prompt=input.trim();if(prompt.isEmpty()||!loaded)return@Button
                input="";messages.add(Message(true,prompt));current="";generating=true
                scope.launch(Dispatchers.IO){
                    val ok=engine.generate(prompt,TokenCallback{token->scope.launch(Dispatchers.Main){current+=token}})
                    launch(Dispatchers.Main){if(current.isNotEmpty())messages.add(Message(false,current));current="";generating=false;if(!ok)messages.add(Message(false,"Geração interrompida ou falhou."))}
                }
            },enabled=loaded||generating,modifier=Modifier.height(56.dp)){Text(if(generating)"Pausar" else "Enviar")}
            OutlinedButton(onClick={picker.launch(arrayOf("application/octet-stream","application/x-gguf","*/*"))},enabled=!generating,modifier=Modifier.height(56.dp)){Text("Carregar modelo")}
        }
    }
}
