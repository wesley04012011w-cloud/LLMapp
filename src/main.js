import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import "./style.css";

const messagesEl=document.querySelector("#messages"),form=document.querySelector("#composer"),promptEl=document.querySelector("#prompt"),sendBtn=document.querySelector("#send"),loadBtn=document.querySelector("#load");
const conversation=[]; let controller=null, modelLoaded=false;

function addBubble(role,text=""){const el=document.createElement("div");el.className=`bubble ${role}`;el.textContent=text;messagesEl.appendChild(el);messagesEl.scrollTop=messagesEl.scrollHeight;return el}
function setBusy(busy){sendBtn.textContent=busy?"Pausar":"Enviar";promptEl.disabled=busy;loadBtn.disabled=busy}

async function streamChat(){
 if(!modelLoaded||controller)return; const content=promptEl.value.trim(); if(!content)return;
 promptEl.value="";conversation.push({role:"user",content});addBubble("user",content);const assistant=addBubble("assistant");
 controller=new AbortController();setBusy(true);
 try{
  const response=await fetch("http://127.0.0.1:39271/v1/chat/completions",{method:"POST",headers:{"Content-Type":"application/json"},signal:controller.signal,body:JSON.stringify({model:"local",messages:conversation,stream:true,temperature:.7,top_p:.95,max_tokens:-1,cache_prompt:true})});
  if(!response.ok)throw new Error(await response.text());
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer="",answer="";
  while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const lines=buffer.split("\n");buffer=lines.pop()??"";
   for(const line of lines){if(!line.startsWith("data:"))continue;const payload=line.slice(5).trim();if(!payload||payload==="[DONE]")continue;const data=JSON.parse(payload),delta=data.choices?.[0]?.delta?.content??"";if(!delta)continue;answer+=delta;assistant.textContent=answer;messagesEl.scrollTop=messagesEl.scrollHeight}
  }
  conversation.push({role:"assistant",content:answer});
 }catch(error){if(error.name!=="AbortError"){assistant.textContent="Erro: "+(error.message||String(error));conversation.pop()}}finally{controller=null;setBusy(false);promptEl.focus()}
}

form.addEventListener("submit",e=>{e.preventDefault();if(controller)controller.abort();else streamChat()});
loadBtn.addEventListener("click",async()=>{if(controller)return;const selected=await open({multiple:false,directory:false,filters:[{name:"GGUF model",extensions:["gguf"]}]});if(!selected||Array.isArray(selected))return;try{await invoke("load_model",{modelPath:selected});conversation.length=0;messagesEl.replaceChildren();modelLoaded=true;promptEl.focus()}catch(e){addBubble("assistant",e?.toString()??"Falha ao carregar modelo.")}});
