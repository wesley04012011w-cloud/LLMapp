# LLMapp 1.0.0 — Android

App Android minimalista para inferência local de GGUF com llama.cpp via JNI.

A V1 contém somente input, Enviar/Pausar, bolhas de chat e Carregar modelo.

A engine mantém um llama_context vivo, sem HTTP. O histórico fica no native e o contexto/KV cache é reaproveitado entre turnos: o prompt novo é tokenizado, comparado com os tokens já avaliados e apenas o sufixo diferente é enviado ao decoder.

O build é arm64-v8a e usa KleidiAI, Release/LTO, batch 512 e Flash Attention em AUTO. A documentação oficial do llama.cpp recomenda arm64-v8a, GGML_NATIVE=OFF em cross-compilation e KleidiAI para Android.

O modelo é escolhido pelo Storage Access Framework e copiado para o armazenamento privado do app antes do carregamento.

V1 intencionalmente não possui configurações, lista de modelos, download, histórico externo ou outros elementos de UI.