# 🦙 llamacpp4dart

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

### A Dart bind for C/C++ [Llama.cpp](https://github.com/ggml-org/llama.cpp)'s code designed for Android.

> **⚠️ [IMPORTANT]**

- This project is on alpha preview and can have unexpected bugs!

- It is not fully implemented!

## Features

- Inference with llama.cpp using the `.gguf` extension 
- GPU support with Vulkan

## Compatbility

- Only Android ecosystem (aarch64 and x86_64)
    - Not tested on x86_64

## Installation 

- Install [Vulkan SDK](https://vulkan.lunarg.com/doc/sdk/1.4.328.1/linux/getting_started.html)

- Add this project to your code

```yaml
dependencies:
    llamacpp4dart:
        git: https://github.com/mysterious-man01/llamacpp4dart.git
        ref: main
```

- To compile an app, use **`flutter build apk`** or command on next topic

## Compiling Example App

- Run the following command on terminal inside `llamacpp4dart/example`

```bash
    # Replace [n_core] with the number of CPU cores to speed up compilation
    CMAKE_BUILD_PARALLEL_LEVEL=[n_core] flutter build apk -v
```

## Limitations

- **Using Flutter's debuging tool may cause crash when Infering an answer**
- Streamed response not supported

## TODO

- Add suport for other platforms
- Add suport for multimodal models
- Improve response precision
- Add streamed response

## Source Project and References

- Ggerganov's [Llama.cpp](https://github.com/ggml-org/llama.cpp) project.

- Netdur's [llama_cpp_dart](https://github.com/netdur/llama_cpp_dart) project.

- Dane Madsen's [maid_llm](https://github.com/Mobile-Artificial-Intelligence/llama_sdk.git) project.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.