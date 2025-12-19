## Llama.cpp bind for Dart/Flutter.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

### A Dart bind for C/C++ [Llama.cpp](https://github.com/ggml-org/llama.cpp)'s code for use mainly on Android.

> **[IMPORTANT]**

- This project is on alfa preview and can have unespected bugs!

- It is not fully implemented!

## Features

- Inference with llama.cpp using the `.gguf` extension 
- GPU suport with Vulkan

## TODO

- Add suport for other platforms

## Installation 

- Install [Vulkan SDK](https://vulkan.lunarg.com/doc/sdk/1.4.328.1/linux/getting_started.html)

- Add this project to your code

```yaml
dependencies:
    llamacpp4dart:
        git: https://github.com/mysterious-man01/llamacpp4dart.git
        ref: main
```

- To compile an app, use `flutter build apk` or command on next topic

## Compiling Example App

- Run the following command on terminal inside `llamacpp4dart/example`

> CMAKE_BUILD_PARALLEL_LEVEL=[n_core] flutter build apk -v

## Limitations

- Using Flutter's debuging tool may cause crash when Infering an answer

## Source Projects and References

- Ggerganov's [Llama.cpp](https://github.com/ggml-org/llama.cpp) project.

- Netdur's [llama_cpp_dart](https://github.com/netdur/llama_cpp_dart) project.

- Dane Madsen's [maid_llm](https://github.com/Mobile-Artificial-Intelligence/llama_sdk.git) project.

## Licence

This project is licensed under the MIT License - see the `LICENSE.md` file for details.