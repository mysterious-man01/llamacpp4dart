## Llama.cpp bind for Dart/Flutter.

### A Dart bind for C/C++ Llama.cpp's code for use mainly on Android.

## Features

- Inference with llama.cpp using the `.gguf` extension 
- GPU suport with Vulkan

## TODO

- Add suport for other platforms

## Installation 

- Install [Vulkan SDK](https://vulkan.lunarg.com/doc/sdk/1.4.328.1/windows/getting_started.html)

- Add this project to your code

```yaml
depndencies:
    llamacpp4dart:
        git: https://github.com/mysterious-man01/llamacpp4dart.git
        ref: main
```

## Compiling Exemple App

- Run the following command on terminal inside `llamacpp4dart/example`

> CMAKE_BUILD_PARALLEL_LEVEL=[n_core] flutter build apk -v

## Limitations

- Using Flutter's debuging tool may cause crash when Infering an answer