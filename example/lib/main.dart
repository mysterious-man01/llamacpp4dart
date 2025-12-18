import 'dart:async';
import 'dart:isolate';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:llamacpp4dart/llamacpp4dart.dart';

void main() {
  runApp(const MyApp());
}

class LlamaTask {
  final String modelPath;
  final String prompt;
  final SendPort sendPort;

  LlamaTask({
    required this.modelPath,
    required this.prompt,
    required this.sendPort,
  });
}

/// Llama.cpp inference Isolate
Future<void> _llamaWorker(LlamaTask task) async {
  final mParams = LlamaModelParams(nGpuLayers: 99);
  final cParams = LlamaCtxParams(nThreads: 8, nThreadsbatch: 8, offloadKqv: true);
  final llama = Llama(mParams: mParams, cParams: cParams);

  try {
    final isLoaded = await llama.loadModel(task.modelPath);
    if(!isLoaded.$1){
      throw Exception(isLoaded.$2);
    }

    final sw = Stopwatch()..start();
    final result = await llama.generate(task.prompt);

    sw.stop();
    String debugString = "\t\t\ttime: ${sw.elapsedMilliseconds}ms | tokens: ${llama.tokenize(result).$1.length}";
    task.sendPort.send("$result\n$debugString");
  } catch (e) {
    task.sendPort.send('Isolate error: $e');
  } finally {
    llama.dispose();
  }
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String _response = '';
  String? _modelPath;
  bool _isLoading = false;
  bool _modelLoaded = false;

  final TextEditingController _textController = TextEditingController();

  Future<void> _pickModelFile() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['gguf', 'bin', 'model'],
      );

      if (result != null && result.files.single.path != null) {
        final path = result.files.single.path!;
        setState(() {
          _modelPath = path;
          _modelLoaded = true;
          _response = 'Model selected successfuly!';
        });
      }
    } catch (e) {
      setState(() {
        _response = 'Error on select a model: $e';
      });
    }
  }

  Future<void> _sendText() async {
    if (!_modelLoaded || _modelPath == null) {
      setState(() {
        _response = '⚠️ Load a model first!';
      });
      return;
    }

    final input = _textController.text.trim();
    if (input.isEmpty) return;

    setState(() {
      _isLoading = true;
      _response = 'Generating...';
    });

    final receivePort = ReceivePort();

    try {
      await Isolate.spawn(
        _llamaWorker,
        LlamaTask(
          modelPath: _modelPath!,
          prompt: input,
          sendPort: receivePort.sendPort,
        ),
      );

      final result = await receivePort.first;

      setState(() {
        _response = result.toString();
      });
    } catch (e) {
      setState(() {
        _response = 'Processing error on Isolate: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
      receivePort.close();
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Llamacpp4Dart Example'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton.icon(
                  onPressed: _isLoading ? null : _pickModelFile,
                  icon: const Icon(Icons.folder_open),
                  label: const Text('Select a model'),
                ),
                if (_modelPath != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    'Model: ${_modelPath!.split('/').last}',
                    style: const TextStyle(fontSize: 14),
                  ),
                ],
                const SizedBox(height: 30),

                TextField(
                  controller: _textController,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'Make a question...',
                  ),
                ),
                const SizedBox(height: 10),

                ElevatedButton(
                  onPressed: _isLoading ? null : _sendText,
                  child: _isLoading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('Send'),
                ),
                const SizedBox(height: 20),

                const Text(
                  'Answer:',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 10),

                Text(
                  _response.isEmpty ? '(none response yet)' : _response,
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
