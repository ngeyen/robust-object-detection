import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite/tflite.dart';
import 'package:image_picker/image_picker.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: ObjectDetectionScreen());
  }
}

class ObjectDetectionScreen extends StatefulWidget {
  @override
  _ObjectDetectionScreenState createState() => _ObjectDetectionScreenState();
}

class _ObjectDetectionScreenState extends State<ObjectDetectionScreen> {
  late CameraController _cameraController;
  late List<CameraDescription> _cameras;
  File? _image;
  List? _recognitions;
  double _imageHeight = 0;
  double _imageWidth = 0;
  bool _isDetecting = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  Future<void> _initializeCamera() async {
    _cameras = await availableCameras();
    if (_cameras.isNotEmpty) {
      _cameraController = CameraController(
        _cameras.first,
        ResolutionPreset.high,
      );
      await _cameraController.initialize();
      if (mounted) {
        setState(() {});
      }
    }
  }

  Future _loadModel() async {
    String? res = await Tflite.loadModel(
      model:
          "assets/retinanet.tflite", // Make sure this is the correct path to your model
      labels: "assets/retinanet.txt",
    ); //And this is the path to labels file.
    print("Tflite Load Model status $res");
  }

  Future _detectObject(File image) async {
    if (!_isDetecting) {
      setState(() {
        _isDetecting = true;
      });
      var recognitions = await Tflite.detectObjectOnImage(
        path: image.path,
        model: "RetinaNet", // Use "RetinaNet" here
        imageMean:
            0.0, // Different mean/std for RetinaNet?  Check your model's requirements
        imageStd: 255.0,
        threshold:
            0.5, // Adjust this threshold as needed for your RetinaNet model
        numResultsPerClass:
            100, // Or another suitable value for your model.  RetinaNet often returns more detections.
      );

      if (mounted) {
        setState(() {
          _recognitions = recognitions;
          _image = image;
        });
      }
      _getImageSize(image);
      setState(() {
        _isDetecting = false;
      });
    }
  }

  void _getImageSize(File image) async {
    final decodedImage = await decodeImageFromList(image.readAsBytesSync());
    setState(() {
      _imageWidth = decodedImage.width.toDouble();
      _imageHeight = decodedImage.height.toDouble();
    });
  }

  Future _takePicture() async {
    if (!_cameraController.value.isInitialized) {
      return null;
    }
    if (_cameraController.value.isTakingPicture) {
      return null;
    }
    try {
      await _cameraController.setFlashMode(FlashMode.off);
      XFile picture = await _cameraController.takePicture();
      File imageFile = File(picture.path);
      _detectObject(imageFile);
    } on CameraException catch (e) {
      debugPrint("Error occured while taking picture: $e");
      return null;
    }
  }

  _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? pickedFile = await picker.pickImage(
      source: ImageSource.gallery,
    );
    if (pickedFile != null) {
      File imageFile = File(pickedFile.path);
      _detectObject(imageFile);
    }
  }

  List<Widget> _renderBoxes(Size screen) {
    if (_recognitions == null) return [];
    if (_imageHeight == 0 || _imageWidth == 0) return [];

    double factorX = screen.width;
    double factorY = _imageHeight / _imageWidth * screen.width;

    Color color = Colors.red; // You can choose a different color

    return _recognitions!.map((re) {
      return Positioned(
        left: re["rect"]["x"] * factorX,
        top: re["rect"]["y"] * factorY,
        width: re["rect"]["w"] * factorX,
        height: re["rect"]["h"] * factorY,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: color,
              width: 2, // Adjust border width as needed
            ),
          ),
          child: Text(
            "${re["detectedClass"]} ${(re["confidenceInClass"] * 100).toStringAsFixed(0)}%", // Display class and confidence
            style: TextStyle(
              background: Paint()..color = color,
              color: Colors.white,
              fontSize: 12, // Adjust font size as needed
            ),
          ),
        ),
      );
    }).toList();
  }

  @override
  void dispose() {
    _cameraController.dispose();
    Tflite.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    List<Widget> list = [];
    if (_cameraController.value.isInitialized) {
      list.add(
        Center(
          child: Container(
            height: size.height - 180,
            child: AspectRatio(
              aspectRatio: _cameraController.value.aspectRatio,
              child: CameraPreview(_cameraController),
            ),
          ),
        ),
      );
      if (_image != null) {
        list.add(
          Container(child: _image == null ? Container() : Image.file(_image!)),
        );
      }
      list.addAll(_renderBoxes(size));
    }

    return Scaffold(
      appBar: AppBar(title: Text('Object Detection')),
      body: Stack(children: list),
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            onPressed: () => _takePicture(),
            tooltip: 'Take Picture',
            child: Icon(Icons.camera_alt),
          ),
          SizedBox(width: 16),
          FloatingActionButton(
            onPressed: () => _pickImage(),
            tooltip: 'Pick Image',
            child: Icon(Icons.photo),
          ),
        ],
      ),
    );
  }
}
