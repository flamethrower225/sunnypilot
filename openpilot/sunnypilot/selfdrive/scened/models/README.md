# scene_detector.onnx

YOLOX-nano, 416 x 416, trained on COCO, exported by its authors:
https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
(sha256 c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d)

Copyright (c) 2021-2022 Megvii Inc. Licensed under the Apache License, Version 2.0:
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE

The build compiles it to `scene_detector_tinygrad.pkl` (see ../SConscript). scened uses its car, motorcycle, bus, truck and
traffic light classes.
