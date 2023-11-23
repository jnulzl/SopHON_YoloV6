# ONNX模型RK35XX平台部署(以yolov6为例)


## 环境配置

- Ubuntu18.04/Ubuntu20.04

- cmake, git, make等等,根据实际情况缺什么安装什么

- [rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)

- [rknpu2](https://github.com/rockchip-linux/rknpu2)

- [android-ndk-r25c-linux](https://dl.google.com/android/repository/android-ndk-r25c-linux.zip)

## 模型量化

见[rknn-toolkit2-onnx-yolov5](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5)

## 构建

```shell
cd $ROOT_DIR
mkdir build && cd build
export ANDROID_NDK='YOUR ANDROID_NDK ROOT'
cmake ../ -DCMAKE_BUILD_TYPE=Release -DENGINE_LOWER=rk356x -DCMAKE_SYSTEM_NAME=Android -DANDROID_ABI=arm64-v8a -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake
```
输出产物在`ROOT_DIR/bin/Linux`下面，拷贝到板子跑即可。
