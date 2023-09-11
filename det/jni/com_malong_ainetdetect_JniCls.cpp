#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>

#include "com_malong_ainetdetect_JniCls.h"
#include "Module_cls.h"

static std::mutex io_mutex;

#define TAG "JNICLS"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,    TAG, __VA_ARGS__)
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE,    TAG, __VA_ARGS__)


static void Yuv2Rgb(int width, int height, const uint8_t *y_buffer, const uint8_t *u_buffer,
                    const uint8_t *v_buffer, int y_pixel_stride, int uv_pixel_stride,
                    int y_row_stride, int uv_row_stride, int *argb_output)
{
    uint32_t a = (255u << 24);
    uint8_t r, g, b;
    int16_t y_val, u_val, v_val;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            // Y plane should have positive values belonging to [0...255]
            int y_idx = (y * y_row_stride) + (x * y_pixel_stride);
            y_val = static_cast<int16_t>(y_buffer[y_idx]);

            int uvx = x / 2;
            int uvy = y / 2;
            // U/V Values are sub-sampled i.e. each pixel in U/V channel in a
            // YUV_420 image act as chroma value for 4 neighbouring pixels
            int uv_idx = (uvy * uv_row_stride) + (uvx * uv_pixel_stride);

            u_val = static_cast<int16_t>(u_buffer[uv_idx]) - 128;
            v_val = static_cast<int16_t>(v_buffer[uv_idx]) - 128;

            // Compute RGB values per formula above.
            r = y_val + 1.370705f * v_val;
            g = y_val - (0.698001f * v_val) - (0.337633f * u_val);
            b = y_val + 1.732446f * u_val;

            int argb_idx = y * width + x;
            argb_output[argb_idx] = a | r << 16 | g << 8 | b;
        }
    }
}

static void BitmapToMat2(JNIEnv *env, jobject &bitmap, cv::Mat &dst, jboolean needUnPremultiplyAlpha)
{
    AndroidBitmapInfo info;
    void *pixels = 0;
    try
    {
#ifndef NDEBUG
        LOGD("nBitmapToMat");
#endif
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        dst.create(info.height, info.width, CV_8UC3);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888)
        {
#ifndef NDEBUG
            LOGD("nBitmapToMat: RGBA_8888 -> CV_8UC3");
#endif
            cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (needUnPremultiplyAlpha)
            {
                cv::cvtColor(tmp, dst, cv::COLOR_RGBA2BGR);
            } else
            {
                tmp.copyTo(dst);
            }
        } else
        {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            LOGD("nBitmapToMat: RGB_565 -> CV_8UC4");
            cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cv::cvtColor(tmp, dst, cv::COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e)
    {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nBitmapToMat catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...)
    {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nBitmapToMat catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}

static void BitmapToMat(JNIEnv *env, jobject &bitmap, cv::Mat &mat)
{
    BitmapToMat2(env, bitmap, mat, true);
}

#ifdef __cplusplus
extern "C" {
#endif


/*
 * Class:     com_malong_ainetdetect_JniCls
 * Method:    alg_cls_init
 * Signature: (Ljava/lang/String;)
 */
JNIEXPORT void JNICALL Java_com_malong_ainetdetect_JniCls_init(JNIEnv *env, jobject self,
                                                               jstring model_path, jint net_inp_width,
                                                               jint net_inp_height, jint net_inp_channel)
{
    std::lock_guard<std::mutex> lk(io_mutex);
    auto *cls_ptr_ = new CModule_cls();

    BaseConfig baseConfig;
    baseConfig.input_names.resize(1);
    baseConfig.input_names[0] = "input1";

    baseConfig.output_names.resize(1);
    baseConfig.output_names[0] = "output1";

    const char *weight_path = env->GetStringUTFChars(model_path, nullptr);
    baseConfig.weights_path = std::string(weight_path);
    baseConfig.deploy_path = "";
    env->ReleaseStringUTFChars(model_path, weight_path);

    baseConfig.means[0] = 0.0f;
    baseConfig.means[1] = 0.0f;
    baseConfig.means[2] = 0.0f;

    baseConfig.scales[0] = 1.0f;
    baseConfig.scales[1] = 1.0f;
    baseConfig.scales[2] = 1.0f;

    baseConfig.mean_length = net_inp_channel;
    baseConfig.net_inp_channels = net_inp_channel;
    baseConfig.net_inp_width = net_inp_width;
    baseConfig.net_inp_height = net_inp_height;

    baseConfig.num_threads = 1;
    ANY_POINTER_CAST(cls_ptr_, CModule_cls)->init(baseConfig);

    // Get a reference to this object's class
    jclass selfClass = env->GetObjectClass(self);
    // long
    // Get the Field ID of the instance variables "handle"
    jfieldID fidHandle = env->GetFieldID(selfClass, "handle", "J");

    jlong handle = reinterpret_cast<jlong>(cls_ptr_);
    // Change the variable
    env->SetLongField(self, fidHandle, handle);
#ifndef NDEBUG
    LOGD("Java_com_malong_ainetdetect_JniCls_init");
#endif
}

/*
 * Class:     com_malong_ainetdetect_JniCls
 * Method:    alg_cls_uninit
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_malong_ainetdetect_JniCls_destory(JNIEnv *env, jobject self)
{
    std::lock_guard<std::mutex> lk(io_mutex);

    // Get a reference to this object's class
    jclass selfClass = env->GetObjectClass(self);
    // long
    // Get the Field ID of the instance variables "handle"
    jfieldID fidHandle = env->GetFieldID(selfClass, "handle", "J");
    // Get the long given the Field ID
    jlong handle = env->GetLongField(self, fidHandle);

    ANY_POINTER_CAST(handle, CModule_cls)->deinit();
    delete ANY_POINTER_CAST(handle, CModule_cls);
#ifndef NDEBUG
    LOGD("Java_com_malong_ainetdetect_JniCls_destory");
#endif
}

/*
 * Class:     com_malong_ainetdetect_JniCls
 * Method:    alg_cls_run
 * Signature: (Landroid/graphics/Bitmap;)V
 */
JNIEXPORT void JNICALL Java_com_malong_ainetdetect_JniCls_run(JNIEnv *env, jobject self, jobject bitmap)
{
    std::lock_guard<std::mutex> lk(io_mutex);
    // Get a reference to this object's class
    jclass selfClass = env->GetObjectClass(self);
    // long
    // Get the Field ID of the instance variables "handle"
    jfieldID fidHandle = env->GetFieldID(selfClass, "handle", "J");
    // Get the long given the Field ID
    jlong handle = env->GetLongField(self, fidHandle);

    cv::Mat img;
    LOGD("BitmapToMat begin ......");
    BitmapToMat(env, bitmap, img);
    LOGD("BitmapToMat end and  process ......");
    ANY_POINTER_CAST(handle, CModule_cls)->process(img);
    LOGD("process end!");
#ifndef NDEBUG
    LOGD("Java_com_malong_ainetdetect_JniCls_run");
#endif
}

/*
 * Class:     com_malong_ainetdetect_JniCls
 * Method:    alg_cls_get
 * Signature: ()Lcom/malong/jnicls/ClsInfo;
 */
JNIEXPORT jobject JNICALL Java_com_malong_ainetdetect_JniCls_get(JNIEnv *env, jobject self)
{
    std::lock_guard<std::mutex> lk(io_mutex);
    // Get a reference to this object's class
    jclass selfClass = env->GetObjectClass(self);
    // long
    // Get the Field ID of the instance variables "handle"
    jfieldID fidHandle = env->GetFieldID(selfClass, "handle", "J");
    // Get the long given the Field ID
    jlong handle = env->GetLongField(self, fidHandle);

    const ClsInfo &cls_info = ANY_POINTER_CAST(handle, CModule_cls)->get_result();

    // Get a class reference for java.lang.Integer
    jclass cls = env->FindClass("com/malong/ainetdetect/ClsInfo");
    // Get the Method ID of the constructor which takes an int
    jmethodID method_id = env->GetMethodID(cls, "<init>", "(IF)V");
    // Call back constructor to allocate a new instance, with an int argument
    jobject newObj = env->NewObject(cls, method_id, cls_info.label, cls_info.score);
#ifndef NDEBUG
    LOGD("Java_com_malong_ainetdetect_JniCls_get");
#endif
    return newObj;
}

JNIEXPORT jboolean JNICALL
Java_com_malong_ainetdetect_YuvConvertor_yuv420toArgbNative(JNIEnv *env, jclass clazz, jint width,
                                                            jint height, jobject y_byte_buffer,
                                                            jobject u_byte_buffer,
                                                            jobject v_byte_buffer,
                                                            jint y_pixel_stride,
                                                            jint uv_pixel_stride, jint y_row_stride,
                                                            jint uv_row_stride,
                                                            jintArray argb_array)
{
    auto y_buffer = reinterpret_cast<uint8_t *>(env->GetDirectBufferAddress(y_byte_buffer));
    auto u_buffer = reinterpret_cast<uint8_t *>(env->GetDirectBufferAddress(u_byte_buffer));
    auto v_buffer = reinterpret_cast<uint8_t *>(env->GetDirectBufferAddress(v_byte_buffer));
    jint *argb_result_array = env->GetIntArrayElements(argb_array, nullptr);
    if (argb_result_array == nullptr || y_buffer == nullptr || u_buffer == nullptr
        || v_buffer == nullptr)
    {
        LOGE("[yuv420toArgbNative] One or more inputs are null.");
        return false;
    }

    Yuv2Rgb(width, height, reinterpret_cast<const uint8_t *>(y_buffer),
            reinterpret_cast<const uint8_t *>(u_buffer),
            reinterpret_cast<const uint8_t *>(v_buffer),
            y_pixel_stride, uv_pixel_stride, y_row_stride, uv_row_stride,
            argb_result_array);
    return true;
}


#ifdef __cplusplus
}
#endif