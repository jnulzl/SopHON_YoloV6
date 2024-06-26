//
// Created by lizhaoliang-os on 2020/6/23.

#include <iostream>
#include <random>
#include <string>
#include <chrono>

#include <thread>
#include <mutex>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include "det/Module_det.h"
#include "utils/file_process.hpp"

void det_thread_func(const std::string& model_path, int net_inp_width, int net_inp_height,
                        int det_num_cls, int batch_size, int device_id,
                        const std::string& img_list_file_path, int is_save_res)
{
    std::string project_root = std::string(PROJECT_ROOT);

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    input_names.clear();
    input_names.emplace_back("images");

    output_names.clear();
    output_names.emplace_back("pred_bboxes");
    output_names.emplace_back("score_indexes");

    YoloConfig config_tmp;
    float means_rgb[3] = {0, 0, 0};
    float scales_rgb[3] = {0.0039215, 0.0039215, 0.0039215}; // 1.0 / 255

    config_tmp.means[0] = means_rgb[0];
    config_tmp.means[1] = means_rgb[1];
    config_tmp.means[2] = means_rgb[2];
    config_tmp.scales[0] = scales_rgb[0];
    config_tmp.scales[1] = scales_rgb[1];
    config_tmp.scales[2] = scales_rgb[2];

    config_tmp.mean_length = 3;
    config_tmp.net_inp_channels = 3;
    config_tmp.model_include_preprocess = 0;
    config_tmp.conf_thres = 0.5;
    config_tmp.nms_thresh = 0.3;
    config_tmp.strides = {8, 16, 32};
    config_tmp.anchor_grids = { {10, 13, 16, 30, 33, 23} , {30, 61, 62, 45, 59, 119}, {116, 90, 156, 198, 373, 326} };

    config_tmp.net_inp_width = net_inp_width;
    config_tmp.net_inp_height = net_inp_height;
    config_tmp.num_cls = det_num_cls;
#ifdef USE_RK3588
    config_tmp.batch_size = batch_size;
    config_tmp.device_id = device_id;
#endif
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.weights_path = model_path;
    config_tmp.deploy_path = model_path;

    rk35xx_det::CModule_det det;
    det.init(config_tmp);

    long frame_id = 0;
    ImageInfoUint8 img_batch;

    std::vector<std::string> img_list;
    alg_utils::get_all_line_from_txt(img_list_file_path, img_list);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(img_list.begin(), img_list.end(), g);
    for (int idx = 0; idx < img_list.size(); ++idx)
    {
        cv::Mat frame = cv::imread(img_list[idx]);
        if (frame.empty())
        {
            break;
        }
        if (frame.data)
        {
            img_batch.data = frame.data;
            img_batch.img_height = frame.rows;
            img_batch.img_width = frame.cols;
            img_batch.img_data_type = InputDataType::IMG_BGR;
        }
        else
        {
            img_batch.data = nullptr;
            img_batch.img_height = 0;
            img_batch.img_width = 0;
            img_batch.img_data_type = InputDataType::IMG_BGR;
        }

        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
        det.process_batch(&img_batch, 1);
        std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();

        const BoxInfos* res = det.get_result();
        std::cout << "Thread id = " << std::this_thread::get_id() << " Frame = " << frame_id << " Batch = " << 1 << " TensorRT process time = " << std::chrono::duration_cast<std::chrono::microseconds>(finishTP1 - startTP).count() << " us" << std::endl;
        if(img_batch.data)
        {
            std::cout << "Video " << 0 << " detected " << res[0].size << " objs" << std::endl;
        }
        else
        {
            std::cout << "Video is end!" << std::endl;
        }

        if(1 == is_save_res)
        {
            //show result
            if(img_batch.data)
            {
                int bs = 0;
                cv::Mat img_show = cv::Mat(img_batch.img_height, img_batch.img_width, CV_8UC3, img_batch.data);
                for (size_t idx = 0; idx < res[bs].size; idx++)
                {
                    int xmin    = res[bs].boxes[idx].x1;
                    int ymin    = res[bs].boxes[idx].y1;
                    int xmax    = res[bs].boxes[idx].x2;
                    int ymax    = res[bs].boxes[idx].y2;
                    float score = res[bs].boxes[idx].score;
                    int label   = res[bs].boxes[idx].label;
                    //std::cout << "xyxy : " << xmin << " " << ymin << " " << xmax << " " << ymax << " " << score << " " << label << std::endl;
                    cv::rectangle(img_show, cv::Point2i(xmin, ymin), cv::Point2i(xmax, ymax), cv::Scalar(255, 0, 0), 2);
                    cv::putText(img_show, std::to_string(label), cv::Point(xmin, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255), 2);
                    cv::putText(img_show, std::to_string(score), cv::Point(xmax, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 2);
                }
                cv::imwrite("res/frame" + std::to_string(frame_id) + ".jpg",img_show);
            }
        }
        frame_id++;
    }
    det.deinit();
}


int main(int argc, char* argv[])
{
    if(argc < 9)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << " onnx_model_path input_size num_cls batch_size device_id is_save_res thread_num video_path"
                  << std::endl;
        return -1;
    }
    std::string weights_path = std::string(argv[1]);
    int net_inp_width = std::atoi(argv[2]);
    int net_inp_height = net_inp_width;
    int det_num_cls = std::atoi(argv[3]);
    int batch_size = std::atoi(argv[4]);
    //int device_id = std::atoi(argv[5]);
    int is_save_res = std::atoi(argv[6]);
    int thread_num = std::atoi(argv[7]);
    std::string input_src = std::string(argv[8]);

    std::vector<std::thread> threads;
    for (int idx = 0; idx < thread_num; ++idx)
    {
        threads.emplace_back(det_thread_func,weights_path, net_inp_width, net_inp_height,
                             det_num_cls, batch_size, idx, input_src, is_save_res);
    }

    for (int idx = 0; idx < thread_num; ++idx)
    {
        threads[idx].join();
    }
    return 0;
}