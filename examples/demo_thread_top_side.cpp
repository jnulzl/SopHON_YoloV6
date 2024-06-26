//
// Created by lizhaoliang-os on 2020/6/23.

#include <iostream>
#include <random>
#include <string>
#include <chrono>

#include <thread>
#include <queue>
#include <mutex>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include "det/Module_det.h"
#include "utils/file_process.hpp"

void det_thread_func(std::queue<cv::Mat>& frame_queue, std::mutex& frame_mutex, bool& is_stop,
                     const std::string& model_path, int net_inp_width, int net_inp_height,
                     int det_num_cls, int batch_size, int device_id, int is_save_res)
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

    while (true)
    {
        frame_mutex.lock();
        std::printf("Thread id = %ld, is_stop = %d, frame_mutex.size = %d\n", std::this_thread::get_id(), is_stop ? 1 : 0, frame_queue.size());
        if(is_stop && frame_queue.empty())
        {
            frame_mutex.unlock();
            break;
        }
        if(frame_queue.empty())
        {
            frame_mutex.unlock();
            continue;
        }
        cv::Mat frame = frame_queue.front().clone();
        frame_queue.pop();
        frame_mutex.unlock();

        img_batch.data = frame.data;
        img_batch.img_height = frame.rows;
        img_batch.img_width = frame.cols;
        img_batch.img_data_type = InputDataType::IMG_BGR;

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
                cv::imwrite("res/npu" + std::to_string(device_id) + "_frame" + std::to_string(frame_id) + ".jpg",img_show);
            }
        }
        frame_id++;
    }
    det.deinit();
}

int main(int argc, char* argv[])
{
    if(argc < 8)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << " onnx_model_path input_size num_cls batch_size device_id is_save_res video_path"
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
    std::string input_src = std::string(argv[7]);

    std::queue<cv::Mat> top_queue_g;
    std::queue<cv::Mat> side_queue_g;
    std::mutex top_mutex_g;
    std::mutex side_mutex_g;
    bool is_top_stop_g = false;
    bool is_side_stop_g = false;

    std::vector<std::thread> threads;
    threads.emplace_back(det_thread_func, std::ref(top_queue_g), std::ref(top_mutex_g), std::ref(is_top_stop_g),
                         weights_path, net_inp_width, net_inp_height,
                         det_num_cls, batch_size, 0, is_save_res);

    threads.emplace_back(det_thread_func, std::ref(side_queue_g), std::ref(side_mutex_g), std::ref(is_side_stop_g),
                         weights_path, net_inp_width, net_inp_height,
                         det_num_cls, batch_size, 1, is_save_res);

    std::vector<std::string> top_img_list;
    alg_utils::get_all_line_from_txt(input_src, top_img_list);
    std::vector<std::string> side_img_list;
    alg_utils::get_all_line_from_txt(input_src, side_img_list);

//    std::random_device rd_top;
//    std::mt19937 g_top(rd_top());
//    std::shuffle(top_img_list.begin(), top_img_list.end(), g_top);
//
//    std::random_device rd_side;
//    std::mt19937 g_side(rd_side());
//    std::shuffle(side_img_list.begin(), side_img_list.end(), g_side);

    int img_num = std::min(top_img_list.size(), side_img_list.size());
    for (int idx = 0; idx < img_num; ++idx)
    {
        cv::Mat top_frame = cv::imread(top_img_list[idx]);
        cv::Mat side_frame = cv::imread(side_img_list[idx]);
        top_mutex_g.lock();
        top_queue_g.push(top_frame);
        int top_queue_size = top_queue_g.size();
        top_mutex_g.unlock();

        side_mutex_g.lock();
        side_queue_g.push(side_frame);
        int side_queue_size = side_queue_g.size();
        side_mutex_g.unlock();

        if(top_queue_size > 150 || side_queue_size > 150)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    top_mutex_g.lock();
    is_top_stop_g = true;
    top_mutex_g.unlock();

    side_mutex_g.lock();
    is_side_stop_g = true;
    side_mutex_g.unlock();

    for (int idx = 0; idx < 2; ++idx)
    {
        threads[idx].join();
    }
    return 0;
}