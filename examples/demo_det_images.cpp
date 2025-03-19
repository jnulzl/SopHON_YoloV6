//
// Created by lizhaoliang-os on 2021/3/4.
//

#include <iostream>
#include <string>
#include <chrono>

#include "ff_decode.hpp"
#include "bmnn_utils.h"

#include "opencv2/opencv.hpp"

#include "det/Module_det.h"
#include "utils/file_process.hpp"

std::vector<std::string> split(const std::string& string, char separator, bool ignore_empty) {
    std::vector<std::string> pieces;
    std::stringstream ss(string);
    std::string item;
    while (getline(ss, item, separator)) {
        if (!ignore_empty || !item.empty()) {
            pieces.push_back(std::move(item));
        }
    }
    return pieces;
}

std::string trim(const std::string& str) {
    size_t left = str.find_first_not_of(' ');
    if (left == std::string::npos) {
        return str;
    }
    size_t right = str.find_last_not_of(' ');
    return str.substr(left, (right - left + 1));
}

int main(int argc, char* argv[])
{
    std::string deploy_path;
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
    config_tmp.strides = {8, 16, 32};
    config_tmp.anchor_grids = { {10, 13, 16, 30, 33, 23} , {30, 61, 62, 45, 59, 119}, {116, 90, 156, 198, 373, 326} };

    std::string project_root = "./";
    if(argc < 8)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << " onnx_model_path input_size num_cls device_id batch_size is_save_res image_list"
                  << std::endl;
        return -1;
    }

    std::string weights_path = std::string(argv[1]);
    int input_size = std::atoi(argv[2]);
    bool is_save_res = std::atoi(argv[6]);

    /*******************det_obj******************/
    bm1684x_det::CModule_det det_obj;
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.weights_path = weights_path;
    config_tmp.net_inp_width = input_size;
    config_tmp.net_inp_height = config_tmp.net_inp_width;
    config_tmp.num_cls = std::atoi(argv[3]);
#ifdef USE_BM1684X
    config_tmp.device_id = std::atoi(argv[4]);
    config_tmp.batch_size = std::atoi(argv[5]);
#endif
    config_tmp.conf_thres = 0.5;
    config_tmp.nms_thresh = 0.4;

    BMNNHandlePtr handle = std::make_shared<BMNNHandle>(config_tmp.device_id);
    bm_handle_t bm_handle = handle->handle();
    std::printf("bm_handle: %p\n", bm_handle);
    config_tmp.handle = bm_handle;

    std::cout << "Loading model from " << weights_path << std::endl;
    det_obj.init(config_tmp);
    std::cout << "Loading model end!" << std::endl;

    std::vector<std::string> img_list;
    alg_utils::get_all_line_from_txt(argv[7], img_list);

    long frame_id = 0;
    std::vector<ImageInfoUint8> image_Info_Uint8s(config_tmp.batch_size);
    std::vector<bm_image> bmimgs(config_tmp.batch_size);

    AIALG_PRINT("img_list.size():%d\n", img_list.size());
    int batch_num = img_list.size() / config_tmp.batch_size;
    for (int bs = 0; bs < batch_num; ++bs)
    {
        for (int idx = 0; idx < config_tmp.batch_size; ++idx)
        {
            int kk = bs * config_tmp.batch_size + idx;
            std::string img_path = trim(img_list[kk]);
            AIALG_PRINT("img_path[%d]:%s\n", kk, img_path.c_str());
            picDec(bm_handle, img_path.c_str(), bmimgs[idx]);
            image_Info_Uint8s[idx].data = reinterpret_cast<uint8_t*>(bmimgs[idx].image_private);
            image_Info_Uint8s[idx].img_height = bmimgs[idx].height;
            image_Info_Uint8s[idx].img_width = bmimgs[idx].width;
            image_Info_Uint8s[idx].is_device_data = 0;
            image_Info_Uint8s[idx].stride = -1;
            image_Info_Uint8s[idx].frame_id = frame_id;
            frame_id++;
        }
        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
        det_obj.process_batch(image_Info_Uint8s.data(), image_Info_Uint8s.size());
        std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();
        std::cout << "frame_id:" << frame_id << " Using all time = " << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP1 - startTP).count() << " ms" << std::endl;
    }

    const BoxInfos* res = det_obj.get_result();

    if(1 == is_save_res)
    {
        for (int bs = 0; bs < config_tmp.batch_size; ++bs)
        {
            std::string img_path = trim(img_list[bs]);
            cv::Mat frame = cv::imread(img_path);
            for (size_t idy = 0; idy < res[bs].size; idy++)
            {
                int xmin    = res[bs].boxes[idy].x1;
                int ymin    = res[bs].boxes[idy].y1;
                int xmax    = res[bs].boxes[idy].x2;
                int ymax    = res[bs].boxes[idy].y2;
                float score = res[bs].boxes[idy].score;
                int label   = res[bs].boxes[idy].label;
                std::cout << "xywh : " << xmin << " " << ymin << " " << xmax - xmin << " " << ymax - ymin << " " << score << " " << label << std::endl;
                cv::rectangle(frame, cv::Point2i(xmin, ymin), cv::Point2i(xmax, ymax), cv::Scalar(255, 0, 0), 2);
                if(config_tmp.num_cls > 25)
                {
                    cv::putText(frame, std::to_string(label), cv::Point(xmin, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255), 2);
                }
                else
                {
                    cv::putText(frame, std::to_string(label), cv::Point(xmin, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255), 2);
                }
                cv::putText(frame, std::to_string(score), cv::Point(xmax, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 2);
            }

            if (res[bs].size > 0)
            {
                cv::imwrite("res/img_" + std::to_string(bs) + ".jpg", frame);
            }
        }
    }

    det_obj.deinit();
    return 0;
}
