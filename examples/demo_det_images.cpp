//
// Created by lizhaoliang-os on 2021/3/4.
//

#include <iostream>
#include <string>
#include <chrono>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

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

const static std::vector<std::string> det_labels = {"cleaning", "dump", "heating", "imbedding", "put_off", "titrate",
                                                    "wipe", "write", "paste_label", "stir", "write_label", "others"};

int main(int argc, char* argv[])
{
    std::string deploy_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    input_names.clear();
    input_names.emplace_back("images");

    output_names.clear();
    output_names.emplace_back("pred_bboxes");
    output_names.emplace_back("max_index");
//    output_names.emplace_back("max_scores");

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
    if(argc < 5)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << " onnx_model_path input_size num_cls image_list"
                  << std::endl;
        return -1;
    }

    std::string weights_path = std::string(argv[1]);
    int input_size = std::atoi(argv[2]);

    /*******************det_obj******************/
    rk35xx_det::CModule_det det_obj;
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.weights_path = weights_path;
    config_tmp.net_inp_width = input_size;
    config_tmp.net_inp_height = config_tmp.net_inp_width;
    config_tmp.num_cls = std::atoi(argv[3]);
    config_tmp.conf_thres = 0.6;
    config_tmp.nms_thresh = 0.4;

    std::cout << "Loading rknn model from " << weights_path << std::endl;
    det_obj.init(config_tmp);
    std::cout << "Loading rknn model end!" << std::endl;

    std::vector<std::string> img_list;
    alg_utils::get_all_line_from_txt(argv[4], img_list);
    
    long frame_id = 0;
    ImageInfoUint8 image_Info_Uint8;
    for (int idx = 0; idx < img_list.size(); ++idx)
    {
        std::string img_path = trim(img_list[idx]);
        cv::Mat frame = cv::imread(img_path);
        if (!frame.data)
        {
            break;
        }
        image_Info_Uint8.data = frame.data;
        image_Info_Uint8.img_height = frame.rows;
        image_Info_Uint8.img_width = frame.cols;
        image_Info_Uint8.is_device_data = 0;
        image_Info_Uint8.stride = frame.step;
        image_Info_Uint8.frame_id = frame_id;
        image_Info_Uint8.img_data_type = InputDataType::IMG_BGR;

        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
        det_obj.process_batch(&image_Info_Uint8, 1);
        std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();
        std::cout << "frame_id:" << frame_id << " Using RK356X all time = " << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP1 - startTP).count() << " ms" << std::endl;

        const BoxInfos* res = det_obj.get_result();
        std::cout << "Detected : " << res[0].size << " objects" << std::endl;
        for (size_t idy = 0; idy < res[0].size; idy++)
        {
            int xmin    = res[0].boxes[idy].x1;
            int ymin    = res[0].boxes[idy].y1;
            int xmax    = res[0].boxes[idy].x2;
            int ymax    = res[0].boxes[idy].y2;
            float score = res[0].boxes[idy].score;
            int label   = res[0].boxes[idy].label;
            std::cout << "xyxy : " << xmin << " " << ymin << " " << xmax << " " << ymax << " " << score << " " << label << std::endl;
            cv::rectangle(frame, cv::Point2i(xmin, ymin), cv::Point2i(xmax, ymax), cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, det_labels[label], cv::Point(xmin, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255), 2);
            cv::putText(frame, std::to_string(score), cv::Point(xmax, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 2);
        }

        if (res[0].size > 0)
        {
            cv::imwrite("res/video_" + std::to_string(0) + "_frame" + std::to_string(frame_id) +
                        "_bs" + std::to_string(0) + "_obj" + std::to_string(res[0].size) + ".jpg", frame);
        }
        frame_id++;
    }

    det_obj.deinit();
    return 0;
}
