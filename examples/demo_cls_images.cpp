//
// Created by lizhaoliang-os on 2021/3/4.
//

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include "Module_cls.h"


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
    input_names.emplace_back("input1");

    output_names.clear();
    output_names.emplace_back("output1");

    BaseConfig config_tmp;
    float means_rgb[3] = {0.0f, 0.0f, 0.0f};
    float scales_rgb[3] = {1.0f, 1.0f, 1.0f}; // 1.0 / 255

    config_tmp.means[0] = means_rgb[0];
    config_tmp.means[1] = means_rgb[1];
    config_tmp.means[2] = means_rgb[2];

    config_tmp.scales[0] = scales_rgb[0];
    config_tmp.scales[1] = scales_rgb[1];
    config_tmp.scales[2] = scales_rgb[2];

    config_tmp.mean_length = 3;
    config_tmp.net_inp_channels = 3;

    std::string project_root = "./";
    if(argc < 4)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << " model_path input_size image_list"
                  << std::endl;
        return -1;
    }

    std::string weights_path = std::string(argv[1]);
    int input_size = std::atoi(argv[2]);

    /*******************cls_obj******************/
    CModule_cls cls_obj;
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.weights_path = weights_path;
    config_tmp.net_inp_width = input_size;
    config_tmp.net_inp_height = config_tmp.net_inp_width;
    cls_obj.init(config_tmp);

    std::ifstream input(argv[3]);
    std::string line;

    std::string img_path;
    long frame_id = 0;
    while (true)
    {
        cv::Mat frame;
        std::getline(input, line);
        if ("" == line)
            break;
        img_path = trim(line);
        frame = cv::imread(img_path);
        std::vector<std::string> items = split(line, '/', true);
        if (!frame.data)
        {
            break;
        }

        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
        cls_obj.process(frame);
        std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();
        std::cout << "frame_id:" << frame_id << " Using RK356X all time = " << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP1 - startTP).count() << " ms" << std::endl;

        const ClsInfo& clsInfo = cls_obj.get_result();
        std::cout << clsInfo.label << ", " << clsInfo.score << std::endl;
        frame_id++;
    }
    cls_obj.deinit();
    return 0;
}
