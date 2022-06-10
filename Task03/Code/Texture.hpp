//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v)
    {
        float u_img = u * width;
        float v_img = (1-v) * height;
        int x0 = (int)std::floor(u_img);
        int cx = u_img - (float)x0 > 0.5f ? x0 + 1 : x0;
        int y0 = (int)std::floor(v_img);
        int cy = v_img - (float)y0 > 0.5f ? y0 + 1 : y0;
        float s = u_img - ((float)cx-0.5f), t = v_img - ((float)y0 - 0.5f);
        
        cv::Vec3b color00 = image_data.at<cv::Vec3b>(cy, cx);
        cv::Vec3b color01 = image_data.at<cv::Vec3b>(cy + 1, cx);
        cv::Vec3b color10 = image_data.at<cv::Vec3b>(cy, cx + 1);
        cv::Vec3b color11 = image_data.at<cv::Vec3b>(cy + 1, cx + 1);

        cv::Vec3b color0 = (1-s) * color00 + s*color10;
        cv::Vec3b color1 = (1-s) * color01 + s*color11;
        
        cv::Vec3b color = color0 + t*(color1 - color0);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

};
#endif //RASTERIZER_TEXTURE_H
