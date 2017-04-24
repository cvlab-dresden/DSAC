/*
Copyright (c) 2016, TU Dresden
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "read_data.h"
#include "util.h"

#include <fstream>
#include "png++/png.hpp"

namespace jp
{
    void readData(const std::string dFile, jp::img_depth_t& image)
    {
        png::image<depth_t> imgPng(dFile);
        image = jp::img_depth_t(imgPng.get_height(), imgPng.get_width());

        for(int x = 0; x < imgPng.get_width(); x++)
        for(int y = 0; y < imgPng.get_height(); y++)
        {
            image(y, x) = (jp::depth_t) imgPng.get_pixel(x, y);
        }
    }

    void readData(const std::string bgrFile, jp::img_bgr_t& image)
    {
        png::image<png::basic_rgb_pixel<uchar>> imgPng(bgrFile);
        image = jp::img_bgr_t(imgPng.get_height(), imgPng.get_width());

        for(int x = 0; x < imgPng.get_width(); x++)
        for(int y = 0; y < imgPng.get_height(); y++)
        {
            image(y, x)(0) = (uchar) imgPng.get_pixel(x, y).blue;
            image(y, x)(1) = (uchar) imgPng.get_pixel(x, y).green;
            image(y, x)(2) = (uchar) imgPng.get_pixel(x, y).red;
        }
    }
  
    void readData(const std::string bgrFile, const std::string dFile, jp::img_bgrd_t& image)
    {
        readData(bgrFile, image.bgr);
        readData(dFile, image.depth);
    }

    
    bool readData(const std::string infoFile, jp::info_t& info)
    {
        std::ifstream file(infoFile);
        if(!file.is_open())
        {
            info.visible = false;
            return false;
        }

        std::string line;
        std::vector<std::string> tokens;

        cv::Mat_<float> trans = cv::Mat_<float>::eye(4, 4);
        info.rotation = cv::Mat_<float>(3, 3);

        for(unsigned i = 0; i < 3; i++)
        {
            std::getline(file, line);
            tokens = split(line);

            trans(i, 0) = std::atof(tokens[0].c_str());
            trans(i, 1) = std::atof(tokens[1].c_str());
            trans(i, 2) = std::atof(tokens[2].c_str());
            trans(i, 3) = std::atof(tokens[3].c_str());
        }

        std::ifstream transFile("translation.txt");

        if(transFile.is_open())
        {
            std::getline(transFile, line);
            tokens = split(line);
            trans(0, 3) -= std::atof(tokens[0].c_str());
            trans(1, 3) -= std::atof(tokens[1].c_str());
            trans(2, 3) -= std::atof(tokens[2].c_str());
            transFile.close();
        }
        else
        {
            std::cout << REDTEXT("WARNING! Cannot open translation.txt") << std::endl;
        }

        // correction for 7-scene poses (different coordinate frame definition from our internal defintion)
        cv::Mat_<float> correction = cv::Mat_<float>::eye(4, 4);
        correction.col(1) = -correction.col(1);
        correction.col(2) = -correction.col(2);
        trans = trans * correction;

        trans = trans.inv();

        for(unsigned x = 0; x < 3; x++)
        for(unsigned y = 0; y < 3; y++)
            info.rotation(y, x) = trans(y, x);

        for(unsigned x = 0; x < 3; x++)
            info.center[x] = trans(x, 3);

        info.extent[0] = 10;
        info.extent[1] = 10;
        info.extent[2] = 10;

        info.visible = true;
        info.occlusion = 0;
        return true;
    }
}


