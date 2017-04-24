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

#include "properties.h"
#include "util.h"
#include "thread_rand.h"
#include "generic_io.h"

#include <iostream>
#include <fstream>
#include <valarray>

GlobalProperties* GlobalProperties::instance = NULL;

GlobalProperties::GlobalProperties()
{
    // pose parameters
    pP.randomDraw = true;
    
    pP.ransacIterations = 256;
    pP.ransacRefinementIterations = 8;
    pP.ransacBatchSize = 100;
    pP.ransacSubSample = 0.01;

    pP.ransacInlierThreshold2D = 10;
    pP.ransacInlierThreshold3D = 100;

    // dataset parameters
    dP.rawData = true;

    dP.focalLength = 525;
    dP.xShift = 0;
    dP.yShift = 0;

    dP.secondaryFocalLength = 585;
    dP.rawXShift = 0;
    dP.rawYShift = 0;

    dP.imageWidth = 640;
    dP.imageHeight = 480;

    dP.objScript = "train_obj.lua";
    dP.scoreScript = "train_score.lua";

    dP.objModel = "obj_model_init.net";
    dP.scoreModel = "score_model_init.net";

    dP.config = "default";

    // try reading external sensor transformation matrix
    try
    {
        std::ifstream calibFile("./sensorTrans.dat", std::ios::binary | std::ios_base::in);
        jp::read(calibFile, dP.sensorTrans);
        std::cout << GREENTEXT("Successfully loaded sensor transformation:") << std::endl;
        std::cout << dP.sensorTrans << std::endl << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << "Could not load sensor transformation." << std::endl;
        dP.sensorTrans = cv::Mat_<double>::eye(4, 4);
    }
}

GlobalProperties* GlobalProperties::getInstance()
{
    if(instance == NULL)
    instance = new GlobalProperties();
    return instance;
}

    
bool GlobalProperties::readArguments(std::vector<std::string> argv)
{
    int argc = argv.size();

    for(int i = 0; i < argc; i++)
    {
        std::string s = argv[i];

        if(s == "-iw")
        {
            i++;
            dP.imageWidth = std::atoi(argv[i].c_str());
            std::cout << "image width: " << dP.imageWidth << "\n";
            continue;
        }

        if(s == "-ih")
        {
            i++;
            dP.imageHeight = std::atoi(argv[i].c_str());
            std::cout << "image height: " << dP.imageHeight << "\n";
            continue;
        }

        if(s == "-fl")
        {
            i++;
            dP.focalLength = (float)std::atof(argv[i].c_str());
            std::cout << "focal length: " << dP.focalLength << "\n";
            continue;
        }

        if(s == "-xs")
        {
            i++;
            dP.xShift = (float)std::atof(argv[i].c_str());
            std::cout << "x shift: " << dP.xShift << "\n";
            continue;
        }

        if(s == "-ys")
        {
            i++;
            dP.yShift = (float)std::atof(argv[i].c_str());
            std::cout << "y shift: " << dP.yShift << "\n";
            continue;
        }

        if(s == "-rd")
        {
            i++;
            dP.rawData = std::atoi(argv[i].c_str());
            std::cout << "raw data (rescale rgb): " << dP.rawData << "\n";
            continue;
        }

        if(s == "-sfl")
        {
            i++;
            dP.secondaryFocalLength = (float)std::atof(argv[i].c_str());
            std::cout << "secondary focal length: " << dP.secondaryFocalLength << "\n";
            continue;
        }

        if(s == "-rxs")
        {
            i++;
            dP.rawXShift = (float)std::atof(argv[i].c_str());
            std::cout << "raw x shift: " << dP.rawXShift << "\n";
            continue;
        }

        if(s == "-rys")
        {
            i++;
            dP.rawYShift = (float)std::atof(argv[i].c_str());
            std::cout << "raw y shift: " << dP.rawYShift << "\n";
            continue;
        }

        if(s == "-rdraw")
        {
            i++;
            pP.randomDraw = std::atoi(argv[i].c_str());
            std::cout << "random draw: " << pP.randomDraw << "\n";
            continue;
        }

        if(s == "-oscript")
        {
            i++;
            dP.objScript = argv[i];
            std::cout << "object script: " << dP.objScript << "\n";
            continue;
        }

        if(s == "-sscript")
        {
            i++;
            dP.scoreScript = argv[i];
            std::cout << "score script: " << dP.scoreScript << "\n";
            continue;
        }

        if(s == "-omodel")
        {
            i++;
            dP.objModel = argv[i];
            std::cout << "object model: " << dP.objModel << "\n";
            continue;
        }

        if(s == "-smodel")
        {
            i++;
            dP.scoreModel = argv[i];
            std::cout << "score model: " << dP.scoreModel << "\n";
            continue;
        }

        if(s == "-rT2D")
        {
            i++;
            pP.ransacInlierThreshold2D = (float)std::atof(argv[i].c_str());
            std::cout << "ransac inlier threshold: " << pP.ransacInlierThreshold2D << "\n";
            continue;
        }

        if(s == "-rT3D")
        {
            i++;
            pP.ransacInlierThreshold3D = (float)std::atof(argv[i].c_str());
            std::cout << "ransac inlier threshold: " << pP.ransacInlierThreshold3D << "\n";
            continue;
        }

        if(s == "-rRI")
        {
            i++;
            pP.ransacRefinementIterations = std::atoi(argv[i].c_str());
            std::cout << "ransac iterations (refinement): " << pP.ransacRefinementIterations << "\n";
            continue;
        }

        if(s == "-rI")
        {
            i++;
            pP.ransacIterations = std::atoi(argv[i].c_str());
            std::cout << "ransac iterations: " << pP.ransacIterations << "\n";
            continue;
        }

        if(s == "-rB")
        {
            i++;
            pP.ransacBatchSize = std::atoi(argv[i].c_str());
            std::cout << "ransac batch size: " << pP.ransacBatchSize << "\n";
            continue;
        }

        if(s == "-rSS")
        {
            i++;
            pP.ransacSubSample = (float)std::atof(argv[i].c_str());
            std::cout << "ransac refinement gradient sub sampling: " << pP.ransacSubSample << "\n";
            continue;
        }

        std::cout << "unkown argument: " << argv[i] << "\n";
        return false;
    }
}
  
void GlobalProperties::parseCmdLine(int argc, const char* argv[])
{
    std::vector<std::string> argVec;
    for(int i = 1; i < argc; i++) argVec.push_back(argv[i]);
    readArguments(argVec);
}

void GlobalProperties::parseConfig()
{
    std::string configFile = dP.config + ".config";
    std::cout << BLUETEXT("Parsing config file: ") << configFile << std::endl;
    
    std::ifstream file(configFile);
    if(!file.is_open()) return;

    std::vector<std::string> argVec;

    std::string line;
    std::vector<std::string> tokens;
	
    while(true)
    {
        if(file.eof()) break;

        std::getline(file, line);
        if(line.length() == 0) continue; //empty line
        if(line.at(0) == '#') continue; // comment

        tokens = split(line);
        if(tokens.empty()) continue;

        argVec.push_back("-" + tokens[0]);
        argVec.push_back(tokens[1]);
    }
    
    readArguments(argVec);
}

cv::Mat_<float> GlobalProperties::getCamMat()
{
    float centerX = dP.imageWidth / 2 + dP.xShift;
    float centerY = dP.imageHeight / 2 + dP.yShift;
    float f = dP.focalLength;

    cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
    camMat(0, 0) = f;
    camMat(1, 1) = f;
    camMat(2, 2) = 1.f;
    
    camMat(0, 2) = centerX;
    camMat(1, 2) = centerY;
    
    return camMat;
}

static GlobalProperties* instance;
