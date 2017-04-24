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

#pragma once

#include "types.h"

#include <string>
#include <vector>
 
/** Here global parameters are defined which are made available throughout the code. */
 
/**
 * @brief Parameters that affect the pose estimation.
 */
struct PoseParameters
{
    bool randomDraw; // draw a hypothesis randomly (true) or take the one with the largest score (false)

    int ransacIterations; // initial number of pose hypotheses drawn per frame
    int ransacRefinementIterations; // number of refinement iterations
    int ransacBatchSize; // max. inlier count for refinement
    float ransacSubSample; // ratio of pixels for which gradients are calculated during refinement

    float ransacInlierThreshold2D; // reprojection error threshold (in px) for measuring inliers in the pose pipeline
    float ransacInlierThreshold3D; // inlier threshold (in mm) for evaluation of the intermediate object coordinate prediction
};

/**
 * @brief Parameters that affect the data.
 */
struct DatasetParameters
{
    bool rawData; // true if RGB and depth channels are not registered

    float focalLength; // focal length of the RGB camera
    float xShift; // x position of the principal point of the RGB camera
    float yShift; // y position of the principal point of the RGB camera

    float secondaryFocalLength; // focal length of the depth camera
    float rawXShift; // x position of the principal point of the depth camera
    float rawYShift; // y position of the principal point of the depth camera

    cv::Mat_<double> sensorTrans; // rigid body transformation relating depth and RGB camera

    int imageWidth; // width of the input images (px)
    int imageHeight; // height of the input images (px)

    std::string objScript; // lua script for learning object coordinate regression
    std::string scoreScript; // lua script for learning hypothesis score regression

    std::string objModel; // file storing the object coordinate regression CNN
    std::string scoreModel; // file storing the score regression CNN

    std::string config; // name of the config file to read (a file that lists parameter values)
};

/**
 * @brief Singelton class for providing parameter setting globally throughout the code.
 */
class GlobalProperties
{
protected:
  /**
   * @brief Consgtructor. Sets default values for all parameters.
   */
  GlobalProperties();
public:
    // Forest parameters
    PoseParameters pP;
  
    // Testing parameters
    DatasetParameters dP;
    
    /**
     * @brief Get a pointer to the singleton. It will create an instance if none exists yet.
     * 
     * @return GlobalProperties* Singleton pointer.
     */
    static GlobalProperties* getInstance();
        
    /**
     * @brief Returns the 3x3 camera matrix consisting of the intrinsic camera parameters.
     * 
     * @return cv::Mat_< float > Camera/calibration matrix.
     */
    cv::Mat_<float> getCamMat();
        
    /**
     * @brief Parse the arguments given in the command line and set parameters accordingly
     * 
     * @param argc Number of parameters.
     * @param argv Array of parameters.
     * @return void
     */
    void parseCmdLine(int argc, const char* argv[]);
    
    /**
     * @brief Parse a config file (given by the global parameter "config") and set parameters accordingly
     * 
     * @return void
     */
    void parseConfig();
    
    /**
     * @brief Process a list of arguments and set parameters accordingly.
     * 
     * Each parameter consists of a dash followed by a abbreviation of the parameter. In most cases the next entry in the list should be the value that the parameter should take.
     * 
     * @param argv List of parameters and parameter values. 
     * @return bool
     */
    bool readArguments(std::vector<std::string> argv);
    
private:
    static GlobalProperties* instance; // Singleton instance.
};
