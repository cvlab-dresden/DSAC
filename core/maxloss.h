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

#define MAXLOSS 10000000.0 // clamping of the maximal possible loss for numerical stability

#include "Hypothesis.h"

/**
 * @brief Maximum of translational error (cm) and rotational error (deg) between two pose hypothesis.
 * @param h1 Pose 1.
 * @param h2 Pose 2.
 * @return Loss.
 */
double maxLoss(const Hypothesis& h1, const Hypothesis& h2)
{
    double rotErr = h1.calcAngularDistance(h2);
    double tErr = cv::norm(h1.getTranslation() - h2.getTranslation());
    
    return std::min(std::max(rotErr, tErr / 10), MAXLOSS);
}

/**
 * @brief Gradients of the max loss function w.r.t. the first input pose, ie the function is 6 -> 1
 * @param est Estimated pose (3D Rodriquez vector + 3D translation vector)
 * @param gt Ground truth pose (3D Rodriquez vector + 3D translation vector)
 * @return Jacobean of partian derivatives, 1x6.
 */
cv::Mat_<double> dLossMax(std::vector<double> est, const std::vector<double>& gt)
{
    // data conversion
    cv::Mat rod1(3, 1, CV_64F), rod2(3, 1, CV_64F);
    rod1.at<double>(0, 0) = est[0]; rod1.at<double>(1, 0) = est[1]; rod1.at<double>(2, 0) = est[2];
    rod2.at<double>(0, 0) = gt[0]; rod2.at<double>(1, 0) = gt[1]; rod2.at<double>(2, 0) = gt[2];
  
    cv::Mat rot1, rot2, dRod;
    cv::Rodrigues(rod1, rot1, dRod); // OpenCV calculates the derivatives of the vector-matrix conversion
    cv::Rodrigues(rod2, rot2);

    // get the difference rotation
    rot2 = rot2.inv();
    cv::Mat diffRot = rot1 * rot2;
    
    // calculate rotational and translational error
    double trace = cv::trace(diffRot)[0];
    trace = std::min(3.0, std::max(-1.0, trace));
    double rotErr = 180*acos((trace-1.0)/2.0)/CV_PI;      
    double tErr = sqrt((est[3] - gt[3]) * (est[3] - gt[3]) + (est[4] - gt[4]) * (est[4] - gt[4]) + (est[5] - gt[5]) * (est[5] - gt[5])) / 10; // in cm
    
    cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(1, est.size());

    // clamped loss, return zero gradient if loss is bigger than threshold
    if(std::max(rotErr, tErr) > MAXLOSS)
        return jacobean;
    
    // zero error, abort
    if((tErr + rotErr) < EPS)
        return jacobean;
    
    if(tErr > rotErr)
    {
        // return gradient of translational error
        for(unsigned i = 3; i < est.size(); i++)
            jacobean(0, i) = (est[i] - gt[i]) / 10 / tErr; // in cm
    }
    else
    {    
        // return gradient of rotational error
        dRod = dRod.t();

        cv::Mat_<double> dRotDiff = cv::Mat_<double>::zeros(9, 9);
        rot2.row(0).copyTo(dRotDiff.row(0).colRange(0, 3));
        rot2.row(1).copyTo(dRotDiff.row(1).colRange(0, 3));
        rot2.row(2).copyTo(dRotDiff.row(2).colRange(0, 3));

        rot2.row(0).copyTo(dRotDiff.row(3).colRange(3, 6));
        rot2.row(1).copyTo(dRotDiff.row(4).colRange(3, 6));
        rot2.row(2).copyTo(dRotDiff.row(5).colRange(3, 6));

        rot2.row(0).copyTo(dRotDiff.row(6).colRange(6, 9));
        rot2.row(1).copyTo(dRotDiff.row(7).colRange(6, 9));
        rot2.row(2).copyTo(dRotDiff.row(8).colRange(6, 9));

        dRotDiff = dRotDiff.t();

        cv::Mat_<double> dTrace = cv::Mat_<double>::zeros(1, 9);
        dTrace(0, 0) = 1;
        dTrace(0, 4) = 1;
        dTrace(0, 8) = 1;

        cv::Mat_<double> dAngle = (180 / CV_PI * -1 / sqrt(3 - trace * trace + 2 * trace)) * dTrace * dRotDiff * dRod;
        dAngle.copyTo(jacobean.colRange(0, 3));
    }

    if(cv::sum(cv::Mat(jacobean != jacobean))[0] > 0) //check for NaNs
        return cv::Mat_<double>::zeros(1, est.size());

    return jacobean;
}
