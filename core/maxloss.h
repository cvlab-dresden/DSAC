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
 * @brief Inverts a pose hypothesis.
 * @param hyp Input pose hypothesis.
 * @return Pose hypothesis that corresponds to the inverse transformation.
 */
Hypothesis getInvHyp(const Hypothesis& hyp)
{
    cv::Mat_<double> trans = cv::Mat_<float>::eye(4, 4);

    hyp.getRotation().copyTo(trans.rowRange(0,3).colRange(0,3));
    trans(0, 3) = hyp.getTranslation().x;
    trans(1, 3) = hyp.getTranslation().y;
    trans(2, 3) = hyp.getTranslation().z;

    trans = trans.inv();

    cv::Mat_<double>invRot(3, 3);
    trans.rowRange(0,3).colRange(0,3).copyTo(invRot);
    cv::Point3d t;
    t.x = trans(0, 3);
    t.y = trans(1, 3);
    t.z = trans(2, 3);

    Hypothesis invHyp;
    invHyp.setRotation(invRot);
    invHyp.setTranslation(t);
    return invHyp;
}

/**
 * @brief Maximum of translational error (cm) and rotational error (deg) between two pose hypothesis.
 * @param h1 Pose 1.
 * @param h2 Pose 2.
 * @return Loss.
 */
double maxLoss(const Hypothesis& h1, const Hypothesis& h2)
{
    // measure loss of inverted poses (camera pose instead of scene pose)
    Hypothesis invH1 = getInvHyp(h1);
    Hypothesis invH2 = getInvHyp(h2);

    double rotErr = invH1.calcAngularDistance(invH2);
    double tErr = cv::norm(invH1.getTranslation() - invH2.getTranslation());

    return std::min(std::max(rotErr, tErr / 10), MAXLOSS);
}

/**
 * @brief Calculate the derivative of the loss w.r.t. the estimated pose.
 * @param est Estimated pose (6 DoF).
 * @param gt Ground truth pose (6 DoF).
 * @return 1x6 Jacobean.
 */
cv::Mat_<double> dLossMax(std::vector<double> est, const std::vector<double>& gt)
{
    // data conversion
    cv::Mat rod1(3, 1, CV_64F), rod2(3, 1, CV_64F);
    rod1.at<double>(0, 0) = est[0]; rod1.at<double>(1, 0) = est[1]; rod1.at<double>(2, 0) = est[2];
    rod2.at<double>(0, 0) = gt[0]; rod2.at<double>(1, 0) = gt[1]; rod2.at<double>(2, 0) = gt[2];

    cv::Mat rot1, rot2, dRod;
    cv::Rodrigues(rod1, rot1, dRod);
    cv::Rodrigues(rod2, rot2);

    // measure loss of inverted poses (camera pose instead of scene pose)
    cv::Mat_<double> invRot1 = rot1.t();
    cv::Mat_<double> invRot2 = rot2.t();

    // get the difference rotation
    cv::Mat diffRot = rot1 * invRot2;

    // calculate rotational and translational error
    double trace = cv::trace(diffRot)[0];
    trace = std::min(3.0, std::max(-1.0, trace));
    double rotErr = 180*acos((trace-1.0)/2.0)/CV_PI;

    cv::Mat_<double> invT1(3, 1);
    invT1(0, 0) = -est[3] / 10;
    invT1(1, 0) = -est[4] / 10;
    invT1(2, 0) = -est[5] / 10;
    invT1 = invRot1 * invT1;

    cv::Mat_<double> invT2(3, 1);
    invT2(0, 0) = -gt[3] / 10;
    invT2(1, 0) = -gt[4] / 10;
    invT2(2, 0) = -gt[5] / 10;
    invT2 = invRot2 * invT2;

    double tErr = cv::norm(invT1 - invT2);

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
        cv::Mat_<double> dDist_dInvT1(1, 3);
        for(unsigned i = 0; i < 3; i++)
            dDist_dInvT1(0, i) = (invT1(i, 0) - invT2(i, 0)) / tErr;

        cv::Mat_<double> dInvT1_dEstT(3, 3);
        dInvT1_dEstT = -invRot1;

        cv::Mat_<double> dDist_dEstT = dDist_dInvT1 * dInvT1_dEstT;
        dDist_dEstT.copyTo(jacobean.colRange(3, 6));

        cv::Mat_<double> dInvT1_dInvRot1 = cv::Mat_<double>::zeros(3, 9);
        dInvT1_dInvRot1(0, 0) = -est[3] / 10;
        dInvT1_dInvRot1(0, 3) = -est[4] / 10;
        dInvT1_dInvRot1(0, 6) = -est[5] / 10;

        dInvT1_dInvRot1(1, 1) = -est[3] / 10;
        dInvT1_dInvRot1(1, 4) = -est[4] / 10;
        dInvT1_dInvRot1(1, 7) = -est[5] / 10;

        dInvT1_dInvRot1(2, 2) = -est[3] / 10;
        dInvT1_dInvRot1(2, 5) = -est[4] / 10;
        dInvT1_dInvRot1(2, 8) = -est[5] / 10;

        dRod = dRod.t();

        cv::Mat_<double> dDist_dRod = dDist_dInvT1 * dInvT1_dInvRot1 * dRod;
        dDist_dRod.copyTo(jacobean.colRange(0, 3));
    }
    else
    {
        // return gradient of rotational error
        dRod = dRod.t();

        cv::Mat_<double> dRotDiff = cv::Mat_<double>::zeros(9, 9);
        invRot2.row(0).copyTo(dRotDiff.row(0).colRange(0, 3));
        invRot2.row(1).copyTo(dRotDiff.row(1).colRange(0, 3));
        invRot2.row(2).copyTo(dRotDiff.row(2).colRange(0, 3));

        invRot2.row(0).copyTo(dRotDiff.row(3).colRange(3, 6));
        invRot2.row(1).copyTo(dRotDiff.row(4).colRange(3, 6));
        invRot2.row(2).copyTo(dRotDiff.row(5).colRange(3, 6));

        invRot2.row(0).copyTo(dRotDiff.row(6).colRange(6, 9));
        invRot2.row(1).copyTo(dRotDiff.row(7).colRange(6, 9));
        invRot2.row(2).copyTo(dRotDiff.row(8).colRange(6, 9));

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
