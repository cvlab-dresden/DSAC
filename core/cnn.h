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

#include "util.h"
#include "lua_calls.h"
#include "maxloss.h"

/**
* @brief Checks whether the given matrix contains NaN entries.
* @param m Input matrix.
* @return True if m contrains NaN entries.
*/
inline bool containsNaNs(const cv::Mat& m)
{
   return cv::sum(cv::Mat(m != m))[0] > 0;
}

/**
 * @brief Wrapper around the OpenCV PnP function that returns a zero pose in case PnP fails. See also documentation of cv::solvePnP.
 * @param objPts List of 3D points.
 * @param imgPts Corresponding 2D points.
 * @param camMat Calibration matrix of the camera.
 * @param distCoeffs Distortion coefficients.
 * @param rot Output parameter. Camera rotation.
 * @param trans Output parameter. Camera translation.
 * @param extrinsicGuess If true uses input rot and trans as initialization.
 * @param methodFlag Specifies the PnP algorithm to be used.
 * @return True if PnP succeeds.
 */
inline bool safeSolvePnP(
    std::vector<cv::Point3f> objPts,
    const std::vector<cv::Point2f>& imgPts,
    const cv::Mat& camMat,
    const cv::Mat& distCoeffs,
    cv::Mat& rot,
    cv::Mat& trans,
    bool extrinsicGuess,
    int methodFlag)
{
    if(rot.type() == 0) rot = cv::Mat_<double>::zeros(1, 3);
    if(trans.type() == 0) trans= cv::Mat_<double>::zeros(1, 3);

    if(!cv::solvePnP(objPts, imgPts, camMat, distCoeffs, rot, trans, extrinsicGuess,methodFlag))
    {
        rot = cv::Mat_<double>::zeros(1, 3);
        trans = cv::Mat_<double>::zeros(1, 3);
        return false;
    }

    return true;
}

/**
 * @brief Calculate the Shannon entropy of a discrete distribution.
 * @param dist Discrete distribution. Probability per entry, should sum to 1.
 * @return  Shannon entropy.
 */
double entropy(const std::vector<double>& dist)
{
    double e = 0;
    for(unsigned i = 0; i < dist.size(); i++)
	if(dist[i] > 0)
	    e -= dist[i] * std::log2(dist[i]);
    
    return e;
}

/**
 * @brief Draws an entry of a discrete distribution according to the given probabilities.
 *
 * If randomDraw is false in the properties, this function will return the entry with the max. probability.
 *
 * @param probs Discrete distribution. Probability per entry, should sum to 1.
 * @return Chosen entry.
 */
int draw(const std::vector<double>& probs)
{
    std::map<double, int> cumProb; // create a map of cumulative probabilities
    double probSum = 0;
    double maxProb = -1;
    double maxIdx = 0; 
    
    for(unsigned idx = 0; idx < probs.size(); idx++)
    {
        if(probs[idx] < EPS) continue;

        probSum += probs[idx];
        cumProb[probSum] = idx;

        if(maxProb < 0 || probs[idx] > maxProb)
        {
            maxProb = probs[idx];
            maxIdx = idx;
        }
    }
    
    if(GlobalProperties::getInstance()->pP.randomDraw)
      return cumProb.upper_bound(drand(0, probSum))->second; // choose randomly
    else
      return maxIdx; // choose entry with max. probability
}

/**
 * @brief Calculates the expected loss of a list of poses with associated probabilities.
 * @param gt Ground truth pose.
 * @param hyps List of estimated poses.
 * @param probs List of probabilities associated with the estimated poses.
 * @param losses Output parameter. List of losses for each estimated pose.
 * @return Expectation of loss.
 */
double expectedMaxLoss(const Hypothesis& gt, const std::vector<jp::cv_trans_t>& hyps, const std::vector<double>& probs, std::vector<double>& losses)
{
    double loss = 0;
    losses.resize(hyps.size());
    
    for(unsigned i = 0; i < hyps.size(); i++)
    {
        jp::jp_trans_t jpHyp = jp::cv2our(hyps[i]);
        Hypothesis hyp(jpHyp.first, jpHyp.second);
        losses[i] = maxLoss(gt, hyp);
        loss += probs[i] * losses[i];
    }
    
    return loss;
}

/**
 * @brief Calculates the Jacobean of the PNP function w.r.t. the object coordinate inputs.
 *
 * PNP is treated as a n x 3 -> 6 fnuction, i.e. it takes n 3D coordinates and maps them to a 6D pose.
 * The Jacobean is therefore 6x3n. The Jacobean is calculated using central differences.
 *
 * @param imgPts List of 2D points.
 * @param objPts List of corresponding 3D points.
 * @param eps Epsilon used in central differences approximation.
 * @return 6x3n Jacobean matrix of partial derivatives.
 */
cv::Mat_<double> dPNP(    
    const std::vector<cv::Point2f>& imgPts,
    std::vector<cv::Point3f> objPts,
    float eps = 0.1f)
{
    int pnpMethod = (imgPts.size() == 4) ? CV_P3P : CV_ITERATIVE; // CV_ITERATIVE is unstable for 4 points, P3P does not work with many points
  
    cv::Mat_<float> camMat = GlobalProperties::getInstance()->getCamMat();
    cv::Mat_<double> jacobean(6, objPts.size() * 3);
    
    for(unsigned i = 0; i < objPts.size(); i++)
    for(unsigned j = 0; j < 3; j++)
    {
        // forward step
        if(j == 0) objPts[i].x += eps;
        else if(j == 1) objPts[i].y += eps;
        else if(j == 2) objPts[i].z += eps;

        jp::cv_trans_t cvTrans;
        safeSolvePnP(objPts, imgPts, camMat, cv::Mat(), cvTrans.first, cvTrans.second, false, pnpMethod);
        jp::jp_trans_t jpTrans = jp::cv2our(cvTrans);
        std::vector<double> fStep = Hypothesis(jpTrans.first, jpTrans.second).getRodVecAndTrans();

        // backward step
        if(j == 0) objPts[i].x -= 2 * eps;
        else if(j == 1) objPts[i].y -= 2 * eps;
        else if(j == 2) objPts[i].z -= 2 * eps;

        safeSolvePnP(objPts, imgPts, camMat, cv::Mat(), cvTrans.first, cvTrans.second, false, pnpMethod);
        jpTrans = jp::cv2our(cvTrans);
        std::vector<double> bStep = Hypothesis(jpTrans.first, jpTrans.second).getRodVecAndTrans();

        if(j == 0) objPts[i].x += eps;
        else if(j == 1) objPts[i].y += eps;
        else if(j == 2) objPts[i].z += eps;

        // gradient calculation
        for(unsigned k = 0; k < fStep.size(); k++)
            jacobean(k, i * 3 + j) = (fStep[k] - bStep[k]) / (2 * eps);

        if(containsNaNs(jacobean.col(i * 3 + j))) // check for NaNs
            return cv::Mat_<double>::zeros(6, objPts.size() * 3);
    }

    return jacobean;
}

/**
 * @brief Calculate the average of all matrix entries.
 * @param mat Input matrix.
 * @return Average of entries.
 */
double getAvg(const cv::Mat_<double>& mat)
{
    double avg = 0;
    
    for(unsigned x = 0; x < mat.cols; x++)
    for(unsigned y = 0; y < mat.rows; y++)
    {
        avg += std::abs(mat(y, x));
    }
    
    return avg / mat.cols / mat.rows;
}

/**
 * @brief Return the maximum entry of the given matrix.
 * @param mat Input matrix.
 * @return Maximum entry.
 */
double getMax(const cv::Mat_<double>& mat)
{
    double m = -1;
    
    for(unsigned x = 0; x < mat.cols; x++)
    for(unsigned y = 0; y < mat.rows; y++)
    {
        double val = std::abs(mat(y, x));
        if(m < 0 || val > m)
          m = val;
    }
    
    return m;
}

/**
 * @brief Return the median of all entries of the given matrix.
 * @param mat Input matrix.
 * @return Median entry.
 */
double getMed(const cv::Mat_<double>& mat)
{
    std::vector<double> vals;
    
    for(unsigned x = 0; x < mat.cols; x++)
    for(unsigned y = 0; y < mat.rows; y++)
	vals.push_back(std::abs(mat(y, x)));

    std::sort(vals.begin(), vals.end());
    
    return vals[vals.size() / 2];
}

/**
 * @brief Process a RGB image with the object coordinate CNN. Only a subsampling of all RGB image patches is processed.
 * @param colorData Input RGB image.
 * @param sampling Subsampling information. Each 2D location contains the pixel location in the original RGB image.
 * @param patchSize Size of RGB patches to be processed by the CNN.
 * @param patches Output parameters. List of RGB patches that were extracted according to the subsampling.
 * @param state Lua state for access to the object coordinate CNN.
 * @return Object coordinate estimation (sub sampled).
 */
jp::img_coord_t getCoordImg(
    const jp::img_bgr_t& colorData, 
    const cv::Mat_<cv::Point2i>& sampling,
    int patchSize,
    std::vector<cv::Mat_<cv::Vec3f>>& patches,
    lua_State* state)
{
    jp::img_coord_t modeImg = 
        jp::img_coord_t::zeros(sampling.rows, sampling.cols);
    
    StopWatch stopW;
    
    //assemble patches
    for(int px = 0; px < modeImg.cols * modeImg.rows; px++)
    {
        // 2D location in the object coordinate image
        int x = px % modeImg.cols;
        int y = px / modeImg.cols;

        // 2D location in the original RGB image
        int origX = sampling(y, x).x;
        int origY = sampling(y, x).y;

        // skip border patches
        if((origX < patchSize/2)
            || (origY < patchSize/2)
            || (origX > colorData.cols - patchSize/2)
            || (origY > colorData.rows - patchSize/2))
            continue;

        cv::Mat_<cv::Vec3f> patch(patchSize, patchSize);

        // extract patch
        int minX = origX - patchSize/2;
        int maxX = origX + patchSize/2;
        int minY = origY - patchSize/2;
        int maxY = origY + patchSize/2;

        for(int curX = minX; curX < maxX; curX++)
        for(int curY = minY; curY < maxY; curY++)
            patch(curY - minY, curX - minX) = colorData(curY, curX);

        patches.push_back(patch);
    }
    
    // do the prediction
    std::vector<cv::Vec3f> prediction = forward(patches, state);

    // write the prediction back to the object coordinate image
    for(unsigned i = 0; i < prediction.size(); i++)
    {
        int x = i % modeImg.cols;
        int y = i / modeImg.cols;

        modeImg(y, x) = prediction[i] * 1000; // conversion of meters to millimeters
    }
    
    std::cout << "CNN prediction took " << stopW.stop() / 1000 << "s." << std::endl;
    
    return modeImg;
}

/**
 * @brief Create a stratified subsampling of the given image.
 *
 * The input image is divided into cells and a random pixel location is chosen for each cell.
 *
 * @param inputMap Input RGB image.
 * @param targetSize Size of the sub sampled image (assumed to be square).
 * @param patchSize Size of RGB patches to be extracted later (for handling the image border).
 * @return Subsampling of the input image. Each 2D location contains the pixel location in the original RGB image.
 */
cv::Mat_<cv::Point2i> stochasticSubSample(const jp::img_bgr_t& inputMap, int targetSize, int patchSize)
{
    cv::Mat_<cv::Point2i> sampling(targetSize, targetSize);
    
    // width of stratified image cells
    float xStride = (inputMap.cols - patchSize) / (float) targetSize;
    // height of stratified image cells
    float yStride = (inputMap.rows - patchSize) / (float) targetSize;

    int sampleX = 0;
    for(float minX = patchSize/2, x = xStride + patchSize/2; x <= inputMap.cols - patchSize/2 + 1; minX = x, x+=xStride)
    {
        int sampleY = 0;
        for(float minY = patchSize/2, y = yStride + patchSize/2; y <= inputMap.rows - patchSize/2 + 1; minY = y, y+=yStride)
        {
            // choose a random pixel location within each cell
            int curX = drand(minX, x);
            int curY = drand(minY, y);

            sampling(sampleY, sampleX) = cv::Point2i(curX, curY);
            sampleY++;
        }
        sampleX++;
    }
      
    return sampling;
}

/**
 * @brief Calculate an image of reprojection errors for the given object coordinate prediction and the given pose.
 * @param hyp Pose estimate.
 * @param objectCoordinates Object coordinate estimate.
 * @param sampling Subsampling of the input image.
 * @param camMat Calibration matrix of the camera.
 * @return Image of reprojectiob errors.
 */
cv::Mat_<float> getDiffMap(
  const jp::cv_trans_t& hyp,
  const jp::img_coord_t& objectCoordinates,
  const cv::Mat_<cv::Point2i>& sampling,
  const cv::Mat& camMat)
{
    cv::Mat_<float> diffMap(sampling.size());
  
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> projections;	
    std::vector<cv::Point2f> points2D;
    std::vector<cv::Point2f> sources2D;
    
    // collect 2D-3D correspondences
    for(unsigned x = 0; x < sampling.cols; x++)
    for(unsigned y = 0; y < sampling.rows; y++)
    {
        // get 2D location of the original RGB frame
        cv::Point2f pt2D(sampling(y, x).x, sampling(y, x).y);

        // get associated 3D object coordinate prediction
        points3D.push_back(cv::Point3f(
            objectCoordinates(y, x)(0),
            objectCoordinates(y, x)(1),
            objectCoordinates(y, x)(2)));
        points2D.push_back(pt2D);
        sources2D.push_back(cv::Point2f(x, y));
    }
    
    if(points3D.empty()) return diffMap;

    // project object coordinate into the image using the given pose
    cv::projectPoints(points3D, hyp.first, hyp.second, camMat, cv::Mat(), projections);

    // measure reprojection errors
    for(unsigned p = 0; p < projections.size(); p++)
    {
        cv::Point2f curPt = points2D[p] - projections[p];
        float l = std::min(cv::norm(curPt), CNN_OBJ_MAXINPUT);
        diffMap(sources2D[p].y, sources2D[p].x) = l;
    }
    
    return diffMap;    
}

/**
 * @brief Project a 3D point into the image an measures the reprojection error.
 * @param pt Ground truth 2D location.
 * @param obj 3D point.
 * @param rot Rotation matrix of the pose.
 * @param t Translation vector of the pose.
 * @param camMat Calibration matrix of the camera.
 * @return Reprojection error in pixels.
 */
float project(const cv::Point2f& pt, const cv::Point3f& obj, const cv::Mat& rot, const cv::Point3d& t, const cv::Mat& camMat)
{
    double f = camMat.at<float>(0, 0);
    double ppx = camMat.at<float>(0, 2);
    double ppy = camMat.at<float>(1, 2);
    
    //transform point
    cv::Mat objMat = cv::Mat(obj);
    objMat.convertTo(objMat, CV_64F);
    
    objMat = rot * objMat + cv::Mat(t);
    
    // project
    double px = -f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) + ppx; // flip x because of reasons (to conform with OpenCV implementation)
    double py = f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) + ppy;

    // return error
    return std::min(std::sqrt((pt.x - px) * (pt.x - px) + (pt.y - py) * (pt.y - py)), CNN_OBJ_MAXINPUT);
}

/**
 * @brief Calculates the Jacobean of the projection function w.r.t the given 3D point, ie. the function has the form 3 -> 1
 * @param pt Ground truth 2D location.
 * @param obj 3D point.
 * @param rot Rotation matrix of the pose.
 * @param t Translation vector of the pose.
 * @param camMat Calibration matrix of the camera.
 * @return 1x3 Jacobean matrix of partial derivatives.
 */
cv::Mat_<double> dProjectdObj(const cv::Point2f& pt, const cv::Point3f& obj, const cv::Mat& rot, const cv::Point3d& t, const cv::Mat& camMat)
{
    double f = camMat.at<float>(0, 0);
    double ppx = camMat.at<float>(0, 2);
    double ppy = camMat.at<float>(1, 2);
    
    //transform point
    cv::Mat objMat = cv::Mat(obj);
    objMat.convertTo(objMat, CV_64F);
    
    objMat = rot * objMat + cv::Mat(t);

    if(std::abs(objMat.at<double>(2, 0)) < EPS) // prevent division by zero
        return cv::Mat_<double>::zeros(1, 3);

    // project
    double px = -f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) + ppx; // flip x because of reasons (to conform with OpenCV implementation)
    double py = f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) + ppy;
    
    // calculate error
    double err = std::sqrt((pt.x - px) * (pt.x - px) + (pt.y - py) * (pt.y - py));
    
    // early out if projection error is above threshold
    if(err > CNN_OBJ_MAXINPUT)
        return cv::Mat_<double>::zeros(1, 3);
    
    err += EPS; // avoid dividing by zero
    
    // derivative in x direction of obj coordinate
    double pxdx = -f * rot.at<double>(0, 0) / objMat.at<double>(2, 0) + f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 0);
    double pydx = f * rot.at<double>(1, 0) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 0);
    double dx = 0.5 / err * (2 * (pt.x - px) * -pxdx + 2 * (pt.y - py) * -pydx);

    // derivative in x direction of obj coordinate
    double pxdy = -f * rot.at<double>(0, 1) / objMat.at<double>(2, 0) + f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 1);
    double pydy = f * rot.at<double>(1, 1) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 1);
    double dy = 0.5 / err * (2 * (pt.x - px) * -pxdy + 2 * (pt.y - py) * -pydy);
    
    // derivative in x direction of obj coordinate
    double pxdz = -f * rot.at<double>(0, 2) / objMat.at<double>(2, 0) + f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 2);
    double pydz = f * rot.at<double>(1, 2) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 2);
    double dz = 0.5 / err * (2 * (pt.x - px) * -pxdz + 2 * (pt.y - py) * -pydz);	
    
    cv::Mat_<double> jacobean(1, 3);
    jacobean(0, 0) = dx;
    jacobean(0, 1) = dy;
    jacobean(0, 2) = dz;
    
    return jacobean;
}

/**
 * @brief Calculates the Jacobean of the projection function w.r.t the given 6D pose, ie. the function has the form 6 -> 1
 * @param pt Ground truth 2D location.
 * @param obj 3D point.
 * @param rot Rotation matrix of the pose.
 * @param t Translation vector of the pose.
 * @param camMat Calibration matrix of the camera.
 * @return 1x6 Jacobean matrix of partial derivatives.
 */
cv::Mat_<double> dProjectdHyp(const cv::Point2f& pt, const cv::Point3f& obj, const cv::Mat& rot, const cv::Point3d& t, const cv::Mat& camMat)
{
    double f = camMat.at<float>(0, 0);
    double ppx = camMat.at<float>(0, 2);
    double ppy = camMat.at<float>(1, 2);
    
    //transform point
    cv::Mat objMat = cv::Mat(obj);
    objMat.convertTo(objMat, CV_64F);
    
    cv::Mat eyeMat = rot * objMat + cv::Mat(t);
    
    if(std::abs(eyeMat.at<double>(2, 0)) < EPS) // prevent division by zero
        return cv::Mat_<double>::zeros(1, 6);

    // project
    double px = -f * eyeMat.at<double>(0, 0) / eyeMat.at<double>(2, 0) + ppx; // flip x because of reasons (to conform with OpenCV implementation)
    double py = f * eyeMat.at<double>(1, 0) / eyeMat.at<double>(2, 0) + ppy;
    
    // calculate error
    double err = std::sqrt((pt.x - px) * (pt.x - px) + (pt.y - py) * (pt.y - py));

    // early out if projection error is above threshold
    if(err > CNN_OBJ_MAXINPUT)
        return cv::Mat_<double>::zeros(1, 6);
    
    err += EPS; // avoid dividing by zero
    
    // derivative of the error wrt to projection
    cv::Mat_<double> dNdP = cv::Mat_<double>::zeros(1, 2);
    dNdP(0, 0) = -1 / err * (pt.x - px);
    dNdP(0, 1) = -1 / err * (pt.y - py);
    
    // derivative of projection function wrt rotation matrix
    cv::Mat_<double> dPdR = cv::Mat_<double>::zeros(2, 9);
    dPdR.row(0).colRange(0, 3) = -f * objMat.t() / eyeMat.at<double>(2, 0);
    dPdR.row(1).colRange(3, 6) = f * objMat.t() / eyeMat.at<double>(2, 0);
    dPdR.row(0).colRange(6, 9) = f * eyeMat.at<double>(0, 0) / eyeMat.at<double>(2, 0) / eyeMat.at<double>(2, 0) * objMat.t();
    dPdR.row(1).colRange(6, 9) = -f * eyeMat.at<double>(1, 0) / eyeMat.at<double>(2, 0) / eyeMat.at<double>(2, 0) * objMat.t();
    
    // derivative of the rotation matrix wrt the rodriguez vector
    cv::Mat_<double> dRdH = cv::Mat_<double>::zeros(9, 3);
    cv::Mat rod;
    cv::Rodrigues(rot, rod);
    cv::Rodrigues(rod, rot, dRdH);
    dRdH = dRdH.t();
    
    // combined derivative of the error wrt the rodriguez vector
    cv::Mat_<double> dNdH = dNdP * dPdR * dRdH;
    
    // derivative of projection wrt the translation vector
    cv::Mat_<double> dPdT = cv::Mat_<double>::zeros(2, 3);
    dPdT(0, 0) = -f / eyeMat.at<double>(2, 0);
    dPdT(1, 1) = f / eyeMat.at<double>(2, 0);
    dPdT(0, 2) = f * eyeMat.at<double>(0, 0) / eyeMat.at<double>(2, 0) / eyeMat.at<double>(2, 0);
    dPdT(1, 2) = -f * eyeMat.at<double>(1, 0) / eyeMat.at<double>(2, 0) / eyeMat.at<double>(2, 0);
    
    // combined derivative of error wrt the translation vector 
    cv::Mat_<double> dNdT = dNdP * dPdT;
    
    cv::Mat_<double> jacobean(1, 6);
    dNdH.copyTo(jacobean.colRange(0, 3));
    dNdT.copyTo(jacobean.colRange(3, 6));
    return jacobean;
}

/**
 * @brief Applies soft max to the given list of scores.
 * @param scores List of scores.
 * @return Soft max distribution (sums to 1)
 */
std::vector<double> softMax(const std::vector<double>& scores)
{
    double maxScore = 0; // substract maximum for numerical stability
    for(unsigned i = 0; i < scores.size(); i++)
        if(i == 0 || scores[i] > maxScore) maxScore = scores[i];
	
    std::vector<double> sf(scores.size());
    double sum = 0.0;
    
    for(unsigned i = 0; i < scores.size(); i++)
    {
        sf[i] = std::exp(scores[i] - maxScore);
        sum += sf[i];
    }
    for(unsigned i = 0; i < scores.size(); i++)
    {
        sf[i] /= sum;
    }
    
    return sf;
}

/**
 * @brief Calculates the Jacobean matrix of the function that maps n estimated object coordinates to a score, ie. the function has the form n x 3 -> 1. Returns one Jacobean matrix per hypothesis.
 * @param estObj Object coordinate estimation.
 * @param sampling Sub sampling of the RGB image.
 * @param points List of minimal sets. Each one (4 correspondences) defines one hypothesis.
 * @param stateObj Lua state for access to the score CNN.
 * @param jacobeans Output paramter. List of Jacobean matrices. One 1 x 3n matrix per pose hypothesis.
 * @param scoreOutputGradients Gradients w.r.t the score i.e. the gradients of the output of the score CNN.
 */
void dScore(
    jp::img_coord_t estObj, 
    const cv::Mat_<cv::Point2i>& sampling,
    const std::vector<std::vector<cv::Point2i>>& points,
    lua_State* stateObj,
    std::vector<cv::Mat_<double>>& jacobeans,
    const std::vector<double>& scoreOutputGradients)
{  
    GlobalProperties* gp = GlobalProperties::getInstance();
    cv::Mat_<float> camMat = gp->getCamMat();  
  
    int hypCount = points.size();
    
    std::vector<std::vector<cv::Point2f>> imgPts(hypCount);
    std::vector<std::vector<cv::Point3f>> objPts(hypCount);
    std::vector<jp::jp_trans_t> hyps(hypCount);
    std::vector<cv::Mat_<float>> diffMaps(hypCount);
    
    // calculate Hypotheses and reprojection error images from object coordinate estiamtions
    #pragma omp parallel for
    for(unsigned h = 0; h < hypCount; h++)
    {
        for(unsigned i = 0; i < points[h].size(); i++)
        {
            int x = points[h][i].x;
            int y = points[h][i].y;

            imgPts[h].push_back(sampling(y, x));
            objPts[h].push_back(cv::Point3f(estObj(y, x)));
        }

        // calculate hypothesis
        jp::cv_trans_t cvHyp;
        safeSolvePnP(objPts[h], imgPts[h], camMat, cv::Mat(), cvHyp.first, cvHyp.second, false, CV_P3P);
        hyps[h] = jp::cv2our(cvHyp);

        // calculate projection errors
        diffMaps[h] = getDiffMap(cvHyp, estObj, sampling, camMat);
    }
    
    // backward pass of the score cnn, returns gradients w.r.t. to image of reprojection errors
    std::vector<cv::Mat_<double>> dDiffMaps;
    backward(diffMaps, stateObj, scoreOutputGradients, dDiffMaps);
    
    // one Jacobean per hypothesis
    jacobeans.resize(hypCount);
    #pragma omp parallel for
    for(unsigned h = 0; h < hypCount; h++)
    {
        cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(1, estObj.cols * estObj.rows * 3);

        // accumulate derivate of score wrt the object coordinates that are used to calculate the pose
        cv::Mat_<double> supportPointGradients = cv::Mat_<double>::zeros(1, 12);
        cv::Mat_<double> dHdO = dPNP(imgPts[h], objPts[h]); // 6x12

        for(unsigned x = 0; x < dDiffMaps[h].cols; x++)
        for(unsigned y = 0; y < dDiffMaps[h].rows; y++)
        {
            cv::Point2f pt(sampling(y, x).x, sampling(y, x).y);
            cv::Point3f obj(estObj(y, x));

            // account for the direct influence of all object coordinates in the score
            cv::Mat_<double> dPdO = dProjectdObj(pt, obj, hyps[h].first, hyps[h].second, camMat);
            dPdO *= dDiffMaps[h](y, x);
            dPdO.copyTo(jacobean.colRange(x * dDiffMaps[h].cols * 3 + y * 3, x * dDiffMaps[h].cols * 3 + y * 3 + 3));

            // account for the indirect influence of the object coorindates that are used to calculate the pose
            cv::Mat_<double> dPdH = dProjectdHyp(sampling(y, x), cv::Point3f(estObj(y, x)), hyps[h].first, hyps[h].second, camMat);
            supportPointGradients += dDiffMaps[h](y, x) * dPdH * dHdO;
        }

        // add the accumulated derivatives for the object coordinates that are used to calculate the pose
        for(unsigned i = 0; i < points[h].size(); i++)
        {
            unsigned x = points[h][i].x;
            unsigned y = points[h][i].y;

            jacobean.colRange(x * dDiffMaps[h].cols * 3 + y * 3, x * dDiffMaps[h].cols * 3 + y * 3 + 3) += supportPointGradients.colRange(i * 3, i * 3 + 3);
        }

        jacobeans[h] = jacobean;
    }
}

/**
 * @brief Calculates the Jacobean matrix of the function that maps n estimated object coordinates to a soft max score, ie. the function has the form n x 3 -> 1. Returns one Jacobean matrix per hypothesis.
 *
 * This is the Soft maxed version of dScore (see above).
 *
 * @param estObj Object coordinate estimation.
 * @param sampling Sub sampling of the RGB image.
 * @param points List of minimal sets. Each one (4 correspondences) defines one hypothesis.
 * @param losses Loss measured for the hypotheses given by the points parameter.
 * @param sfScores Soft max probabilities for the hypotheses given by the points parameter.
 * @param stateObj Lua state for access to the score CNN.
 * @return List of Jacobean matrices. One 1 x 3n matrix per pose hypothesis.
 */
std::vector<cv::Mat_<double>> dSMScore(
    jp::img_coord_t estObj, 
    const cv::Mat_<cv::Point2i>& sampling,
    const std::vector<std::vector<cv::Point2i>>& points,
    const std::vector<double>& losses,
    const std::vector<double>& sfScores,
    lua_State* stateObj)
{
    // assemble the gradients wrt the scores, ie the gradients of soft max function
    std::vector<double> scoreOutputGradients(points.size());
        
    for(unsigned i = 0; i < points.size(); i++)
    {
        scoreOutputGradients[i] = sfScores[i] * losses[i];
        for(unsigned j = 0; j < points.size(); j++)
            scoreOutputGradients[i] -= sfScores[i] * sfScores[j] * losses[j];
    }
 
    // calculate gradients of the score function
    std::vector<cv::Mat_<double>> jacobeans;
    dScore(estObj, sampling, points, stateObj, jacobeans, scoreOutputGradients);
 
    // data conversion
    for(unsigned i = 0; i < jacobeans.size(); i++)
    {
        // reorder to points row first into rows
        cv::Mat_<double> reformat(estObj.cols * estObj.rows, 3);

        for(unsigned x = 0; x < estObj.cols; x++)
        for(unsigned y = 0; y < estObj.rows; y++)
        {
            cv::Mat_<double> patchGrad = jacobeans[i].colRange(
              x * estObj.cols * 3 + y * 3,
              x * estObj.cols * 3 + y * 3 + 3);

            patchGrad.copyTo(reformat.row(y * estObj.rows + x));
        }

        jacobeans[i] = reformat;
    }
    
    return jacobeans;
}

/**
 * @brief Helper function for calculating derivative of refinement (in dRefine()). Refines a given pose by iteratively collecting inlier correspondences and recomputing PNP on all inliers.
 *
 * The function assumes that the pose has been refined before (in processImage) and performs the exact refinement again but with changing object coordinate inputs (for gradient computation using finite differences).
 *
 * @param inlierCount Max. number of inliers collected.
 * @param refSteps Max. number of refinement steps (iterations).
 * @param inlierThreshold2D Inlier threshold in pixels.
 * @param pixelIdxs Random permuation of pixels to be checked for being an inlier in each refinement step. One permutation per refinement step.
 * @param estObj Estimated object coordiantes.
 * @param sampling Subsampling of the original RGB frame.
 * @param camMat Calibration parameters of the camera.
 * @param imgPts 2D points of the original RGB frame that define this pose.
 * @param objPts 3D object coordinates that define this pose.
 * @return Refined pose as a 6D vector (3D Rodriguez vector and 3D translation vector)
 */
std::vector<double> refine(
    int inlierCount,
    int refSteps,
    float inlierThreshold2D,
    const std::vector<std::vector<int>>& pixelIdxs,
    const jp::img_coord_t& estObj,
    const cv::Mat_<cv::Point2i>& sampling,
    const cv::Mat& camMat,
    const std::vector<cv::Point2f>& imgPts,
    const std::vector<cv::Point3f>& objPts)
{
    // reconstruct the input pose given the 2D-3D correspondences
    jp::cv_trans_t hyp;
    safeSolvePnP(objPts, imgPts, camMat, cv::Mat(), hyp.first, hyp.second, false, CV_P3P);
    cv::Mat_<float> diffMap = getDiffMap(hyp, estObj, sampling, camMat);
  
    for(unsigned rStep = 0; rStep < refSteps; rStep++) // refinement iterations
    {
        // collect 2D-3D correspondences
        std::vector<cv::Point2f> localImgPts;
        std::vector<cv::Point3f> localObjPts;

        for(unsigned idx = 0; idx < pixelIdxs[rStep].size(); idx++)
        {
            int x = pixelIdxs[rStep][idx] % diffMap.cols;
            int y = pixelIdxs[rStep][idx] / diffMap.cols;

            // inlier check
            if(diffMap(y, x) < inlierThreshold2D)
            {
                localImgPts.push_back(sampling(y, x));
                localObjPts.push_back(cv::Point3f(estObj(y, x)));
            }

            if(localImgPts.size() >= inlierCount)
            break;
        }

        if(localImgPts.size() < 50) // abort for stability: to few inliers
            break;

        // recalculate pose
        jp::cv_trans_t hypUpdate;
        hypUpdate.first = hyp.first.clone();
        hypUpdate.second = hyp.second.clone();

        if(!safeSolvePnP(localObjPts, localImgPts, camMat, cv::Mat(), hypUpdate.first, hypUpdate.second, true, (localImgPts.size() > 4) ? CV_ITERATIVE : CV_P3P))
            break; //abort if PnP fails

        if(containsNaNs(hypUpdate.first) || containsNaNs(hypUpdate.second))
            break; // abort if PnP fails

        hyp = hypUpdate;

        // recalculate pose errors
        diffMap = getDiffMap(hyp, estObj, sampling, camMat);
    }  

    // data conversion
    jp::jp_trans_t jpHyp = jp::cv2our(hyp);
    return Hypothesis(jpHyp.first, jpHyp.second).getRodVecAndTrans();
}

/**
 * @brief Calculates the Jacobean of the refinement function (see above) using central differences. The refinement function is assumed to take n object coordinates and map it to a refined 6D pose, ie 3n -> 6.
 * @param inlierCount Max. inliers collected in each refinement step.
 * @param refSteps Max. number of refinment steps (iterations).
 * @param subSampleFactor Sub sampling of object coordinate for which gradients are computed (for speed).
 * @param inlierThreshold2D Inlier threshold in pixels.
 * @param pixelIdxs Random permuation of pixels to be checked for being an inlier in each refinement step. One permutation per refinement step.
 * @param estObj Estimated object coordiantes.
 * @param sampling Subsampling of the original RGB frame.
 * @param camMat Calibration parameters of the camera.
 * @param imgPts 2D points of the original RGB frame that define this pose.
 * @param objPts 3D object coordinates that define this pose.
 * @param sampledPoints Input pose given as 4 2D locations in the sub sampled image.
 * @param inlierMap Map indicating for each pixel whether it has been an inlier in any refinement iteration. Only for these pixels Gradients are calculated (for speed)
 * @param eps Epsilon used for central differences.
 * @return Jacobean matrix of partial derivatives, size 6x3n.
 */
cv::Mat_<double> dRefine(
    int inlierCount,
    int refSteps,
    float subSampleFactor,
    float inlierThreshold2D,
    const std::vector<std::vector<int>>& pixelIdxs,
    const jp::img_coord_t& estObj,
    const cv::Mat_<cv::Point2i>& sampling,
    const cv::Mat& camMat,
    const std::vector<cv::Point2f>& imgPts,
    std::vector<cv::Point3f> objPts,
    const std::vector<cv::Point2i>& sampledPoints,
    const cv::Mat_<int>& inlierMap,
    float eps = 2.f)
{
    jp::img_coord_t localEstObj = estObj.clone();
    cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(6, localEstObj.cols * localEstObj.rows * 3);
    
    // calculate gradient wrt the initial 4 points (minimal set that define the pose)
    for(unsigned pt = 0; pt < sampledPoints.size() - 1; pt++) // skip last point, because gradient is anyway zero
    for(unsigned c = 0; c < 3; c++)
    {
        localEstObj(sampledPoints[pt].y, sampledPoints[pt].x)[c] += eps;
        if(c == 0) objPts[pt].x += eps;
        else if(c == 1) objPts[pt].y += eps;
        else objPts[pt].z += eps;

        // forward step
        std::vector<double> fStep = refine(
            inlierCount,
            refSteps,
            inlierThreshold2D,
            pixelIdxs,
            localEstObj,
            sampling,
            camMat,
            imgPts,
            objPts
        );

        localEstObj(sampledPoints[pt].y, sampledPoints[pt].x)[c] -= 2 * eps;
        if(c == 0) objPts[pt].x -= 2 * eps;
        else if(c == 1) objPts[pt].y -= 2 * eps;
        else objPts[pt].z -= 2 * eps;

        // backward step
        std::vector<double> bStep = refine(
            inlierCount,
            refSteps,
            inlierThreshold2D,
            pixelIdxs,
            localEstObj,
            sampling,
            camMat,
            imgPts,
            objPts
        );

        localEstObj(sampledPoints[pt].y, sampledPoints[pt].x)[c] += eps;
        if(c == 0) objPts[pt].x += eps;
        else if(c == 1) objPts[pt].y += eps;
        else objPts[pt].z += eps;

        // gradient calculation
        for(unsigned k = 0; k < fStep.size(); k++)
            jacobean(k, sampledPoints[pt].y * CNN_OBJ_PATCHSIZE * 3 + sampledPoints[pt].x * 3 + c) = (fStep[k] - bStep[k]) / (2 * eps);
    }
              
    // calculate gradient wrt to other points
    int inCount = 0;
    int skip = 1 / subSampleFactor;
    
    for(unsigned x = 0; x < inlierMap.cols; x++)
    for(unsigned y = 0; y < inlierMap.rows; y++)
    {
        if(inlierMap(y, x) == 0) // this pixel has never been an inlier, skip
            continue;

        inCount++;

        if(inCount % skip != 0) // only process a percentage of pixels
            continue;

        for(unsigned c = 0; c < 3; c++)
        {
            // forward step
            localEstObj(y, x)[c] += eps;

            std::vector<double> fStep = refine(
                inlierCount,
                refSteps,
                inlierThreshold2D,
                pixelIdxs,
                localEstObj,
                sampling,
                camMat,
                imgPts,
                objPts
            );

            // backward step
            localEstObj(y, x)[c] -= 2 * eps;

            std::vector<double> bStep = refine(
            inlierCount,
                refSteps,
                inlierThreshold2D,
                pixelIdxs,
                localEstObj,
                sampling,
                camMat,
                imgPts,
                objPts
            );

            localEstObj(y, x)[c] += eps;

            // gradient computation
            for(unsigned k = 0; k < fStep.size(); k++)
                jacobean(k, y * CNN_OBJ_PATCHSIZE * 3 + x * 3 + c) = (fStep[k] - bStep[k]) / (2 * eps) * skip;
        }
    }
    
    return jacobean;
}

/**
 * @brief Processes a frame, ie. predicts object coordinatge, estimates poses, selects the best one and measures the error.
 *
 * This function performs the forward pass of DSAC but also calculates many intermediate results
 * for the backward pass (ie it can be made faster if one cares only about the forward pass).
 *
 * @param imgBGR Input RGB image.
 * @param poseGT Ground truth pose.
 * @param stateRGB Lua state for access to the object coordinate CNN.
 * @param stateObj Lua state for access to the score CNN.
 * @param objHyps Number of hypotheses to be drawn.
 * @param ptCount Size of the minimal set to sample one hypothesis.
 * @param camMat Calibration parameters of the camera.
 * @param inlierThreshold2D Inlier threshold in pixels.
 * @param inlierCount Max. inlier count considered during refinement.
 * @param refSteps Max. refinement steps (iterations).
 * @param expectedLoss Output paramter. Expectation of loss of the discrete hypothesis distributions.
 * @param sfEntropy Output parameter. Shannon entropy of the soft max distribution of hypotheses.
 * @param correct Output parameter. Was the final, selected hypothesis correct?
 * @param hyps Output parameter. List of unrefined hypotheses sampled for the given image.
 * @param refHyps Output parameter. List of refined hypotheses sampled for the given image.
 * @param imgPts Output parameter. List of initial 2D pixel locations of the input RGB image. 4 pixels per hypothesis.
 * @param objPts Output parameter. List of initial 3D object coordinates. 4 points per hypothesis.
 * @param imgIdx Output parameter. List of initial pixel indices (encoded 2D locations for easier access in some data containers) of the subsampled input RGB image. 4 indices per hypothesis.
 * @param patches Output parameter. List of patches for which object coordinate have been estimated.
 * @param sfScores Output parameter. Soft max distribution for the sampled hypotheses.
 * @param estObj Output parameter. Estimated object coordinates (subsampling of the complete image).
 * @param sampling Output parameter. Subsampling of the RGB image.
 * @param sampledPoints Output parameter. List of initial 2D pixel locations of the subsampled input RGB image. 4 pixels per hypothesis.
 * @param losses Output parameter. List of losses of the sampled hypotheses.
 * @param inlierMaps Output parameter. Maps indicating which pixels of the subsampled input image have been inliers during hypothesis refinement, one map per hypothesis.
 * @param pixelIdxs Output parameter. List of indices of pixels of the subsampled input image that have been inliers during hypothesis refinement, one list per hypothesis.
 * @param tErr Output parameter. Translational (in mm) error of the final, selected hypothesis.
 * @param rotErr Output parameter. Rotational error of the final, selected hypothesis.
 * @param hypIdx Output parameter. Index of the final, selected hypothesis.
 */
void processImage(
    const jp::img_bgr_t& imgBGR,
    const Hypothesis& poseGT,
    lua_State* stateRGB,
    lua_State* stateObj,
    int objHyps,
    int ptCount,
    const cv::Mat& camMat,
    int inlierThreshold2D,
    int inlierCount,
    int refSteps,
    double& expectedLoss,
    double& sfEntropy,
    bool& correct,
    std::vector<jp::cv_trans_t>& hyps,
    std::vector<jp::cv_trans_t>& refHyps,
    std::vector<std::vector<cv::Point2f>>& imgPts,
    std::vector<std::vector<cv::Point3f>>& objPts,
    std::vector<std::vector<int>>& imgIdx,
    std::vector<cv::Mat_<cv::Vec3f>>& patches,
    std::vector<double>& sfScores,
    jp::img_coord_t& estObj,
    cv::Mat_<cv::Point2i>& sampling,
    std::vector<std::vector<cv::Point2i>>& sampledPoints,
    std::vector<double>& losses,
    std::vector<cv::Mat_<int>>& inlierMaps,
    std::vector<std::vector<std::vector<int>>>& pixelIdxs,
    double& tErr,
    double& rotErr,
    int& hypIdx)
{
    std::cout << BLUETEXT("Predicting object coordinates.") << std::endl;
    StopWatch stopW;  
    
    // generate subsampling of input image for speed
    sampling = stochasticSubSample(imgBGR, CNN_OBJ_PATCHSIZE, CNN_RGB_PATCHSIZE);
    patches.clear();
    // get object coordinate predictions for subsampled image locations
    estObj = getCoordImg(imgBGR, sampling, CNN_RGB_PATCHSIZE, patches, stateRGB);

    std::cout << BLUETEXT("Sampling " << objHyps << " hypotheses.") << std::endl;

    hyps.resize(objHyps);
    imgPts.resize(objHyps);
    objPts.resize(objHyps);
    sampledPoints.resize(objHyps);
    
    // keep track of the points each hypothesis is samples from
    // the index references the patch list above
    imgIdx.resize(objHyps); 
  
    #pragma omp parallel for
    for(unsigned h = 0; h < hyps.size(); h++)
    while(true)
    {
        std::vector<cv::Point2f> projections;
        cv::Mat_<uchar> alreadyChosen = cv::Mat_<uchar>::zeros(estObj.size()); // no points should appear more than once in a minimal set
        imgPts[h].clear();
        objPts[h].clear();
        imgIdx[h].clear();
        sampledPoints[h].clear();

        for(int j = 0; j < ptCount; j++)
        {
            // 2D location in the subsampled image
            int x = irand(0, estObj.cols);
            int y = irand(0, estObj.rows);

            if(alreadyChosen(y, x) > 0)
            {
                j--;
                continue;
            }

            alreadyChosen(y, x) = 1;

            imgPts[h].push_back(sampling(y, x)); // 2D location in the original RGB image
            objPts[h].push_back(cv::Point3f(estObj(y, x))); // 3D object coordinate
            imgIdx[h].push_back(y * CNN_OBJ_PATCHSIZE + x); // pixel index in the subsampled image (for easier access in some data containers)
            sampledPoints[h].push_back(cv::Point2i(x, y)); // 2D pixel location in the subsampled image
        }

        // solve PNP
        if(!safeSolvePnP(objPts[h], imgPts[h], camMat, cv::Mat(), hyps[h].first, hyps[h].second, false, CV_P3P))
            continue; // abort if PNP failes

        // project 3D points back into the image
        cv::projectPoints(objPts[h], hyps[h].first, hyps[h].second, camMat, cv::Mat(), projections);

        // check reconstruction, 4 sampled points should be reconstructed perfectly
        bool foundOutlier = false;
        for(unsigned j = 0; j < imgPts[h].size(); j++)
        {
            if(cv::norm(imgPts[h][j] - projections[j]) < inlierThreshold2D) continue;
            foundOutlier = true;
            break;
        }
        if(foundOutlier)
            continue;
        else
            break;
    }	

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
    std::cout << BLUETEXT("Calculating scores.") << std::endl;

    // compute reprojection error images
    std::vector<cv::Mat_<float>> diffMaps(objHyps);
    #pragma omp parallel for 
    for(unsigned h = 0; h < hyps.size(); h++)
        diffMaps[h] = getDiffMap(hyps[h], estObj, sampling, camMat);

    // execute score CNN to get hypothesis scores
    std::vector<double> scores = forward(diffMaps, stateObj);
    
    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
    std::cout << BLUETEXT("Drawing final Hypothesis.") << std::endl;	
    
    // apply soft max to scores to get a distribution
    sfScores = softMax(scores);
    sfEntropy = entropy(sfScores); // measure distribution entropy
    hypIdx = draw(sfScores); // select winning hypothesis
    
    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
    std::cout << BLUETEXT("Refining poses:") << std::endl;

    // we refine all hypothesis because we are interested in the expectation of loss
    refHyps.resize(hyps.size());
    
    #pragma omp parallel for 
    for(unsigned h = 0; h < hyps.size(); h++)
    {
        refHyps[h].first = hyps[h].first.clone();
        refHyps[h].second = hyps[h].second.clone();
    }
    
    // collect inliers for all hypotheses
    inlierMaps.resize(hyps.size());
    pixelIdxs.resize(hyps.size());
    
    #pragma omp parallel for 
    for(unsigned h = 0; h < hyps.size(); h++)
    {         
        std::mt19937 randG;
        cv::Mat_<float> localDiffMap = diffMaps[h];
        inlierMaps[h] = cv::Mat_<int>::zeros(diffMaps[h].size());
        pixelIdxs[h].resize(refSteps);

        for(unsigned rStep = 0; rStep < refSteps; rStep++)
        {
            // generate a random permutation of pixels to be checked for being inliers
            for(unsigned idx = 0; idx < localDiffMap.cols * localDiffMap.rows; idx++)
                pixelIdxs[h][rStep].push_back(idx);
            std::shuffle(pixelIdxs[h][rStep].begin(), pixelIdxs[h][rStep].end(), randG);

            std::vector<cv::Point2f> localImgPts;
            std::vector<cv::Point3f> localObjPts;

            for(unsigned idx = 0; idx < pixelIdxs[h][rStep].size(); idx++)
            {
                int x = pixelIdxs[h][rStep][idx] % localDiffMap.cols;
                int y = pixelIdxs[h][rStep][idx] / localDiffMap.cols;

                // inlier check
                if(localDiffMap(y, x) < inlierThreshold2D)
                {
                    localImgPts.push_back(sampling(y, x));
                    localObjPts.push_back(cv::Point3f(estObj(y, x)));
                    inlierMaps[h](y, x) = inlierMaps[h](y, x) + 1;
                 }

                if(localImgPts.size() >= inlierCount)
                    break; // max number of inlier reached
            }

            if(localImgPts.size() < 50)
                break; // abort for stability: too few inliers

            // recalculate pose
            jp::cv_trans_t hypUpdate;
            hypUpdate.first = refHyps[h].first.clone();
            hypUpdate.second = refHyps[h].second.clone();

            if(!safeSolvePnP(localObjPts, localImgPts, camMat, cv::Mat(), hypUpdate.first, hypUpdate.second, true, (localImgPts.size() > 4) ? CV_ITERATIVE : CV_P3P))
                break; //abort if PnP fails

            if(containsNaNs(hypUpdate.first) || containsNaNs(hypUpdate.second))
                break; // abort if PnP fails

            refHyps[h] = hypUpdate;

            // recalculate pose errors
            localDiffMap = getDiffMap(refHyps[h], estObj, sampling, camMat);
        }

        // set the initial minimal set per hypothesis to not being an inlier (for finite differences later)
        for(unsigned pt = 0; pt < sampledPoints[h].size(); pt++)
        {
            int x = sampledPoints[h][pt].x;
            int y = sampledPoints[h][pt].y;
            inlierMaps[h](y, x) = 0;
        }
    }
       
    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
    std::cout << BLUETEXT("Final Result:") << std::endl;
    
    // evaluate result
    jp::jp_trans_t jpHyp = jp::cv2our(refHyps[hypIdx]);
    Hypothesis poseEst(jpHyp.first, jpHyp.second);	
    
    expectedLoss = expectedMaxLoss(poseGT, refHyps, sfScores, losses);
    std::cout << "Loss of winning hyp: " << maxLoss(poseGT, poseEst) << ", prob: " << sfScores[hypIdx] << ", expected loss: " << expectedLoss << std::endl;
    
    // measure pose error in the inverted system (scene pose vs camera pose)
    Hypothesis invPoseGT = getInvHyp(poseGT);
    Hypothesis invPoseEst = getInvHyp(poseEst);

    rotErr = invPoseGT.calcAngularDistance(invPoseEst);
    tErr = cv::norm(invPoseEst.getTranslation() - invPoseGT.getTranslation());

    correct = false;
    if(rotErr < 5 && tErr < 50)
    {
        std::cout << GREENTEXT("Rotation Err: " << rotErr << ", Translation Err: " << tErr) << std::endl << std::endl;
        correct = true;
    }
    else
        std::cout << REDTEXT("Rotation Err: " << rotErr << ", Translation Err: " << tErr) << std::endl << std::endl;
}
