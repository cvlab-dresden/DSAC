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

#include "properties.h"
#include "types.h"
#include "read_data.h"
#include "Hypothesis.h"
#include "util.h"
#include <stdexcept>

/** Interface for reading and writing datasets and some basis operations.*/

namespace jp
{
    /**
     * @brief Calculate the camera coordinate given a pixel position and a depth value.
     * 
     * @param x X component of the pixel position.
     * @param y Y component of the pixel position.
     * @param depth Depth value at that position in mm.
     * @return jp::coord3_t Camera coordinate.
     */
    jp::coord3_t pxToEye(int x, int y, jp::depth_t depth);

    /**
     * @brief Checks whether the given object coordinate lies on the object (is not 0 0 0).
     * 
     * @param pt Object coordinate.
     * @return bool True if not background.
     */
    bool onObj(const jp::coord3_t& pt); 
    
    /**
     * @brief Class that is a interface for reading and writing object specific data.
     * 
     */
    class Dataset
    {
        public:

        Dataset()
        {
        }

        /**
         * @brief Constructor.
         *
         * @param basePath The directory where there are subdirectories "rgb_noseg", "depth_noseg", "seg", "obj", "info".
         * @param objID Object ID this dataset belongs to.
         */
        Dataset(const std::string& basePath, jp::id_t objID) : objID(objID)
        {
            readFileNames(basePath);
        }

        /**
         * @brief Calculates the 2D points in the RGB image that corresponds to the given 2D point in the depth images.
         *
         * The function calculates the camera coordinate for the given pixel, applies the sensor
         * transformation (see properties.h) and projects back into the image.
         *
         * @param x X position of the point in the depth frame.
         * @param y Y position of the poibt in the depth frame.
         * @param depth Depth of the point.
         * @return Position of the corresponding point in the RGB frame.
         */
        cv::Point2f mapDepthToRGB(int x, int y, short depth) const
        {
            GlobalProperties* gp = GlobalProperties::getInstance();

            // project to 3D point in camera coordinates (with intrinsic parameters of the depth sensor)
            cv::Mat_<double> eye = cv::Mat_<double>::ones(4, 1);
            eye(0, 0) = ((x - (gp->dP.imageWidth / 2.0 + gp->dP.rawXShift)) / (gp->dP.secondaryFocalLength / (double) depth));
            eye(1, 0) = -((y - (gp->dP.imageHeight / 2.0 + gp->dP.rawYShift)) / (gp->dP.secondaryFocalLength / (double) depth));
            eye(2, 0) = -depth;

            // apply relative transformation between sensors
            eye = gp->dP.sensorTrans * eye;

            // project to 2D point in image coordiantes (with intrinsic parameters of the rgb sensor)
            int newX = (eye(0, 0) * (gp->dP.focalLength / (double) depth)) + (gp->dP.imageWidth / 2.f + gp->dP.xShift) + 0.5;
            int newY = -(eye(1, 0) * (gp->dP.focalLength / (double) depth)) + (gp->dP.imageHeight / 2.f + gp->dP.yShift) + 0.5;

            return cv::Point2f(newX, newY);
        }

        /**
         * @brief Return the object ID this dataset belongs to.
         *
         * @return jp::id_t Object ID.
         */
        jp::id_t getObjID() const
        {
            return objID;
        }

        /**
         * @brief Size of the dataset (number of frames).
         *
         * @return size_t Size.
         */
        size_t size() const
        {
            return bgrFiles.size();
        }

        /**
         * @brief Return the RGB image file name of the given frame number.
         *
         * @param i Frame number.
         * @return std::string File name.
         */
        std::string getFileName(size_t i) const
        {
            return bgrFiles[i];
        }

        /**
         * @brief Get ground truth information for the given frame.
         *
         * @param i Frame number.
         * @return bool Returns if there is no valid ground truth for this frame (object not visible).
         */
        bool getInfo(size_t i, jp::info_t& info) const
        {
            if(infoFiles.empty()) return false;
            if(!readData(infoFiles[i], info)) return false;
            return true;
        }

        /**
         * @brief Get the RGB image of the given frame.
         *
         * @param i Frame number.
         * @param img Output parameter. RGB image.
         * @return void
         */
        void getBGR(size_t i, jp::img_bgr_t& img) const
        {
            std::string bgrFile = bgrFiles[i];
            readData(bgrFile, img);
        }

        /**
         * @brief Get the depth image of the given frame.
         *
         * If RGB and Depth are not registered (rawData flag in properties.h), Depth will be
         * mapped to RGB using calibration parameters and the external sensor transformation matrix.
         *
         * @param i Frame number.
         * @param img Output parameter. depth image.
         * @return void
         */
        void getDepth(size_t i, jp::img_depth_t& img) const
        {
            std::string dFile = depthFiles[i];

            readData(dFile, img);

            if(GlobalProperties::getInstance()->dP.rawData)
            {
                jp::img_depth_t depthMapped = jp::img_depth_t::zeros(img.size());

                for(unsigned x = 0; x < img.cols; x++)
                for(unsigned y = 0; y < img.rows; y++)
                {
                    jp::depth_t depth = img(y, x);
                    if(depth == 0) continue;

                    cv::Point2f pt = mapDepthToRGB(x, y, depth);
                    depthMapped(pt.y, pt.x) = depth;
                }

                img = depthMapped;
            }
        }

        /**
         * @brief Get the RGB-D image of the given frame.
         *
         * @param i Frame number.
         * @param img Output parameter. RGB-D image.
         * @return void
         */
        void getBGRD(size_t i, jp::img_bgrd_t& img) const
        {
            getBGR(i, img.bgr);
            getDepth(i, img.depth);
        }

        /**
         * @brief Get the ground truth object coordinate image of the given frame.
         *
         * Object coordinates will be generated from image depth and the ground truth pose.
         *
         * @param i Frame number.
         * @param img Output parameter. Object coordinate image.
         * @return void
         */
        void getObj(size_t i, jp::img_coord_t& img) const
        {
            jp::img_depth_t depthData;
            getDepth(i, depthData);

            jp::info_t poseData;
            getInfo(i, poseData);

            Hypothesis h(poseData);

            img = jp::img_coord_t(depthData.rows, depthData.cols);

            #pragma omp parallel for
            for(unsigned x = 0; x < img.cols; x++)
            for(unsigned y = 0; y < img.rows; y++)
            {
                if(depthData(y, x) == 0)
                {
                    img(y, x) = jp::coord3_t(0, 0, 0);
                    continue;
                }

                img(y, x) = pxToEye(x, y, depthData(y, x));

                cv::Point3d pt = h.invTransform(cv::Point3d(img(y, x)[0], img(y, x)[1], img(y, x)[2]));
                img(y, x)[0] = pt.x;
                img(y, x)[1] = pt.y;
                img(y, x)[2] = pt.z;
            }
        }

        /**
         * @brief Get the camera coordinate image of the given frame (generated from the depth channel).
         *
         * @param i Frame number.
         * @param img Output parameter. Camera coordinate image.
         * @return void
         */
        void getEye(size_t i, jp::img_coord_t& img) const
        {
            jp::img_depth_t imgDepth;
            getDepth(i, imgDepth);

            img = jp::img_coord_t(imgDepth.rows, imgDepth.cols);

            #pragma omp parallel for
            for(int x = 0; x < img.cols; x++)
            for(int y = 0; y < img.rows; y++)
            {
               img(y, x) = pxToEye(x, y, imgDepth(y, x));
            }
        }

        private:

          /**
           * @brief Reads all file names in the various sub-folders of a dataset.
           *
           * @param basePath Folder where all data sub folders lie.
           * @return void
           */
          void readFileNames(const std::string& basePath)
        {
            std::cout << "Reading file names... " << std::endl;
            std::string bgrPath = "/rgb_noseg/", bgrSuf = ".png";
            std::string dPath = "/depth_noseg/", dSuf = ".png";
            std::string infoPath = "/poses/", infoSuf = ".txt";

            bgrFiles = getFiles(basePath + bgrPath, bgrSuf);
            depthFiles = getFiles(basePath + dPath, dSuf);
            infoFiles = getFiles(basePath + infoPath, infoSuf, true);
        }

        jp::id_t objID; // object ID this dataset belongs to

        // image data files
        std::vector<std::string> bgrFiles; // list of RGB files
        std::vector<std::string> depthFiles; // list of depth files
        // groundtruth data files
        std::vector<std::string> infoFiles; // list of ground truth annotation files
    };
}
