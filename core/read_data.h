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

/** Read several custom data formats. */

namespace jp
{
    /**
    * @brief Read a depth image.
    * 
    * Depth images are stored as 1 channel, 16 bit, unsigned short PNGs.
    * 
    * @param dFile Name of the file to read including the path.
    * @param image Output parameter. Depth image to read.
    * @return void
    */
    void readData(const std::string dFile, jp::img_depth_t& image);
    
    /**
    * @brief Read a bgr image.
    * 
    * BGR images are stored as 3 channel, 8 bit, unsigned char PNGs. Channels are swapped from RGB.
    * 
    * @param bgrFile Name of the file to read including the path.
    * @param image Output parameter. BGR image to read.
    * @return void
    */
    void readData(const std::string bgrFile, jp::img_bgr_t& image);
    
    /**
    * @brief Reads an image with BGR channels and a depth channel.
    * 
    * BGR and depth are read from separate files. See documentation of the respective readData methods.
    * 
    * @param bgrFile Name of the file to read for the BGR image including the path.
    * @param dFile Name of the file to read for the depth image including the path.
    * @param image Output parameter. RGBD image to read.
    * @return void
    */
    void readData(const std::string bgrFile, const std::string dFile, jp::img_bgrd_t& image);
    
    /**
     * @brief Read an info file.
     * 
     * @param infoFile Name of the file to read including the path.
     * @param info Output parameter. Info object to read. 
     * @return void
     */
    bool readData(const std::string infoFile, jp::info_t& info);
}
