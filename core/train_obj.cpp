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

#include "dataset.h"
#include "stop_watch.h"
#include "thread_rand.h"
#include "lua_calls.h"
#include "cnn.h"

#include <fstream>

#define DOVALIDATION 0 // compile flag, if true a validation set has to be available

/**
 * @brief Generates a list of RGB patches with associated groud truth object coordinates
 * @param trainingImages From how many (random) images should the list be generated?
 * @param trainingPatches How many patches should be extracted per image?
 * @param inputSize How large are the RGB patches (assumed to be square).
 * @param dataset Dataset.
 * @param data Output parameter. List of RGB patches.
 * @param labels Output parameter. List of ground truth object coordinates.
 */
void assembleData(
    int trainingImages,
    int trainingPatches,
    int inputSize,
    const jp::Dataset& dataset,
    std::vector<cv::Mat_<cv::Vec3f>>& data,
    std::vector<cv::Vec3f>& labels)
{
    StopWatch stopW;
    
    data.resize(trainingImages * trainingPatches);
    labels.resize(trainingImages * trainingPatches);
    
    #pragma omp parallel for 
    for(int i = 0; i < trainingImages; i++)
    {
        int imgIdx = irand(0, dataset.size()); // choose a random image

        // RGB image
        jp::img_bgr_t imgBGR;
        dataset.getBGR(imgIdx, imgBGR);

        // Ground truth object coordinates
        jp::img_coord_t imgObj;
        dataset.getObj(imgIdx, imgObj);

        // extract patches
        for(int j = 0; j < trainingPatches; j++)
        while(true)
        {
            int x = irand(inputSize/2, imgBGR.cols - inputSize/2); // no patch should reach outside the image
            int y = irand(inputSize/2, imgBGR.rows - inputSize/2);

            if(!jp::onObj(imgObj(y, x))) continue;
            data[i * trainingPatches + j] = cv::Mat_<cv::Vec3f>(inputSize, inputSize);

            int minX = x - inputSize/2;
            int maxX = x + inputSize/2;
            int minY = y - inputSize/2;
            int maxY = y + inputSize/2;

            // copy patch
            for(int curX = minX; curX < maxX; curX++)
            for(int curY = minY; curY < maxY; curY++)
                data[i * trainingPatches + j](curY - minY, curX - minX) = imgBGR(curY, curX);

            // store object coordinate of center pixel
            labels[i * trainingPatches + j] = imgObj(y, x);
            labels[i * trainingPatches + j] /= 1000; // conversion of millimeters to meters

            break;
        }
    }
	
    std::cout << "Generated " << data.size() << " patches from " << trainingImages << " images in " << stopW.stop() << "ms." << std::endl;
}

/**
 * @brief Calculate the ratio of object coorindate inliers.
 * @param est List of estimated object coordinates.
 * @param gt List of ground truth object coordinates.
 * @param threshold Distance threshold.
 * @return Ratio of correctly estimated inliers.
 */
float getInliers(const std::vector<cv::Vec3f>& est, const std::vector<cv::Vec3f>& gt, float threshold)
{
    float inliers = 0;
    
    for(unsigned i = 0; i < est.size(); i++)
    {
        if(cv::norm(est[i] - gt[i]) < threshold)
            inliers += 1;
    }
    
    return inliers / est.size();
}

/**
 * @brief Extracts a subset of the given data into a batch of data.
 * @param offset Index offset wrt the complete data, where batch extraction should be started.
 * @param size Batch size.
 * @param permutation List of shuffled indices of the complete data. Batch is extracted from the shuffled data.
 * @param data List of RGB patches.
 * @param labels List of associated ground truth object coordinates.
 * @param batchData Output parameter. Batch of RGB patches.
 * @param batchLabels Output parameter. Batch of associated ground truth object coordinates.
 */
void assembleBatch(
    int offset, 
    int size, 
    const std::vector<int>& permutation, 
    const std::vector<cv::Mat_<cv::Vec3f>>& data, 
    const std::vector<cv::Vec3f>& labels, 
    std::vector<cv::Mat_<cv::Vec3f>>& batchData, 
    std::vector<cv::Vec3f>& batchLabels)
{
    batchData.resize(size);
    batchLabels.resize(size);
    
    for(unsigned i = 0; i < size; i++)
    {
        batchData[i] = data[permutation[i+offset]];
        batchLabels[i] = labels[permutation[i+offset]];
    }
}

/**
 * @brief Performs a forward backward pass with the given data and ground truth.
 * @param maps Batch of RGB patches.
 * @param labels Associated ground truth object coordinates.
 * @param state Lua state for access to the object coordinate CNN.
 * @return Loss of the forward pass.
 */
double train(const std::vector<cv::Mat_<cv::Vec3f>>& maps, const std::vector<cv::Vec3f>& labels, lua_State* state)
{
    lua_getglobal(state, "train"); 
    lua_pushinteger(state, maps.size());
    pushMaps(maps, state);
    pushVec(labels, state);
    lua_pcall(state, 3, 1, 0);
    
    double loss = lua_tonumber(state, -1);
    lua_pop(state, 1);
    
    return loss;
}

/**
 * @brief Measure CNN loss of given prediction.
 * @param pred List of object coordiante predictions.
 * @param labels List of ground truth object coordinates.
 * @param state Lua state for access to the object coordiante CNN.
 * @return Loss wrt the predictions.
 */
double getLoss(const std::vector<cv::Vec3f>& pred, const std::vector<cv::Vec3f>& labels, lua_State* state)
{
    lua_getglobal(state, "getLoss"); 
    lua_pushinteger(state, pred.size());
    pushVec(pred, state);
    pushVec(labels, state);
    lua_pcall(state, 3, 1, 0);
    
    double loss = lua_tonumber(state, -1);
    lua_pop(state, 1);
    
    return loss;
}

int main(int argc, const char* argv[])
{
    int inputSize = 42; // RGB patch size
    int channels = 3; // number of RGB channels
    
    int trainingImages = 100; // number of training images randonly chosen in each training round
    int trainingPatches = 512; // number of patches extracted from each training image
    int trainingLimit = 300000; // number of parameter updates performed
    int batchSize = 64; // training batch size
    
    int validationImages = 100; // number of validation images randomly chosen from the validation set
    int validationPatches = 512; // number of patches checked per validation image

    std::mt19937 randG;
    
    // parse config file and command line properties to set parameters
    GlobalProperties* gp = GlobalProperties::getInstance();
    gp->parseConfig();
    gp->parseCmdLine(argc, argv);
   
    // lua script used
    std::string baseScriptRGB = gp->dP.objScript;

     // inlier threshold to measure performance of validation set
    float inlierThreshold = gp->pP.ransacInlierThreshold3D / 1000.f; //in mm

    // load training and validation set
    std::string dataDir = "./";
    std::string trainingDir = dataDir + "training/";

    std::vector<std::string> trainingSets = getSubPaths(trainingDir);
    std::cout << std::endl << BLUETEXT("Loading training set ...") << std::endl;
    jp::Dataset trainDataset = jp::Dataset(trainingSets[0], 1);

    #if DOVALIDATION
    std::string validationDir = dataDir + "validation/";

    std::vector<std::string> validationSets = getSubPaths(validationDir);
    std::cout << std::endl << BLUETEXT("Loading validation set ...") << std::endl;
    jp::Dataset valDataset = jp::Dataset(validationSets[0], 1);
    #endif
    
    // initialize data
    std::vector<cv::Mat_<cv::Vec3f>> data;
    std::vector<cv::Mat_<cv::Vec3f>> batchData;
    std::vector<cv::Mat_<cv::Vec3f>> valData;
    std::vector<cv::Vec3f> labels;
    std::vector<cv::Vec3f> batchLabels;
    std::vector<cv::Vec3f> valLabels;
    
    // lua and model setup
    lua_State* state = luaL_newstate();
    luaL_openlibs(state);
    
    execute(baseScriptRGB.c_str(), state);
    constructModel(channels, inputSize, state);
    setEvaluate(state);

    // pre-load validation data
    #if DOVALIDATION
    std::cout << GREENTEXT("Choosing Validation data.") << std::endl;
	assembleData(validationImages, validationPatches, inputSize, valDataset, valData, valLabels);    
    #endif
    
    // permutate training data
    std::vector<int> trainPermutation(trainingImages * trainingPatches);
    for(unsigned i = 0; i < trainPermutation.size(); i++)
    trainPermutation[i] = i;
    std::shuffle(trainPermutation.begin(), trainPermutation.end(), randG);    
    
    #if DOVALIDATION
    std::vector<int> valPermutation(valData.size());
    for(unsigned i = 0; i < valData.size(); i++)
    valPermutation[i] = i; // validation data does not need to be permutated
    #endif
    
    std::cout << GREENTEXT("Training CNN.") << std::endl;
    
    // output files for training statistics
    std::ofstream trainFile;
    trainFile.open("training_loss_"+baseScriptRGB+".txt");

    #if DOVALIDATION
    std::ofstream valFile;
    valFile.open("validation_loss_"+baseScriptRGB+".txt");
    #endif
    
    int trainCounter = 0;
    int round = 0;
    
    while(trainCounter <= trainingLimit)
    {
        std::cout << BLUETEXT("Starting training round " << ++round) << std::endl;
        setTraining(state);
        assembleData(trainingImages, trainingPatches, inputSize, trainDataset, data, labels); // load data from the training set

        // process training data batch-wise
        for(int b = 0; b < data.size() / batchSize; b++)
        {
            assembleBatch(b * batchSize, batchSize, trainPermutation, data, labels, batchData, batchLabels);

            float loss = train(batchData, batchLabels, state); // forward-backward

            trainFile << trainCounter++ << " " << loss << std::endl;
            std::cout << YELLOWTEXT("Training loss: " << loss) << std::endl;
        }

        #if DOVALIDATION
        float valLoss = 0;
        float valInliers = 0;

        std::cout << YELLOWTEXT("Validation pass.") << std::endl;
        setEvaluate(state);

        // process validation data batch-wise
        for(int b = 0; b < (valData.size() / batchSize); b++)
        {
            assembleBatch(b * batchSize, batchSize, valPermutation, valData, valLabels, batchData, batchLabels);

            std::vector<cv::Vec3f> pred = forward(batchData, state);
            valLoss += getLoss(pred, batchLabels, state);
            valInliers += getInliers(pred, batchLabels, inlierThreshold);
        }

        valLoss /= (valData.size() / batchSize);
        valInliers /= (valData.size() / batchSize);

        valFile << (trainCounter-1) << " " << valLoss << " " << valInliers << std::endl;
        std::cout << YELLOWTEXT("Validation loss: " << valLoss << " (Inliers: " << valInliers * 100 << "%)") << std::endl;
        #endif
    }
    
    trainFile.close();

    #if DOVALIDATION
    valFile.close();
    #endif

    lua_close(state);  
    return 0;
}
