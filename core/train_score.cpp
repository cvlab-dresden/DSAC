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

#include <iostream>

#include "properties.h"
#include "dataset.h"
#include "thread_rand.h"
#include "util.h"
#include "stop_watch.h"

#include "lua_calls.h"
#include "cnn.h"
#include <fstream>

#define DOVALIDATION 0 // compile flag, if true a validation set has to be available

/**
 * @brief Returns a random 6D pose hypothesis.
 * @param gaussRot Standard deviation from which the rotation angle is sampled from. Rotation axis is sampled from a uniform distribution.
 * @param gaussTrans Standard deviation from which the X, Y, Z components of the translation are sampled.
 * @return Random pose.
 */
Hypothesis getRandHyp(float gaussRot, float gaussTrans)
{  
    // sample random translation
    cv::Point3d trans(
	dgauss(0, gaussTrans), 
	dgauss(0, gaussTrans), 
	dgauss(0, gaussTrans));
    
    // sample random rotation axis
    cv::Point3d rotAxis(drand(0, 1), drand(0, 1), drand(0, 1));
    rotAxis = rotAxis * (1.0 / cv::norm(rotAxis));

    // sample random rotation angle
    rotAxis = rotAxis * (dgauss(0, gaussRot)) * (PI / 180.0);
    
    std::vector<double> v(6);
    v[0] = rotAxis.x;
    v[1] = rotAxis.y;
    v[2] = rotAxis.z;
    v[3] = trans.x;
    v[4] = trans.y;
    v[5] = trans.z;
    
    return Hypothesis(v);
}

/**
 * @brief Calculates the loss of the given predictions wrt the given labels. The LUA loss function is used.
 * @param pred List of scalar predictions.
 * @param labels List of scalar ground truth.
 * @param state LUA state which contains the loss function.
 * @return Loss over all predictions.
 */
double getLoss(const std::vector<double>& pred, const std::vector<double>& labels, lua_State* state)
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

/**
 * @brief Perform a CNN training pass with the given input and given labels.
 * @param maps List of single channel image patches.
 * @param labels List of scalar ground truth labels.
 * @param state LUA state for access to the CNN.
 * @return  Loss of the forward pass.
 */
double train(const std::vector<cv::Mat_<float>>& maps, const std::vector<double>& labels, lua_State* state)
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
 * @brief Checks whether the entry with the highest prediction is correct w.r.t the 5cm5deg pose threshold.
 * @param predictions List of scalar predictions.
 * @param labels List of scalar ground truth labels. Its assumed that a label is calculated: -temperature * max(rotErr, transErr).
 * @param temperature Temperature that was used to calculate the labels (see above).
 * @return One if highest prediction corresponds to a correct pose, zero otherwise.
 */
int selectBest(const std::vector<double>& predictions, const std::vector<double>& labels, float temperature)
{
    // choose entry with highest prediction
    double bestPrediction;
    double bestLabel;
    
    for(int i = 0; i < predictions.size(); i++)
    {
       if(i == 0 || predictions[i] > bestPrediction)
       {
          bestPrediction = predictions[i];
          bestLabel = labels[i];
       }
    }    

    // check whether corresponding labels indicates a correct pose
    if(bestLabel > -temperature * 5)
    {
        std::cout << GREENTEXT("Selected pose was correct!") << std::endl;
        return 1;
    }
    else
    {
        std::cout << REDTEXT("Selected pose was wrong!") << std::endl;
        return 0;
    }
}

/**
 * @brief Extract a list of reprojection error images with associated score ground truth values.
 *
 * Score ground truth is the negative pose error value times a scaling factor (temperature).
 *
 * @param imageCount How many different input images should be used to extract reprojection error images?
 * @param hypsPerImage How many reprojection error images should be generated per input image?
 * @param objInputSize Size of the reprojection error images (assumed to be square).
 * @param rgbInputSize Size of the input RGB patches.
 * @param dataset Dataset to load images and ground truth.
 * @param state LUA state for access to the object coordiante CNN.
 * @param data Output parameter. List of reprojection error images.
 * @param labels Output parameter. List of associated ground truth score values.
 * @param temperature Scaling factor for ground truth score values (affect the sharpness of the soft max distribution in the full pipeline).
 */
void assembleData(
    int imageCount,
    int hypsPerImage,
    int objInputSize,
    int rgbInputSize,  
    const jp::Dataset& dataset,
    lua_State* state,
    std::vector<cv::Mat_<float>>& data,
    std::vector<double>& labels,
    float temperature)
{
    GlobalProperties* gp = GlobalProperties::getInstance();
    cv::Mat_<float> camMat = gp->getCamMat(); // camera calibration parameters

    float correct = 0;
    StopWatch stopW;
    
    data.resize(imageCount * hypsPerImage);
    labels.resize(imageCount * hypsPerImage);
    
    for(int i = 0; i < imageCount; i++)
    {
        // load random frame and ground truth pose
        int imgIdx = irand(0, dataset.size());

        jp::img_bgr_t imgBGR;
        dataset.getBGR(imgIdx, imgBGR);

        jp::info_t info;
        dataset.getInfo(imgIdx, info);

        // predict object coordinates
        cv::Mat_<cv::Point2i> sampling = stochasticSubSample(imgBGR, objInputSize, rgbInputSize);
        std::vector<cv::Mat_<cv::Vec3f>> patches; // not used here
        jp::img_coord_t estObj = getCoordImg(imgBGR, sampling, rgbInputSize, patches, state);

        Hypothesis poseGT(info);

        // sample multiple random poses and calculate a reprojection error images for each one
        #pragma omp parallel for
        for(unsigned h = 0; h < hypsPerImage; h++)
        {
            int driftLevel = irand(0, 2); // decides whether pose with large error or pose with small error
            Hypothesis poseNoise;

            if(driftLevel == 0)
                poseNoise = poseGT * getRandHyp(2, 2); // small random pose error
            else
                poseNoise = poseGT * getRandHyp(10, 100); // large random pose error

            // calculate reporojection error for pose
            data[i * hypsPerImage + h] = getDiffMap(jp::our2cv(jp::jp_trans_t(poseNoise.getRotation(), poseNoise.getTranslation())), estObj, sampling, camMat);

            // check whether pose is above or below error tolerance
            if(poseGT.calcAngularDistance(poseNoise) < 5 && cv::norm(poseGT.getTranslation() - poseNoise.getTranslation()) < 50)
            {
                #pragma omp critical
                {
                    correct += 1.0;
                }
            }

            // calculate ground truth score value for this pose (-pose error * temperature)
            labels[i * hypsPerImage + h] = -temperature * std::max(poseGT.calcAngularDistance(poseNoise), cv::norm(poseGT.getTranslation() - poseNoise.getTranslation()) / 10.0);
        }

        std::cin.ignore();
    }
	
    std::cout << "Generated " << data.size() << " patches (" << correct / data.size() * 100 << "% correct) in " << stopW.stop() / 1000 << "s." << std::endl;
}

/**
 * @brief Extracts a subset of the given data into a batch of data.
 * @param offset Index offset wrt the complete data, where batch extraction should be started.
 * @param size Batch size.
 * @param permutation List of shuffled indices of the complete data. Batch is extracted from the shuffled data.
 * @param data List of reprojection error images.
 * @param labels List of associated ground truth score values.
 * @param batchData Output parameter. Batch of reprojection error images.
 * @param batchLabels Output parameter. Batch of associated ground truth scores.
 */
void assembleBatch(
    int offset, 
    int size, 
    const std::vector<int>& permutation, 
    const std::vector<cv::Mat_<float>>& data, 
    const std::vector<double>& labels, 
    std::vector<cv::Mat_<float>>& batchData, 
    std::vector<double>& batchLabels)
{
    batchData.resize(size);
    batchLabels.resize(size);
    
    for(unsigned i = 0; i < size; i++)
    {
        batchData[i] = data[permutation[i+offset]];
        batchLabels[i] = labels[permutation[i+offset]];
    }
}

int main(int argc, const char* argv[])
{
    int objBatchSize = 64; // batch size when training score CNN
    float objTemperature = 10; // scaling factor of ground truth scores, affects sharpness of score distribution later in the pipeline
    
    int trainingImages = 100; // number of training images used to extract reprojection error images in each training round
    int trainingHyps = 16; // number of reprojection error images per training image in each training round
    int trainingRounds = 80; // total number of training rounds
   
    int validationImages = 100; // number of validation images used to extract reprojection error images for the validation set
    int validationHyps = 64; // number of reprojection error images per validation image for the validation set
    int validationInterval = 10; // after how many training rounds is a validation pass performed?

    GlobalProperties* gp = GlobalProperties::getInstance();   
    gp->parseConfig();
    gp->parseCmdLine(argc, argv);
    
    std::string baseScriptRGB = gp->dP.objScript;
    std::string baseScriptObj = gp->dP.scoreScript;
    std::string modelFile = gp->dP.objModel;
    
    std::mt19937 randG;

    // load datasets
    std::string dataDir = "./";
    
    std::string trainingDir = dataDir + "training/"; 
    
    std::vector<std::string> trainingSets = getSubPaths(trainingDir);
    std::cout << std::endl << BLUETEXT("Loading training set ...") << std::endl;
    jp::Dataset trainingDataset = jp::Dataset(trainingSets[0], 1);
  
    #if DOVALIDATION
    std::string validationDir = dataDir + "validation/";

    std::vector<std::string> validationSets = getSubPaths(validationDir);
    std::cout << std::endl << BLUETEXT("Loading validation set ...") << std::endl;
    jp::Dataset valDataset = jp::Dataset(validationSets[0], 1);
    #endif
    
    // lua and models
    std::cout << "Loading script: " << baseScriptRGB << std::endl;    
    lua_State* stateRGB = luaL_newstate();
    luaL_openlibs(stateRGB);
    execute(baseScriptRGB.c_str(), stateRGB);    
    loadModel(modelFile, CNN_RGB_CHANNELS, CNN_RGB_PATCHSIZE, stateRGB);
    setEvaluate(stateRGB);
    
    std::cout << "Loading script: " << baseScriptObj << std::endl;    
    lua_State* stateObj = luaL_newstate();
    luaL_openlibs(stateObj);
    execute(baseScriptObj.c_str(), stateObj);
    constructModel(CNN_OBJ_CHANNELS, CNN_OBJ_PATCHSIZE, stateObj);
    setEvaluate(stateObj);
    
    // initialize data
    std::vector<cv::Mat_<float>> trainData;
    std::vector<cv::Mat_<float>> trainBatchData;
    std::vector<double> trainLabels;
    std::vector<double> trainBatchLabels;

    std::cout << GREENTEXT("Choosing validation data.") << std::endl;
    std::vector<cv::Mat_<float>> valData;
    std::vector<cv::Mat_<float>> valBatchData;
    std::vector<double> valLabels;
    std::vector<double> valBatchLabels; 

    #if DOVALIDATION
	assembleData(validationImages, validationHyps, CNN_OBJ_PATCHSIZE, CNN_RGB_PATCHSIZE, valDataset, stateRGB, valData, valLabels, objTemperature);
    #endif
    
    std::vector<int> valPermutation(valData.size());
    for(unsigned i = 0; i < valData.size(); i++)
        valPermutation[i] = i; // validation data not shuffled
    
    std::ofstream trainFile;
    trainFile.open("score_training_loss_"+baseScriptObj+".txt");

    #if DOVALIDATION
    std::ofstream valFile;
    valFile.open("score_validation_loss_"+baseScriptObj+".txt");          
    #endif
    
    int trainCounter = 0;
    
    for(int round = 0; round <= trainingRounds; round++)
    {
        #if DOVALIDATION
	    if(round % validationInterval == 0)
	    {
            std::cout << GREENTEXT("Validation pass.") << std::endl;
            setEvaluate(stateObj);
            double valLoss = 0;
            double correctPoseChosen = 0;

            for(int b = 0; b < valData.size() / objBatchSize; b++)
            {
                assembleBatch(b * objBatchSize, objBatchSize, valPermutation, valData, valLabels, valBatchData, valBatchLabels);
                std::vector<double> results = forward(valBatchData, stateObj); // forward pass of validation data
                valLoss += getLoss(results, valBatchLabels, stateObj); // calculate loss of predictions
                correctPoseChosen += selectBest(results, valBatchLabels, objTemperature); // calculate ratio of correctly estimated poses (inliers)
            }

            valLoss /= (valData.size() / objBatchSize);
            correctPoseChosen /= (valData.size() / objBatchSize);

            std::cout << GREENTEXT("Vaidation loss: " << valLoss << ", correct: " << correctPoseChosen * 100 << "%") << std::endl;
            valFile << trainCounter << " " << valLoss << " " << correctPoseChosen << std::endl;
	    }
        #endif

        std::cout << BLUETEXT("Starting training round " << round+1 << " of " << trainingRounds) << std::endl;
        setTraining(stateObj);
        assembleData(trainingImages, trainingHyps, CNN_OBJ_PATCHSIZE, CNN_RGB_PATCHSIZE, trainingDataset, stateRGB, trainData, trainLabels, objTemperature);

        // permutate data
        std::vector<int> trainPermutation(trainData.size());
        for(unsigned i = 0; i < trainData.size(); i++)
            trainPermutation[i] = i;
        std::shuffle(trainPermutation.begin(), trainPermutation.end(), randG);

        for(int b = 0; b < trainData.size() / objBatchSize; b++)
        {
            assembleBatch(b * objBatchSize, objBatchSize, trainPermutation, trainData, trainLabels, trainBatchData, trainBatchLabels);
            double trainLoss = train(trainBatchData, trainBatchLabels, stateObj); // forward-backward

            std::cout << YELLOWTEXT("Training loss: " << trainLoss) << std::endl;
            trainFile << trainCounter++ << " " << trainLoss << std::endl;
        }
    }
    
    trainFile.close();
    #if DOVALIDATION
    valFile.close();
    #endif

    lua_close(stateRGB);
    lua_close(stateObj);
    
    return 0;
}
