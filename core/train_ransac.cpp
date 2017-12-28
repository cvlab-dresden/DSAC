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

#define DOVALIDATION 0  // compile flag, if true a validation set has to be available

#include "properties.h"
#include "thread_rand.h"
#include "util.h"
#include "stop_watch.h"
#include "dataset.h"

#include "lua_calls.h"
#include "cnn.h"

#include <fstream>

int main(int argc, const char* argv[])
{
    // load parameters
    GlobalProperties* gp = GlobalProperties::getInstance();
    gp->parseConfig();
    gp->parseCmdLine(argc, argv);

    int trainingRounds = 5000; // total number of parameter updates to perform (both CNNs)
    int valInterval = 1000; // perform a validation pass after this many parameter updates
    int valImages = 1000; // number of validation images (if validation set is smaller, images might be used multiple times)
    
    int refInlierCount = gp->pP.ransacBatchSize;
    int refSteps = gp->pP.ransacRefinementIterations;
    float refSubSample = gp->pP.ransacSubSample;
    
    int objHyps = gp->pP.ransacIterations;

    int ptCount = 4;
    int inlierThreshold2D = gp->pP.ransacInlierThreshold2D;
  
    std::string baseScriptRGB = gp->dP.objScript;
    std::string baseScriptObj = gp->dP.scoreScript;
    std::string modelFileRGB = gp->dP.objModel;
    std::string modelFileObj = gp->dP.scoreModel;
   
    // load datasets
    std::string dataDir = "./";
    
    std::string trainingDir = dataDir + "training/"; 
    
    std::vector<std::string> trainingSets = getSubPaths(trainingDir);
    std::cout << std::endl << BLUETEXT("Loading training set ...") << std::endl;
    jp::Dataset trainingDataset = jp::Dataset(trainingSets[0], 1);


    //check if the traning set is empty
    if (!trainingDataset.size())  
    {
        std::cout << std::endl << REDTEXT("The training set is empty") << std::endl;
        return 0;
    }
    
    #if DOVALIDATION
    std::string validationDir = dataDir + "validation/";

    std::vector<std::string> validationSets = getSubPaths(validationDir);
    std::cout << std::endl << BLUETEXT("Loading validation set ...") << std::endl;
    jp::Dataset valDataset = jp::Dataset(validationSets[0], 1);
    
    // pre-load validation set (images chosen randomly)
    std::cout << YELLOWTEXT("Loading " << valImages << " validation images.") << std::endl;
    std::vector<jp::img_bgr_t> valData(valImages);
    std::vector<Hypothesis> valGT(valImages);

	for(unsigned v = 0; v < valImages; v++)
	{
	    int imgID = irand(0, valDataset.size());
	  
        valDataset.getBGR(imgID, valData[v]);
	    
	    jp::info_t info;
        valDataset.getInfo(imgID, info);
	    valGT[v] = Hypothesis(info);      
    }
    #endif
    
    // lua and models
    std::cout << "Loading script: " << baseScriptObj << std::endl;    
    lua_State* stateObj = luaL_newstate();
    luaL_openlibs(stateObj);
    execute(baseScriptObj.c_str(), stateObj);
    loadModel(modelFileObj, CNN_OBJ_CHANNELS, CNN_OBJ_PATCHSIZE, stateObj);    
    setEvaluate(stateObj);
    
    std::cout << "Loading script: " << baseScriptRGB << std::endl;    
    lua_State* stateRGB = luaL_newstate();
    luaL_openlibs(stateRGB);
    execute(baseScriptRGB.c_str(), stateRGB);    
    loadModel(modelFileRGB, CNN_RGB_CHANNELS, CNN_RGB_PATCHSIZE, stateRGB);
    setEvaluate(stateRGB);
       
    cv::Mat camMat = gp->getCamMat();
    
    std::ofstream trainFile;
    trainFile.open("ransac_training_loss_"+gp->dP.objScript+".txt"); // contains statistics of the training process

    #if DOVALIDATION
    std::ofstream valFile;
    valFile.open("ransac_validation_loss_"+gp->dP.objScript+".txt"); // contains statistics of validation passes
    #endif
    
    for(unsigned round = 0; round <= trainingRounds; round++)
    {
        #if DOVALIDATION
        if(round % valInterval == 0)
        {
            std::cout << YELLOWTEXT("Validation round.") << std::endl;
            setEvaluate(stateRGB);
            setEvaluate(stateObj);

            double avgCorrect = 0;

            std::vector<double> expLosses;
            std::vector<double> sfEntropies;

            for(unsigned v = 0; v < valData.size(); v++)
            {
                std::cout << YELLOWTEXT("Processing validation image " << v << " of " << valData.size()) << "." << std::endl;

                // process frame (forward pass)
                std::vector<jp::cv_trans_t> hyps;
                std::vector<jp::cv_trans_t> refHyps;
                std::vector<std::vector<cv::Point2f>> imgPts;
                std::vector<std::vector<cv::Point3f>> objPts;
                std::vector<std::vector<int>> imgIdx;
                std::vector<cv::Mat_<cv::Vec3f>> patches;
                std::vector<double> sfScores;
                jp::img_coord_t estObj;
                cv::Mat_<cv::Point2i> sampling;
                std::vector<std::vector<cv::Point2i>> sampledPoints;
                std::vector<double> losses;
                std::vector<cv::Mat_<int>> inlierMaps;
                std::vector<std::vector<std::vector<int>>> pixelIdxs; //hyps -> refIts -> pixelIdxs
                double tErr;
                double rotErr;
                int hypIdx;

                double expectedLoss;
                double sfEntropy;
                bool correct;

                processImage(
                    valData[v],
                    valGT[v],
                    stateRGB,
                    stateObj,
                    objHyps,
                    ptCount,
                    camMat,
                    inlierThreshold2D,
                    refInlierCount,
                    refSteps,
                    expectedLoss,
                    sfEntropy,
                    correct,
                    hyps,
                    refHyps,
                    imgPts,
                    objPts,
                    imgIdx,
                    patches,
                    sfScores,
                    estObj,
                    sampling,
                    sampledPoints,
                    losses,
                    inlierMaps,
                    pixelIdxs,
                    tErr,
                    rotErr,
                    hypIdx);

                avgCorrect += correct;

                //store statistics for calculation of mean and std dev
                expLosses.push_back(expectedLoss);
                sfEntropies.push_back(sfEntropy);
            }

            // mean and stddev of loss of selected hypotheses
            std::vector<double> lossMean;
            std::vector<double> lossStdDev;
            cv::meanStdDev(expLosses, lossMean, lossStdDev);

            // mean and stddev of entropy of hypotheses score distribution
            std::vector<double> entropyMean;
            std::vector<double> entropyStdDev;
            cv::meanStdDev(sfEntropies, entropyMean, entropyStdDev);

            avgCorrect /= valData.size();

            std::cout << YELLOWTEXT("Avg. validation loss: " << lossMean[0] << ", accuracy: " << avgCorrect * 100 << "%") << std::endl;
            valFile
                << round << " "             // 0 - number of parameter updates
                << avgCorrect << " "        // 1 - percentage of correct poses
                << lossMean[0] << " "       // 2 - mean loss of selected hypotheses
                << lossStdDev[0] << " "     // 3 - standard deviation of losses of selected hypotheses
                << entropyMean[0] << " "    // 4 - mean of the score distribution entropy
                << entropyStdDev[0]         // 5 - standard deviation of the score distribution entropy
                << std::endl;
        }
        #endif

        std::cout << YELLOWTEXT("Round " << round << " of " << trainingRounds << ".") << std::endl;
        setTraining(stateRGB);
        setTraining(stateObj);

        // load random training frame
        int imgID = irand(0, trainingDataset.size());

        jp::img_bgr_t imgBGR;
        trainingDataset.getBGR(imgID, imgBGR);

        jp::info_t info;
        trainingDataset.getInfo(imgID, info);
        Hypothesis poseGT(info);

        // forward pass (also calculates many things needed for backward pass), see method documentation for parameter explanation
        double expectedLoss;
        double sfEntropy;
        bool correct;

        std::vector<jp::cv_trans_t> hyps;
        std::vector<jp::cv_trans_t> refHyps;
        std::vector<std::vector<cv::Point2f>> imgPts;
        std::vector<std::vector<cv::Point3f>> objPts;
        std::vector<std::vector<int>> imgIdx;
        std::vector<cv::Mat_<cv::Vec3f>> patches;
        std::vector<double> sfScores;
        jp::img_coord_t estObj;
        cv::Mat_<cv::Point2i> sampling;
        std::vector<std::vector<cv::Point2i>> sampledPoints;
        std::vector<double> losses;
        std::vector<cv::Mat_<int>> inlierMaps;
        std::vector<std::vector<std::vector<int>>> pixelIdxs; //hyps -> refIts -> pixelIdxs
        double tErr;
        double rotErr;
        int hypIdx;

        processImage(
            imgBGR,
            poseGT,
            stateRGB,
            stateObj,
            objHyps,
            ptCount,
            camMat,
            inlierThreshold2D,
            refInlierCount,
            refSteps,
            expectedLoss,
            sfEntropy,
            correct,
            hyps,
            refHyps,
            imgPts,
            objPts,
            imgIdx,
            patches,
            sfScores,
            estObj,
            sampling,
            sampledPoints,
            losses,
            inlierMaps,
            pixelIdxs,
            tErr,
            rotErr,
            hypIdx);

        // === doing the backward pass ====================================================================
        // caluclating gradients according to chainrule backwards through the pipeline (starting with the loss)
        // there are two paths of influence of the object coordinates wrt the end loss:
        //   path I: direct influence of object coordinates on the selected hypothesis
        //   path II: indirect influence of the object coordinates on the hypothesis selection via hypotheses scores
        StopWatch stopW;

        // --- path I, direct influence of object coordinate on loss --------------------------------------
        std::cout << BLUETEXT("Calculating gradients wrt hypotheses.") << std::endl;

        // precalculate gradients per of (refined) hypothesis wrt object coordinates
        std::vector<cv::Mat_<double>> dHyp_dObjs(refHyps.size());

        #pragma omp parallel for
        for(unsigned h = 0; h < refHyps.size(); h++)
        {
            if(sfScores[h] > 0.0001) // skip hypothesis with very low probability to save training time
            {
                // derivative of the refinement process
                dHyp_dObjs[h] = dRefine(
                    refInlierCount,
                    refSteps,
                    refSubSample,
                    inlierThreshold2D,
                    pixelIdxs[h],
                    estObj,
                    sampling,
                    camMat,
                    imgPts[h],
                    objPts[h],
                    sampledPoints[h],
                    inlierMaps[h]
                );
            }
            else
            {
                dHyp_dObjs[h] = cv::Mat_<double>::zeros(6, CNN_OBJ_PATCHSIZE * CNN_OBJ_PATCHSIZE * 3);
            }
        }

        // combine gradients (dLoss * dHyp) per hypothesis
        std::vector<cv::Mat_<double>> gradients(refHyps.size());

        #pragma omp parallel for
        for(unsigned h = 0; h < refHyps.size(); h++)
        {
            jp::jp_trans_t jpTrans = jp::cv2our(refHyps[h]);
            cv::Mat_<double> dLoss_dHyp = dLossMax(Hypothesis(jpTrans.first, jpTrans.second).getRodVecAndTrans(), poseGT.getRodVecAndTrans());
            gradients[h] = dLoss_dHyp * dHyp_dObjs[h];
        }

        // acumulate hypotheses gradients for patches (expectation)
        cv::Mat_<double> dLoss_dObj = cv::Mat_<double>::zeros(patches.size(), 3);

        for(unsigned h = 0; h < refHyps.size(); h++)
        for(unsigned idx = 0; idx < CNN_OBJ_PATCHSIZE * CNN_OBJ_PATCHSIZE; idx++)
        {
            dLoss_dObj(idx, 0) += sfScores[h] * gradients[h](idx * 3 + 0);
            dLoss_dObj(idx, 1) += sfScores[h] * gradients[h](idx * 3 + 1);
            dLoss_dObj(idx, 2) += sfScores[h] * gradients[h](idx * 3 + 2);
        }

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        // --- path II, influence of object coordinates and score CNN on selection probability -------
        std::cout << BLUETEXT("Calculating gradients wrt scores.") << std::endl;

        // calculate derivative of score (sampling prob.) wrt object coordinates, invoces backward pass on score CNN
        std::vector<cv::Mat_<double>> dLoss_dScore_dObjs = dSMScore(estObj, sampling, sampledPoints, losses, sfScores, stateObj);

        // accumulate gradients wrt object coordinate via the hypothesis score (sampling probability)
        cv::Mat_<double> dLoss_dScore_dObj = cv::Mat_<double>::zeros(patches.size(), 3);

        for(unsigned h = 0; h < hyps.size(); h++)
            dLoss_dScore_dObj += dLoss_dScore_dObjs[h];

        dLoss_dObj += dLoss_dScore_dObj;

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        // gradient statistics
        std::cout << BLUETEXT("Combined statistics:") << std::endl;
        int zeroGrads = 0;
        for(int row = 0; row < dLoss_dObj.rows; row++)
        {
            if(cv::norm(dLoss_dObj.row(row)) < EPS)
            zeroGrads++;
        }

        std::cout << "Max gradient: " << getMax(dLoss_dObj) << std::endl;
        std::cout << "Avg gradient: " << getAvg(dLoss_dObj) << std::endl;
        std::cout << "Med gradient: " << getMed(dLoss_dObj) << std::endl;
        std::cout << "Zero gradients: " << zeroGrads << std::endl;

        // finally, backward pass on object coordinate CNN
        std::cout << BLUETEXT("Update object coordinate CNN.") << std::endl;
        backward(expectedLoss, patches, dLoss_dObj, stateRGB);

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        trainFile
            << round << " "         // 0 - training round (or number of parameter updates)
            << expectedLoss << " "  // 1 - expected loss of this training round
            << sfEntropy            // 2 - entropy of the score distribution
            << std::endl;
        std::cout << std::endl;
    }
    
    trainFile.close();
    #if DOVALIDATION
    valFile.close();
    #endif
    
    lua_close(stateRGB);
    lua_close(stateObj);
    
    return 0;    
}
