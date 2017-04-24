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
#include "cnn_softam.h"

#include <fstream>

int main(int argc, const char* argv[])
{
    GlobalProperties* gp = GlobalProperties::getInstance();
    gp->parseConfig();
    gp->parseCmdLine(argc, argv);

    int trainingRounds = 10000; // total number of parameter updates to perform (both CNNs)
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
    
    #if DOVALIDATION
    std::string validationDir = dataDir + "validation/";

    std::vector<std::string> validationSets = getSubPaths(validationDir);
    std::cout << std::endl << BLUETEXT("Loading validation set ...") << std::endl;
    jp::Dataset valDataset = jp::Dataset(validationSets[0], 1);
    
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
    
    std::cout << "Loading script: " << baseScriptRGB << std::endl;    
    lua_State* stateRGB = luaL_newstate();
    luaL_openlibs(stateRGB);
    execute(baseScriptRGB.c_str(), stateRGB);    
    loadModel(modelFileRGB, CNN_RGB_CHANNELS, CNN_RGB_PATCHSIZE, stateRGB);
       
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

            std::vector<double> losses;
            std::vector<double> sfEntropies;

            for(unsigned v = 0; v < valData.size(); v++)
            {
                std::cout << YELLOWTEXT("Processing validation image " << v << " of " << valData.size()) << "." << std::endl;

                // process frame (forward pass)
                std::vector<jp::cv_trans_t> hyps;
                jp::cv_trans_t refAvgHyp;
                jp::cv_trans_t avgHyp;
                std::vector<std::vector<cv::Point2f>> imgPts;
                std::vector<std::vector<cv::Point3f>> objPts;
                std::vector<std::vector<int>> imgIdx;
                std::vector<cv::Mat_<cv::Vec3f>> patches;
                std::vector<double> sfScores;
                jp::img_coord_t estObj;
                cv::Mat_<cv::Point2i> sampling;
                std::vector<std::vector<cv::Point2i>> sampledPoints;
                double loss;
                cv::Mat_<int> inlierMap;
                std::vector<std::vector<int>> pixelIdxs; //refIts -> pixelIdxs
                double tErr;
                double rotErr;

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
                    loss,
                    sfEntropy,
                    correct,
                    hyps,
                    refAvgHyp,
                    avgHyp,
                    imgPts,
                    objPts,
                    imgIdx,
                    patches,
                    sfScores,
                    estObj,
                    sampling,
                    sampledPoints,
                    inlierMap,
                    pixelIdxs,
                    tErr,
                    rotErr);

                avgCorrect += correct;

                //store statistics for calculation of mean and std dev
                losses.push_back(loss);
                sfEntropies.push_back(sfEntropy);
            }
	    
            // mean and stddev of loss of avg hypothesis
            std::vector<double> lossMean;
            std::vector<double> lossStdDev;
            cv::meanStdDev(losses, lossMean, lossStdDev);

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
        double sfEntropy;
        bool correct;

        std::vector<jp::cv_trans_t> hyps;
        jp::cv_trans_t refAvgHyp;
        jp::cv_trans_t avgHyp;
        std::vector<std::vector<cv::Point2f>> imgPts;
        std::vector<std::vector<cv::Point3f>> objPts;
        std::vector<std::vector<int>> imgIdx;
        std::vector<cv::Mat_<cv::Vec3f>> patches;
        std::vector<double> sfScores;
        jp::img_coord_t estObj;
        cv::Mat_<cv::Point2i> sampling;
        std::vector<std::vector<cv::Point2i>> sampledPoints;
        double loss;
        cv::Mat_<int> inlierMap;
        std::vector<std::vector<int>> pixelIdxs; //refIts -> pixelIdxs
        double tErr;
        double rotErr;

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
            loss,
            sfEntropy,
            correct,
            hyps,
            refAvgHyp,
            avgHyp,
            imgPts,
            objPts,
            imgIdx,
            patches,
            sfScores,
            estObj,
            sampling,
            sampledPoints,
            inlierMap,
            pixelIdxs,
            tErr,
            rotErr);

        // === doing the backward pass ====================================================================
        // caluclating gradients according to chainrule backwards through the pipeline (starting with the loss)
        // there are two paths of influence of the object coordinates wrt the end loss:
        //   path I: direct influence of object coordinates on the selected hypothesis
        //   path II: indirect influence of the object coordinates on the hypothesis selection via hypotheses scores
        StopWatch stopW;
        cv::Mat_<double> dLoss_dObj = cv::Mat_<double>::zeros(patches.size(), 3);
        cv::Mat_<double> dLoss_dObj_1row = cv::Mat_<double>::zeros(1, patches.size() * 3);

        // --- path I, hypothesis path --------------------------------------------------------------------
        std::cout << BLUETEXT("Calculating gradients of refinement wrt object coordintes.") << std::endl;

        // derivative of loss wrt refined avg hypothesis
        jp::jp_trans_t refAvgHypJP = jp::cv2our(refAvgHyp);
        cv::Mat_<double> dLoss_dRAvgHyp = dLossMax(
            Hypothesis(refAvgHypJP.first, refAvgHypJP.second).getRodVecAndTrans(),
            poseGT.getRodVecAndTrans());

        // refinement is a function of object coordinates and of the avg hypothesis
        // FIRST) calculate gradients of refined avg hypothesis wrt object coordinates
        cv::Mat_<double> dRAvgHyp_dObj = dRefineObj(
            refInlierCount,
            refSteps,
            refSubSample,
            inlierThreshold2D,
            pixelIdxs,
            estObj,
            sampling,
            camMat,
            avgHyp,
            inlierMap
        );

        dLoss_dObj_1row += dLoss_dRAvgHyp * dRAvgHyp_dObj;

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        std::cout << BLUETEXT("Calculating gradients of refinement wrt sum of poses.") << std::endl;

        // SECOND) calculate gradients of refined avg hypothesis wrt avg hypothesis
        cv::Mat_<double> dRAvgHyp_dHyp = dRefineHyp(
                refInlierCount,
                refSteps,
                inlierThreshold2D,
                pixelIdxs,
                estObj,
                sampling,
                camMat,
                avgHyp
            );

        // combine all gradients of path I
        cv::Mat_<double> dHyp_dObj_complete = cv::Mat_<double>::zeros(6, patches.size() * 3);

        // SECOND) + gradients of hypotheses wrt object coordinates
        //  -> gives indirect influence of object coordinates via hypotheses that get averaged
        for(unsigned h = 0; h < hyps.size(); h++)
        {
            cv::Mat_<double> dHyp_dObj = sfScores[h] * dPNP(imgPts[h], objPts[h]); // gradients of hypotheses wrt object coordinates
            for(unsigned i = 0; i < imgIdx[h].size(); i++)
                dHyp_dObj_complete.colRange(imgIdx[h][i] * 3, imgIdx[h][i] * 3 + 3) += dHyp_dObj.colRange(i * 3, i * 3 + 3);
        }

        // add FIRST)
        //   -> gives direct influence of object coordinates on refinement
        dLoss_dObj_1row += dLoss_dRAvgHyp * dRAvgHyp_dHyp * dHyp_dObj_complete;

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        // --- path II, influence of object coordinates and score CNN on averaging weights -------
        std::cout << BLUETEXT("Calculating gradients of refinement wrt scores.") << std::endl;

        // calculate gradients at the output of the Score CNN (per hypothesis)
        std::vector<double> scoreOutputGradients(hyps.size(), 0);

        for(unsigned h = 0; h < hyps.size(); h++)
        {
            cv::Mat_<double> hypMat(6, 1);
            hyps[h].first.copyTo(hypMat.rowRange(0, 3));
            hyps[h].second.copyTo(hypMat.rowRange(3, 6));
            hypMat.rowRange(3, 6) /= 1000;

            hypMat = dLoss_dRAvgHyp * dRAvgHyp_dHyp * hypMat;
            double hypFactor = hypMat.at<double>(0, 0);

            scoreOutputGradients[h] += sfScores[h] * hypFactor;
            for(unsigned j = 0; j < hyps.size(); j++)
                scoreOutputGradients[j] -= sfScores[h] * sfScores[j] * hypFactor;
        }

        // backward pass through the Score CNN to get gradients of object coordinates wrt scores
        std::vector<cv::Mat_<double>> dLoss_dScores;
        dScore(estObj, sampling, sampledPoints, stateObj, dLoss_dScores, scoreOutputGradients);

        for(unsigned h = 0; h < hyps.size(); h++)
            dLoss_dObj_1row += dLoss_dScores[h];


        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        // reformat gradient matrix
        for(unsigned idx = 0; idx < CNN_OBJ_PATCHSIZE * CNN_OBJ_PATCHSIZE; idx++)
        {
            dLoss_dObj(idx, 0) = dLoss_dObj_1row(idx * 3 + 0);
            dLoss_dObj(idx, 1) = dLoss_dObj_1row(idx * 3 + 1);
            dLoss_dObj(idx, 2) = dLoss_dObj_1row(idx * 3 + 2);
        }

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
        backward(loss, patches, dLoss_dObj, stateRGB);

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        trainFile
            << round << " "     // 0 - training round (or number of parameter updates)
            << loss << " "      // 1 - loss of the average hypothesis in this training round
            << sfEntropy        // 2 - entropy of the score distribution (averaging weights)
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
