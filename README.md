# DSAC
Code associated with the paper:

E. Brachmann, A. Krull, S. Nowozin, J. Shotton, F. Michel, S. Gumhold, C. Rother,  
"[DSAC – Differentiable RANSAC for Camera Localization](https://arxiv.org/abs/1611.05705)",  
CVPR 2017

Please see the documentation.pdf for additional information. Also see the [project page](https://hci.iwr.uni-heidelberg.de/vislearn/research/scene-understanding/pose-estimation/#DSAC). You can download pre-trained models for 7Scenes [here](https://heidata.uni-heidelberg.de/api/access/datafile/:persistentId?persistentId=doi:10.11588/data/3JVZSH/TSMZA8)

**Note:** Beginning of August 2017, we updated the public version of the code to contain a fix in the pose evaluation metric, and to utilize a more stable variant of the PnP algorithm. These changes result in improved numbers compared to the original version of the paper. The improved numbers and a more detailed explanation can be found in the current version of the paper on arXiv. The current version of the code contains both fixes, and also the pre-trained models have been updated, accordingly.

**Update:** An improved version of this system is available [here](https://github.com/vislearn/LessMore). Higher accuracy, stable end-to-end training, training without a 3D model or depth maps. 
