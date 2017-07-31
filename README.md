
In general I worked on two different topics to which you can group most of the scripts.
1) multi frame experiments with eos and analysis of the 3d error
2) Face recognition experiments with CNNs and merging of isomaps for face recognition

I try two keep most of the functions and classes as high level as possible and not data specific. These py scripts are often called *_lib.py (IJB_A_template_lib, cnn_db_loader, eos_starter_lib, obj_analysis_lib) They are then imported by other scripts that are dataset specific. 

Scripts are often called similarly if they have similar tasks. I want to describe high level functions for some of the groups:

IJB_A_*
	For this dataset specific tools for the face recognition experiments we run on it. Often use IJB_A_template_lib.

analyse_* 
	To analyse 3D fitting error on different datasets and different fitting methods. Often use ojb_analysis_lib

cnn_* 
	All the scripts used for the cnn training, evaluation and testing. 
	cnn_db_loader is used for loading pasc and casia datasets, 
	cnn_tf_graphs has all the tensorflow graphs (CNN definitions) 
	tf_utils provides some helper functions for tensorflow

plot_* 
	All scripts that plot something. I use matplotlib for plotting in python. With texfig.py (stolen somewhere, not my own) you can export the graph as pdf and vector graphic.

run_* 
	Scripts that run the fitting on different datasets and different params. Probably all use eos_starter_lib to start the fitting in parallel.


Workflow for a typical fitting 3d error experiment:
1) run the fitting with run_* (eg. run_multiframe_fit_KF-ITW)
2) compare fitting with GT, do all the alignment stuff (procrusts, nr-icp, ...) with analyse_* (eg analyse_KF-ITW_multi_fittings) This will write out some intermediate steps and finally a distances_v?.log
3) This distances file gets read by plot_mesh_distances and gets ploted

Workflow for a typical FR experiment
1) run the fitting with run_* (eg. run_multiframe_fit_CASIA) for all images in training, eval and test set
2) Start a CNN training with cnn_experiment_train. That script will use cnn_db_loader to gather all images (will write to cnn_exp_path/db_input/), cnn_tf_graph to construct the tensorflow graph, tf_utils for loading the images
3) every now and then or when you are sure that the training is over you run cnn_experiment_eval. This will take the eval set (specified in step before) and evaluate performance in top1 accuracy. It will be written to cnn_exp_path/eval/.
4) you can plot evaluation accuracy with plot_cnn_eval
5) To run the network on the test set use cnn_experiment_test. It will write the feature vectors to a file in the cnn_exp_path
6) To measure face recognition performance I use IJB-A verification. In IJB-A you match between templates that can consist of several images. So you have to merge several images together either on image level (isomap merging) or on feature level. IJB_A_generate_feature_vector will load the IJB-A template definition files, merge the feature_vectors (averaging) and write for each split template specific feature vectors to a verification experiment folder. 
7) We can now finally match the feature vectors of the templates against each other. This is done with IJB_A_verification_mathing. This script will write a simmatrix file to the experiment folder for each split.
8) To plot the roc performance use plot_roc. This will take the average and plot the distribution over all splits as errorbars.


