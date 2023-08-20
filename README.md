# PAcT

This repository contains the accompanying code for the paper "_PAcT: Detecting and Classifying Privacy Behaviors of Android Applications_". 

## Install

This repository requires conda. Install the required packages using the script in `install_software.sh`. This will create a conda environment called `exp` and then install the necessary requirements. 

## Files

The `dataset` folder contains all the dataset files. Each dataset for RQ1 (see paper for details) is named in the format of `[practice|purpose]_<dataset-name>.txt`. For example, `practice_collecting.txt` and `purpose_functionality.txt`.  Similarly, dataset files for RQ2 are named in the format of `[practice/purpose]_<N>_hop_<dataset-name>.txt`. For example `practice_1_hop_collecting.txt` and `purpose_1_hop_functionality.txt`. The `models` folder in `dataset` contains the embedding model.

The `code` folder contains the code to train the models. Each python file trains the model with each individual dataset and are named similar to the dataset files in the format of `final_exp_[practice|purpose]_<dataset-name>.py`. For example, `final_exp_practice_1_hop_functionality.py`. All the trained weights for each experiment are saved in the `code/experiment_results/<experiment-name>` folder. For example `code/experiment_results/Practice_Collection/_model.pt`.

The `.config` folder contains the configuration for each python file and can be tweaked.  

## Running Experiments

To run all experiments for RQ1 and RQ2, use the scripts `run_experiment_rq_1.sh` and `run_experiment_rq_2.sh`. To save confusion matrices, change `EVAL=0` to `EVAL=1` in the scripts. Each experiment run will create a new folder in the `experiment_results` according to the name of the experiment and save the logs and tensorboard event files there. Each run will overwrite the results from previous experiments. 

*Note* - To reproduce the results from the paper, change `EXPERIMENT=1` to `EXPERIMENT=0` and `EVAL=0` to `EVAL=1`. This will load the saved models in `experiment_results` directory for each experiment and save the confusion matrix from the results. Running each experiment 

To change hyperparameters, such as number of epochs or learning rate, you can do so in `.toml` files in the `.config` folder. 

## Questions

If you have any trouble running the experiment, please email the question to - `vijayanta.jain@maine.edu`

## Cite

If you use this code academically, please cite our work. You can read the complete work [here](https://dl.acm.org/doi/abs/10.1145/3507657.3528543)

```
@inproceedings{jain2022pact,
  title={PAcT: Detecting and Classifying Privacy Behavior of Android Applications},
  author={Jain, Vijayanta and Gupta, Sanonda Datta and Ghanavati, Sepideh and Peddinti, Sai Teja and McMillan, Collin},
  booktitle={Proceedings of the 15th ACM Conference on Security and Privacy in Wireless and Mobile Networks},
  pages={104--118},
  year={2022}
}
```
