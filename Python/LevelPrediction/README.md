# Python/LevelPrediction
This directory  contains the code for training, tuning and testing the models in the level prediction problem, and is structured into the following files:
* **Python/LevelPrediction/aws_commands.sh** contains a shell script that was used (command: **nohup bash aws_commands.sh** ) to run these files in the correct order on an AWS Linux Machine learning instance and automatically commit and push the results to github
* **Python/LevelPrediction/level_eval_cv.py** contains the code to evaluate the uni and multivariate models tuned in the previous steps using cross validation
* **Python/LevelPrediction/level_multivar_par_tuning_nocv.py** contains the code to tune the parameters of the  multivariate models based on the previously executed variable selection
* **Python/LevelPrediction/level_par_tuning_nocv.py** contains the code to tune the parameters of the  univariate models as the first step in this process
* **Python/LevelPrediction/level_var_selection_nocv.py** contains the select the variables based on the parameters tuned in the first step
* **Python/LevelPrediction/functions.py** contains functions used in the other scripts
* **Python/LevelPrediction/nohup.out** contains the output printed out to the console when running aws_commands.sh
* **Python/LevelPrediction/git_setup.txt** contains a git command to be executed manually on the aws instance before cloning / pulling the repository to enable automatic pushing of results when running aws_commands.sh.
The correct execution of all of these files requires the Python working directory to be set to this directory.
