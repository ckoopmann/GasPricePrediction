# Python/BinaryPrediction
This directory  contains the code for training, tuning and testing the models in the binary prediction problem, and is structured into the following files:
* **Python/BinaryPrediction/aws_commands.sh** contains a shell script that was used (command: **nohup bash aws_commands.sh** ) to run these files in the correct order on an AWS Linux Machine learning instance and automatically commit and push the results to github
* **Python/BinaryPrediction/binary_eval_cv.py** contains the code to evaluate the uni and multivariate models tuned in the previous steps using cross validation
* **Python/BinaryPrediction/binary_multivar_par_tuning_nocv.py** contains the code to tune the parameters of the  multivariate models based on the previously executed variable selection
* **Python/BinaryPrediction/binary_par_tuning_nocv.py** contains the code to tune the parameters of the  univariate models as the first step in this process
* **Python/BinaryPrediction/binary_var_selection_nocv.py** contains the select the variables based on the parameters tuned in the first step
* **Python/BinaryPrediction/functions.py** contains functions used in the other scripts
* **Python/BinaryPrediction/nohup.out** contains the output printed out to the console when running aws_commands.sh
* **Python/BinaryPrediction/git_setup.txt** contains a git command to be executed manually on the aws instance before cloning / pulling the repository to enable automatic pushing of results when running aws_commands.sh.
The correct execution of all of these files requires the Python working directory to be set to this directory.

