source activate tensorflow_p36
conda install scikit-learn
python binary_par_tuning_nocv.py
git pull
git add --all
git commit -m "New AWS results - Binary Par Tuning"
git push
python binary_var_selection_nocv.py
git pull
git add --all
git commit -m "New AWS results - Binary Var Selection"
git push
python binary_multivar_par_tuning_nocv.py
git pull
git add --all
git commit -m "New AWS results - Binary Multivar Par Tuning"
git push
python binary_eval_cv.py
git pull
git add --all
git commit -m "New AWS results - Binary Evaluation"
git push
sudo shutdown -h now