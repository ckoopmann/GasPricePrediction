source activate tensorflow_p36
yes | conda install scikit-learn
python level_par_tuning_nocv.py
git pull
git add --all
git commit -m "New AWS results - Level Par Tuning"
git push
python level_var_selection_nocv.py
git pull
git add --all
git commit -m "New AWS results - Level Var Selection"
git push
python level_multivar_par_tuning_nocv.py
git pull
git add --all
git commit -m "New AWS results - Level Multivar Par Tuning"
git push
python level_eval_cv.py
git pull
git add --all
git commit -m "New AWS results - Level Evaluation"
git push
sudo shutdown -h now