sudo pip3 install keras==2.0.8
sudo pip3 install h5py
python3 binary_par_tuning_nocv.py
git pull
git add --all
git commit -m "New AWS results - Binary Par Tuning"
git push
python3 binary_var_selection_nocv.py
git pull
git add --all
git commit -m "New AWS results - Binary Var Selection"
git push
python3 binary_multivar_par_tuning_nocv.py
git pull
git add --all
git commit -m "New AWS results - Binary Multivar Par Tuning"
git push
python3 binary_eval_cv.py
git pull
git add --all
git commit -m "New AWS results - Binary Evaluation"
git push
sudo shutdown -h now