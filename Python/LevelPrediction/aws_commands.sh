sudo pip3 install keras==2.0.8
sudo pip3 install h5py
python3 level_multivar_par_tuning_nocv.py
git pull
git add --all
git commit -m "New AWS results - Level Multivar Par Tuning"
git push
python3 level_eval_cv.py
git pull
git add --all
git commit -m "New AWS results - Level Evaluation"
git push
sudo shutdown -h now