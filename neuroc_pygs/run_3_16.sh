date
echo 'begin epochs'
python -u sec4_time/epochs.py >sec4_time/exp_log/epochs.log 2>&1
date
echo 'begin automl datasets'
nohup python -u sec5_memory/motivation_automl_datasets.py --model gcn --device cuda:1 >sec5_memory/exp_log/motivation_automl_datasets_gcn.log 2>&1 &
nohup python -u sec5_memory/motivation_automl_datasets.py --model gat --device cuda:2 >sec5_memory/exp_log/motivation_automl_datasets_gat.log 2>&1 &
date
echo "end"
