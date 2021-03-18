date
echo 'begin sec5'
python sec5_memory/motivation_optimize_predict.py >sec5_memory/exp_log/motivation_optimize_predict_v2_1.log 2>&1
date
echo 'begin sec6'
python sec6_cutting/reddit_sage.py >sec6_cutting/exp_log/reddit_sage_v2.log 2>&1
date
python sec6_cutting/cluster_gcn.py >sec6_cutting/exp_log/cluster_gcn_v2.log 2>&1
date
python sec6_cutting/linear_model.py >sec6_cutting/exp_log/linear_model_v2.log 2>&1
date
python sec6_cutting/opt_reddit_sage.py >sec6_cutting/exp_log/opt_reddit_sage_v2.log 2>&1
date
python sec6_cutting/opt_cluster_gcn.py >sec6_cutting/exp_log/opt_cluster_gcn_v2.log 2>&1