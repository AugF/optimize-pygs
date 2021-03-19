date
echo 'reddit_sage'
python reddit_sage_predict.py >exp_diff_log/reddit_sage_predict_8900_v0.log 2>&1
echo 'cluster_gcn'
date
python cluster_gcn_predict.py >exp_diff_log/cluster_gcn_predict_v0.log 2>&1
echo 'end'
date