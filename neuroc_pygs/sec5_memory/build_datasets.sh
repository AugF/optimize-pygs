echo "gcn"
python -u sec5_2_build_datasets.py >>sec5_2_memory_log/sec5_2_build_datasets.log 2>&1

echo "ggnn"
python -u sec5_2_build_datasets.py --model ggnn >>sec5_2_memory_log/sec5_2_build_datasets.log 2>&1

echo "gaan"
python -u sec5_2_build_datasets.py --model gaan >>sec5_2_memory_log/sec5_2_build_datasets.log 2>&1


