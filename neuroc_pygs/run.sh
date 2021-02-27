echo "prefetch_generator_train_step"
python -u prefetch_generator_train_step.py --num_workers 0 --log_batch True >prefetch_generator_train_step.log 2>&1
python -u prefetch_generator_train_step.py --num_workers 0 >prefetch_generator_train_step_0.log 2>&1

echo "preloader_train_step"
python -u preloader_train_step.py --num_workers 0 --log_batch True >preloader_train_step.log 2>&1
python -u preloader_train_step.py --num_workers 0 >preloader_train_step_0.log 2>&1

echo "data_prefetcher_train_step"
python -u data_prefetcher_train_step.py --num_workers 0 --log_batch True >data_prefetcher_train_step.log 2>&1
python -u data_prefetcher_train_step.py --num_workers 0 >data_prefetcher_train_step_0.log 2>&1

echo "end"