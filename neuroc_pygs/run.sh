echo "prefetch_generator_train_step"
python -u prefetch_generator_train_step.py >prefetch_generator_train_step_2_27.log 2>&1

echo "preloader_train_step"
python -u preloader_train_step.py >preloader_train_step_2_27.log 2>&1

echo "data_prefetcher_train_step"
python -u data_prefetcher_train_step.py >data_prefetcher_train_step_2_27.log 2>&1

echo "end"