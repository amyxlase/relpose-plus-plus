## Evaluation Directions


### Order Paths

We pre-computed random orderings of image frames for each sequence. To replicate the
exact same orderings that we used, download the jsons with the frame indices:
```
gdown https://drive.google.com/uc?id=1oGpv2R-n8LAw6SL5-jycTTniLwcQNvId
unzip co3d_order_paths.zip -d data
```

Note that there are 5 randomly generated orderings for each sequence. We averaged all
evaluations over 5 random runs to get the final results.

To compute results for sequences of length N, we use the first N frames of each
ordering.

### Running evaluations

Evaluations can be run with [`eval_driver.py`](relpose/eval/eval_driver.py) for any set
of configurations, eg:
```
python relpose/eval_driver.py --checkpoint_path weights/relposepp --mode pairwise \
    --num_frames 2 --category apple --sample_num 0
```

Description of arguments:
* `checkpoint_path`: path to the model checkpoint
* `mode`: `pairwise` for pairwise rotation evaluation, `coordinate_ascent` for joint
    rotation accuracy using coordinate ascent, `cc` for camera center accuracy, and `t`
    for camera translation accuracy. Note that camera center and translation accuracy
    must be run after `coordinate_ascent` has finished
* `num_frames`: number of image frames to use for evaluation
* `category`: CO3D category to evaluate on
* `sample_num`: index of the random ordering to use for evaluation. We averaged over
    5 random orderings for each sequence to get the final results

Because evaluation requires running a large number of jobs, we provide an option to
generate all the commands necesary to improve parallelization:
```
python relpose/eval_driver.py --list_jobs --output_text_path jobs.sh
```