
## Evaluate the offline saved results
Our code support evaluate the already saved results. You can use the following command to evaluate the results.
```bash
python tools/evaluate_offline.py configs/bevdet_occ_evaluation/occ3d_evaluation.py results/occ_submission
```
If your results are saved in a folder structure like Occ3D-nus gts, you can add the `--eval-occ3d` option to the command to evaluate the results.
```bash
python tools/evaluate_offline.py configs/bevdet_occ_evaluation/occ3d_evaluation.py results/ECCV/bevdet_stbase_val --eval-occ3d
```

## Test Submission
If you want to save the results for test server submission, you can use the following command to save the results.
```bash
./tools/dist_test.sh projects/configs/bevformer/bevformer_base_occ_test.py work_dirs/bevformer_base_occ/epoch_24.pth 8 --format-only --eval-options 'submission_prefix=./occ_submission'
```

## Visualization
