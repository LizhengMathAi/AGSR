# Run these commands one by one in this order.

# 1. Reproducibility audit:
python -u exp_reproducibility.py --train-root ./acdc_train --test-root ./acdc_test --threshold 0.25 --rho 1.0 --max-size 8 --vote-radius 2 --bootstrap 10000 --seed 20260827 --output ./exp_results/reproducibility_manifest.json 2>&1 | tee -a ./exp_results/experiment_results.txt

# 2. Matched baseline and no-flow proposed refinement:
python -u exp_refinement_masks.py --train-root ./acdc_train --test-root ./acdc_test --threshold 0.25 --rho 1.0 --max-size 8 --output-dir ./exp_results/refinements 2>&1 | tee -a ./exp_results/experiment_results.txt

# 3. Temporal voting only, without TV-L1:
python -u exp_temporal_controls.py --train-root ./acdc_train --test-root ./acdc_test --threshold 0.25 --vote-radius 2 --output-dir ./exp_results/temporal_controls --save-masks 2>&1 | tee -a ./exp_results/experiment_results.txt

# 4. Native SAM3 video propagation:
python -u exp_sam3_video.py --train-root ./acdc_train --test-root ./acdc_test --threshold 0.25 --device cuda --dtype bfloat16 --output-dir ./exp_results/sam3_native_video 2>&1 | tee -a ./exp_results/experiment_results.txt

# 5. Flow-only control using TV-L1:
python -u exp_temporal_controls.py --train-root ./acdc_train --test-root ./acdc_test --threshold 0.25 --vote-radius 2 --output-dir ./exp_results/temporal_controls_with_flow --save-masks --include-flow-only 2>&1 | tee -a ./exp_results/experiment_results.txt

# Step 5 also computes temporal voting. Therefore, if you intend to complete every cyan `XX`, you can skip step 3 and use the voting results from `temporal_controls_with_flow`.

# 6. Final statistics:
python -u exp_statistics.py --dataset "train=./acdc_train" --dataset "test=./acdc_test" --prediction "train:sam3=./exp_results/refinements/masks/train/sam3_matched" --prediction "train:voting=./exp_results/temporal_controls_with_flow/masks/train/temporal_voting" --prediction "train:flow_only=./exp_results/temporal_controls_with_flow/masks/train/flow_only_bidirectional" --prediction "train:proposed=./exp_results/refinements/masks/train/proposed" --prediction "train:native_video=./exp_results/sam3_native_video/masks/train/sam3_native_video" --prediction "test:sam3=./exp_results/refinements/masks/test/sam3_matched" --prediction "test:voting=./exp_results/temporal_controls_with_flow/masks/test/temporal_voting" --prediction "test:flow_only=./exp_results/temporal_controls_with_flow/masks/test/flow_only_bidirectional" --prediction "test:proposed=./exp_results/refinements/masks/test/proposed" --prediction "test:native_video=./exp_results/sam3_native_video/masks/test/sam3_native_video" --reference proposed --bootstrap 10000 --permutations 10000 --seed 20260827 --output-dir ./exp_results/statistics 2>&1 | tee -a ./exp_results/experiment_results.txt