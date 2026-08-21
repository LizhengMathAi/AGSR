python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.0
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.1
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.2
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.3
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.4
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.5
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.6
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.7
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.8
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.9
python unit_eval.py --root-dir ./acdc_train --mode self_derived_prompts
python unit_eval.py --root-dir ./acdc_train --mode annotation_derived_prompts

python unit_eval.py --root-dir ./acdc_test --mode baseline --thres 0.0
python unit_eval.py --root-dir ./acdc_test --mode baseline --thres 0.1
python unit_eval.py --root-dir ./acdc_test --mode baseline --thres 0.2
python unit_eval.py --root-dir ./acdc_test --mode baseline --thres 0.3
python unit_eval.py --root-dir ./acdc_test --mode baseline --thres 0.4
python unit_eval.py --root-dir ./acdc_test --mode baseline --thres 0.5
python unit_eval.py --root-dir ./acdc_test --mode baseline --thres 0.6
python unit_eval.py --root-dir ./acdc_test --mode baseline --thres 0.7
python unit_eval.py --root-dir ./acdc_test --mode baseline --thres 0.8
python unit_eval.py --root-dir ./acdc_test --mode baseline --thres 0.9
python unit_eval.py --root-dir ./acdc_test --mode self_derived_prompts
python unit_eval.py --root-dir ./acdc_test --mode annotation_derived_prompts

python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.0 --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.1 --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.2 --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.3 --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.4 --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.5 --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.6 --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.7 --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.8 --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.9 --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode self_derived_prompts --operation areas --patient-id 1
python unit_eval.py --root-dir ./acdc_train --mode annotation_derived_prompts --operation areas --patient-id 1


# python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.0 --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.1 --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.2 --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.3 --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.4 --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.5 --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.6 --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.7 --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.8 --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.9 --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode self_derived_prompts --operation neighbor_iou --patient-id 1
# python unit_eval.py --root-dir ./acdc_train --mode annotation_derived_prompts --operation neighbor_iou --patient-id 1