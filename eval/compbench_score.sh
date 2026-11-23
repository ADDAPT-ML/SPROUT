#!/bin/bash
# Score one T2I-CompBench++ category with its official scorer (external/T2I-CompBench README, "Evaluation"), run
# unmodified from its own directory in env_compbench. <dir> holds <dir>/samples from eval/compbench_generate.py.
#   color, shape, texture  BLIP-VQA              spatial  UniDet 2D       numeracy  UniDet numeracy
#   3d_spatial             UniDet 3D             non_spatial  CLIPScore   complex   BLIP-VQA, UniDet 2D, CLIPScore, 3-in-1
# usage: bash eval/compbench_score.sh <category> <dir>      (prints the category score)
set -euo pipefail
CAT=$1; OUT=$(realpath "$2")
REPO=$(cd "$(dirname "$0")/.." && pwd); C=$REPO/external/T2I-CompBench; V=$REPO/env_compbench/bin/python
blip() { (cd $C/BLIPvqa_eval && $V BLIP_vqa.py --out_dir=$OUT/); }
d2()   { (cd $C/UniDet_eval && $V 2D_spatial_eval.py --outpath $OUT/ "$@"); }
num()  { (cd $C/UniDet_eval && $V numeracy_eval.py --outpath $OUT/); }
d3()   { (cd $C/UniDet_eval && $V 3D_spatial_eval.py --outpath $OUT); }   # no trailing slash (the script parses the path)
clip() { (cd $C && $V CLIPScore_eval/CLIP_similarity.py --outpath=$OUT/ "$@"); }
case $CAT in
  color|shape|texture) blip; F=annotation_blip/vqa_result.json ;;
  spatial) d2; F=labels/annotation_obj_detection_2d/vqa_result.json ;;
  numeracy) num; F=annotation_num/vqa_result.json ;;
  3d_spatial) d3; F=labels/annotation_obj_detection_3d/vqa_result.json ;;
  non_spatial) clip; F=annotation_clip/vqa_result.json ;;
  complex) blip; d2 --complex True; clip --complex True
           (cd $C/3_in_1_eval && $V 3_in_1.py --outpath=$OUT/ --data_path=$C/examples/dataset)
           F=annotation_3_in_1/vqa_result.json ;;
  *) echo "unknown category $CAT"; exit 2 ;;
esac
python3 -c "import json, sys; v = [float(x['answer']) for x in json.load(open(sys.argv[1]))]; print(sum(v) / len(v))" $OUT/$F
