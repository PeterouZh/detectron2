

# python setup.py build develop

python tools/train_net.py \
  --num-gpus 8 \
	--config-file configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml

# python3 tools/train_net.py \
#   --num-gpus 6 \
#   --config-file configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml \
#   SOLVER.IMS_PER_BATCH 12 SOLVER.BASE_LR 0.015