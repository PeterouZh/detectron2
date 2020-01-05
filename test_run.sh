


### bash /home/work/run_train.sh bash /home/work/user-job-dir/code/detectron2_origin/detectron2/test_run.sh

export ROOT_OBS=s3://bucket-1892/ZhouPeng
python /home/work/user-job-dir/code/template_lib/examples/copy_tool.py \
  -s $ROOT_OBS/code/ -d /cache/code/ -t copytree
  
## copy pretrained model
python /home/work/user-job-dir/code/template_lib/examples/copy_tool.py \
  -s $ROOT_OBS/torch/ -d $HOME/.torch -t copytree
## copy coco
python /home/work/user-job-dir/code/template_lib/examples/copy_tool.py \
  -s $ROOT_OBS/keras/coco/ -d /cache/code/detectron2_origin/detectron2/datasets/coco -t copytree

## install cocoapi
python /home/work/user-job-dir/code/template_lib/examples/copy_tool.py \
  -s $ROOT_OBS/pypi/detectron2/cocoapi.tar.gz -d $HOME/cocoapi.tar.gz -t copy
cd $HOME
tar -zxvf cocoapi.tar.gz
cd cocoapi/PythonAPI && python setup.py install --user

## install fvcore
python /home/work/user-job-dir/code/template_lib/examples/copy_tool.py \
  -s $ROOT_OBS/pypi/detectron2/fvcore-0.1.dev200104.tar.gz -d $HOME/fvcore-0.1.dev200104.tar.gz -t copy
cd $HOME
pip install fvcore-0.1.dev200104.tar.gz

cd /cache/code/detectron2_origin/detectron2
python setup.py build develop

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train_net.py \
  --num-gpus 8 \
  --config-file configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml


# python3 tools/train_net.py \
#   --num-gpus 6 \
#   --config-file configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml \
#   SOLVER.IMS_PER_BATCH 12 SOLVER.BASE_LR 0.015

# python setup.py build develop