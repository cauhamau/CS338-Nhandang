
update=$1 # force update
caffe_repo=/data/pretrained/caffe/fcos/
onnx_repo=/data/pretrained/onnx/fcos/
pytorch_repo=/data/pretrained/pytorch/fcos/
ncnn_repo=/data/pretrained/ncnn/fcos/
case=FCOS_R_50_1x_bn_head

mkdir -p $caffe_repo $onnx_repo $pytorch_repo $ncnn_repo

if [ ! -e $onnx_repo/$case.onnx ] || [ "$update" != "" ];
then
  cd /workspace/git/uofa-AdelaiDet/ # folder of project https://github.com/aim-uofa/AdelaiDet
  pwd
  python3.7 -V # ensure python3.x
  python3.7 onnx/export_model_to_onnx.py \
    --config-file configs/BAText/VinText/attn_R_50.yaml \
    --output onnx.convert.onnx  \
    --opts MODEL.WEIGHTS /mlcv/WorkingSpace/Personals/thuongpt/scene_text_pipeline_bkai/Detecter/dict_guided/weights/weight_abcnet-dict-guided_final.pth MODEL.FCOS.NORM "BN" cpu
fi

if [ ! -e $onnx_repo/$case-update.onnx ] || [ "$update" != "" ];
then
  # advise version 1.3.0
  cd /workspace/git/onnx-simplifier  # folder of project: https://github.com/daquexian/onnx-simplifier
  pwd
  python3 -V # ensure python3.x
  python3.7 -m onnxsim $onnx_repo/$case.onnx $onnx_repo/$case-update.onnx
fi

# optional
if [ ! -e $caffe_repo/$case-update.caffemodel ];
then
  # switch to python2 and ensure caffe (with the upsample patch) ready
  # refer: https://github.com/blueardour/caffe.git  for patched version
  cd /workspace/git/onnx2caffe  # folder of project: https://github.com/MTlab/onnx2caffe
  pwd
  python -V
  python3.7 convertCaffe.py $onnx_repo/$case-update.onnx $caffe_repo/$case-update.prototxt $caffe_repo/$case-update.caffemodel
fi

# ncnn
if [ ! -e $ncnn_repo/$case-opt.bin ] || [ "$update" != "" ]
then
  cd /workspace/git/ncnn # folder of project: https://github.com/Tencent/ncnn
  pwd
  mkdir -p $ncnn_repo
  ./build-host-gcc-linux/tools/onnx/onnx2ncnn $onnx_repo/$case-update.onnx $ncnn_repo/$case-update.param $ncnn_repo/$case-update.bin
  if [ $? -eq 0 ]; then
    echo "Optimizing"
    ./build-host-gcc-linux/tools/ncnnoptimize $ncnn_repo/$case-update.param $ncnn_repo/$case-update.bin \
      $ncnn_repo/$case-update-opt.param $ncnn_repo/$case-update-opt.bin \
      0
  else
    echo "Convert failed"
  fi
fi

