#export PYTHONPATH=${PWD}

#first run
#python3 setup.py install

#Train
CUDA_VISIBLE_DEVICES="0,1" \
python3 tools/train_net.py \
--config-file configs/BAText/VinText/attn_R_50.yaml \
MODEL.WEIGHTS ./tt_e2e_attn_R_50.pth
