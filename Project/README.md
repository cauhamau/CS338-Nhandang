# CS338-Nhandang
weights/batext/[**pretrain_attn_R_50.pth**](https://uithcm-my.sharepoint.com/:u:/g/personal/20520490_ms_uit_edu_vn/EfaY7y3nlTxFja1p1OHXwNQB0vjDjU9-906cXRDsmMUgQA?e=j6BjEJ)

vietocr/weights/[**vietocr.pth**](https://uithcm-my.sharepoint.com/:u:/g/personal/20520490_ms_uit_edu_vn/ETssn6jMkTpKmfJi91YCYKUBlvdaipxlpvbCHa0V7YFPlw?e=W9G8Os)

## Step1: Docker pull
```sh
docker pull cauhamau/dictguided-env:v1
```
## Step 2:Docker run
```sh
sudo docker run -it --gpus all --name dict -v .:/home iamgeID bash
```
## Step 3: Pip Install
```sh
cd home
python3.7 -m pip install flask
python3.7 -m pip install einops
```
## Step 4: Run app
```sh
python3.7 app.py
```
