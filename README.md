# fs-lmv

Official implementation [here](https://github.com/VicRanger/FSNet-LMV).

## Env

Use `./conda_env.yaml` to build the environment.

## Dataset

Please refer to official impl.

## Train and test

`python ./src/train.py --config ./src./configs/LMV.yaml`

`python ./src/test.py --config ./src./configs/LMV.yaml --model ./src/save/_LMV/epoch-best.pth`

See `./src/configs/LMV.yaml` for more info.