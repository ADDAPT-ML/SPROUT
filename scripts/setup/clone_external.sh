#!/bin/bash
# Clone the code of the methods and benchmarks used into external/, at the commits used for the paper.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p external
clone() {   # <dir> <url> <commit>
  [ -d external/$1 ] || git clone "$2" external/$1
  git -C external/$1 checkout -q "$3"
}
clone dgpo https://github.com/Luo-Yihong/DGPO.git 6a45077c4b743e6b05ec7b3c2c49bd2be0783694   # + the GenEval training set
clone cgpo https://github.com/PRIS-CV/CGPO.git c9133d676069682db4119e7fd6334b373064a926
clone geneval https://github.com/djghosh13/geneval.git af4902f24d3ca90ebbb446dd9891a59e0f82725f
clone mmdetection https://github.com/open-mmlab/mmdetection.git e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1   # branch 2.x
clone ELLA https://github.com/TencentQQGYLab/ELLA.git 3c228f1dc6c4d3cad0a47493816151a419f14db3          # DPG-Bench
clone T2I-CompBench https://github.com/Karine-Huang/T2I-CompBench.git 4aa404212eb5d06e5adbcd9cee696c750d0d25a5
clone PickScore https://github.com/yuvalkirstain/PickScore.git f79607ff68609c1e6dfa78cba38afeea4c248df1
clone HPSv2 https://github.com/tgxs002/HPSv2.git 866735ecaae999fa714bd9edfa05aa2672669ee3
clone ImageReward https://github.com/THUDM/ImageReward.git 78f8051059b34fef8e0032ccb40561c23a2ddbbb
clone improved-aesthetic-predictor https://github.com/christophschuhmann/improved-aesthetic-predictor.git 6934dd81792f086e613a121dbce43082cb8be85e
