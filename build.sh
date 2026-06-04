#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
ROS_DISTRO="${ROS_DISTRO:-humble}"

source "/opt/ros/$ROS_DISTRO/setup.bash"

A=(--symlink-install --cmake-clean-cache --cmake-args -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CXX_FLAGS=-O3 -march=native' '-DCMAKE_C_FLAGS=-O3 -march=native')

if command -v /usr/local/cuda/bin/nvcc >/dev/null 2>&1 &&
   (( $(gcc -dumpversion | cut -d. -f1) > 13 )) &&
   (( $(g++ -dumpversion | cut -d. -f1) > 13 )); then
  v=$(ls /usr/bin/gcc-[0-9]* 2>/dev/null | sed 's/.*-//' | sort -nr | awk '$1<=13&&system("test -x /usr/bin/g++-"$1)==0{print;exit}')
  : "${v:?no gcc/g++ <= 13 found}"
  A+=(-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-$v -DCMAKE_C_COMPILER=/usr/bin/gcc-$v -DCMAKE_CXX_COMPILER=/usr/bin/g++-$v)
fi

colcon build "${A[@]}"
