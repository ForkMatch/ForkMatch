mkdir build
cd build
nvcc ../src/smsm.cu -o ForkMatch -Xcompiler "-std=c++20 -O3 -fpermissive" -maxrregcount=64 -rdc=true --gpu-architecture=compute_75 --machine 64 -cudart static -std c++20 --keep
