#! /bin/bash

mkdir build
cd build
cmake ..
make

if [ $? -eq 0 ]; then
  echo "Build successful"
  # ./test/test_mips_index "../datasets/mnist/database_mnist.bin" "../output/mnist/database_mnist.knng" 800 40 60 "../output/mnist/mnist.mips" 784
  ./test/test_mips_search "../datasets/mnist/database_mnist.bin" "../datasets/mnist/query_mnist.bin" "../output/mnist/mnist.mips" 1000 100 "../output/mnist/result.txt" 784

else
  echo "Build failed"
fi