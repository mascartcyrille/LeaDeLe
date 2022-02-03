cd build && rm -rf * && cmake -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DTBB_DIR=/usr/lib/x86_64-linux-gnu/ .. && make
