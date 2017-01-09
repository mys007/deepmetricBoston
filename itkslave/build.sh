#/bin/bash
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release; 
make
