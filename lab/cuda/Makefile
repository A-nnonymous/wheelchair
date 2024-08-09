.PHONY:
	run
recompile:
	nvcc -O3 -lineinfo -std=c++17 -o main ./main.cu
profile:
	ncu --target-processes all  --set full  --import-source yes -f -o result ./main

run:
	make recompile && make profile
