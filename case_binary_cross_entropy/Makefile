TIMESTAMP=$(shell TZ=":Asia/Shanghai" date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=/work/wheelchair/case_binary_cross_entropy/result


exec_both:
	export GLOG_v=10 && \
	export FLAGS_print_ir=1 && \
	export FLAGS_enable_pir_api=1 && \
	export FLAGS_prim_enable_dynamic=true && \
  export CUDA_VISIBLE_DEVICES=7 && \
	mkdir -p $(LOG_DIR) && \
	python binary_cross_entropy.py > $(LOG_DIR)/$(TIMESTAMP).log 2>&1

test:
	ncu --target-processes all  --set full  --import-source yes -f -o $(LOG_DIR)/result make exec_both
