TIMESTAMP=$(shell TZ=":Asia/Shanghai" date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=/work/log/dblLoad

test:
	export GLOG_v=10 && \
	export FLAGS_print_ir=1 && \
	export FLAGS_enable_pir_api=1 && \
	export FLAGS_prim_enable_dynamic=true && \
	mkdir -p $(LOG_DIR) && \
	python singleOPLab.py 
