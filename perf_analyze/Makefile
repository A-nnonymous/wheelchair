SUBGRAPH_CASE_DIR=/work/PaddleTest/framework/e2e/PaddleLT_new/layercase
SUBGRAPH_JOB_DIR=./targets/subgraph
SUBGRAPH_JOB_CSV=$(SUBGRAPH_JOB_DIR)/simplified_subgraph_targets.csv

GPU_CHOICE=4
NLP_SCRIPTS_DIR=/work/PaddleNLP

test_llama:
	export CUDA_VISIBLE_DEVICES=$(GPU_CHOICE) &&\
	cd $(NLP_SCRIPTS_DIR) && \
	python ./llm/predictor.py \
	--model_name_or_path ./llama_inference \
	--dtype float16 --src_length 1024 --max_length 1024 \
	--output_file ./llama_inference/output.json \
	--decode_strategy greedy_search \
	--mode static

test_llama_1:
	export CUDA_VISIBLE_DEVICES=$(GPU_CHOICE) &&\
	export FLAGS_enable_pir_in_executor=1 && \
	cd $(NLP_SCRIPTS_DIR) && \
	python ./llm/predictor.py \
	--model_name_or_path ./llama_inference \
	--dtype float16 --src_length 1024 --max_length 1024 \
	--output_file ./llama_inference/output.json \
	--decode_strategy greedy_search \
	--mode static

test_llama_2:
	export CUDA_VISIBLE_DEVICES=$(GPU_CHOICE) &&\
	export FLAGS_enable_pir_in_executor=1 && \
	export FLAGS_prim_enable_dynamic=true && export FLAGS_prim_all=true && \
	export FLAGS_pir_apply_shape_optimization_pass=true && \
	export FLAGS_use_cinn=true  && \
	export FLAGS_cinn_new_group_scheduler=1 && \
	export FLAGS_group_schedule_tiling_first=1 && \
	export FLAGS_cinn_bucket_compile=1 && \
	cd $(NLP_SCRIPTS_DIR) && \
	python ./llm/predictor.py \
	--model_name_or_path ./llama_inference \
	--dtype float16 --src_length 1024 --max_length 1024 \
	--output_file ./llama_inference/output.json \
	--decode_strategy greedy_search \
	--mode static
  
test_190subgraph:
	python summon_test.py $(CASE_DIR) $(SUBGRAPH_JOB_CSV)
