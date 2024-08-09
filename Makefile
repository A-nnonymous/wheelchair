#--------------------------------------- COMMON VARs ---------------------------------------
PADDLE_BUILD_DIR=/work/Paddle/build
PYTHON_WHL=$(PADDLE_BUILD_DIR)/python/dist/paddlepaddle_gpu-0.0.0-cp39-cp39-linux_x86_64.whl 

#---------------------------------------- MAKE CHANGE TO HAPPEN ----------------------------
recompile:
	cd $(PADDLE_BUILD_DIR) && make -j40

reinstall_whl:
	pip install -U $(PYTHON_WHL) --force-reinstall

happen:
	make recompile && make reinstall_whl 

#---------------------------------------- TEST RELATED VARs --------------------------------
SUBGRAPH_SET=./subgraph_targets.csv

#---------------------------------------- ALL TEST SCRIPTS  ---------------------------------
dblLoad:
	cd demo_dblLoad && make test

#----------------------------------------- TEST WRAPPER  ------------------------------------
test_dblLoad:
	make happen
	make dblLoad

	


