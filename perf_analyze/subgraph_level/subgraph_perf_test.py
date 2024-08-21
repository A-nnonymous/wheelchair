import argparse
import os
import sys
import importlib.util

os.environ["FLAGS_enable_pir_api"]="1"
os.environ["FLAGS_cinn_bucket_compile"]="1"
os.environ["FLAGS_group_schedule_tiling_first"]="1"
os.environ["FLAGS_cinn_new_group_scheduler"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import paddle

def load_net_and_inputs(unit_name: str):
	paths = unit_name.split("^")
	file_path = ("/").join(paths) + ".py"

	spec = importlib.util.spec_from_file_location(paths[-1], file_path)
	module = importlib.util.module_from_spec(spec)
	sys.modules[paths[-1]] = module
	spec.loader.exec_module(module)

	return module.LayerCase(), module.create_tensor_inputs()


def test(net, inputs, mode, to_static, with_prim=False, with_cinn=False):
	net.eval()
	if to_static:
		paddle.set_flags({'FLAGS_prim_all': with_prim})
		if with_cinn:
			build_strategy = paddle.static.BuildStrategy()
			build_strategy.build_cinn_pass = True
			net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
		else:
			net = paddle.jit.to_static(net, full_graph=True)
	paddle.seed(123)
	#for _ in range(10):
	for _ in range(3):
		outs = net(*inputs)
	import time
#	start = time.time()
	for _ in range(5):
		outs = net(*inputs)
	if mode == "eager":
		paddle.core._cuda_synchronize(paddle.CUDAPlace(0))
#	end = time.time()
#	print("DEBUG time = ", end - start)
	return outs
	

def test_ast_prim_cinn(net, inputs, mode):
	if mode == "eager":
		st_out = test(net, inputs, mode, to_static=False)
	else:
		cinn_out = test(net, inputs, mode, to_static=True, with_prim=True, with_cinn=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("name", help="Unit test name")
	parser.add_argument("mode", choices=["cinn", "eager"], help="Choose mode: 'cinn' or 'eager'")
	args = parser.parse_args()

	net, inputs = load_net_and_inputs(args.name)
	test_ast_prim_cinn(net, inputs, args.mode)

