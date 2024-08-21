import os
import sys

os.environ["GLOG_v"]="10"
os.environ["FLAGS_print_ir"]="1"
os.environ["FLAGS_enable_pir_api"]="1"
os.environ["FLAGS_prim_enable_dynamic"]="true"

compiler_mode = ["cinn", "eager"]
result_dir = "/work/mytest/perf_analyze/result/"

def parse_csv_for_paths(case_dir,csv_file):
    paths2Tags = {}
    with open(csv_file, 'r') as file:
        for line in file:
            parts = line.strip().split('^')
            #file_path = case_dir.join(parts) + ".py"
            file_path = os.path.join(case_dir, *parts[:])
            model_tag = '-'.join(parts[-2:])
            paths2Tags[file_path ] = model_tag
    return paths2Tags

def main(case_dir, csv_file):
    print(f"CASE_DIR: {case_dir}")
    
    paths2Tags = parse_csv_for_paths(case_dir, csv_file)
    for path, tag in paths2Tags.items():
        for mode in compiler_mode:
            exec_script_command = " python perf_test.py " + str(path) + " " + str(mode)

            resultfile_dir = result_dir + str(tag) + "/" + mode 

            os.system("mkdir -p " + resultfile_dir)

            nsys_command = "nsys profile --stats true -w true -t cuda,cudnn,cublas" + \
                            " -o " + resultfile_dir + "/result" + \
                            exec_script_command + \
                            " > " + resultfile_dir + "/result.log 2>&1"

            ncu_command = "ncu --target-processes all  --set full  --import-source yes -f -o" + \
                            resultfile_dir + "/result" + \
                            exec_script_command

            print("Executing command: " + nsys_command)
            os.system(nsys_command)
            print("Executing command: " + ncu_command)
            os.system(ncu_command)


if __name__ == "__main__":
    case_dir = sys.argv[1] if len(sys.argv) > 1 else ''
    csv_file = sys.argv[2] if len(sys.argv) > 2 else ''
    main(case_dir, csv_file)
