# Binary Script Guide

## build binary steps

```
build_binary.sh soc_version opp_run_path output_path
    gen opcinfo.csv    <<< use gen_opcinfo_from_opinfo.py and opinfo json of {soc_version}
    for op_type in all_binary_op_list:
        bash build_binary_single_op.sh {soc_version} {opp_run_path} {output_path} {op_type}
            get 算子py文件                       <<< from opcinfo.csv
            get 算子编译函数 op_func             <<< from opcinfo.csv
            get 算子二进制json文件 binary.json   <<< from binary_config.csv or binary_config_{soc_version}.csv
            拼接算子py文件绝对路径                <<< py_file={opp_run_path}/op_impl/built-in/ai_core/tbe/impl/{算子py文件}
            拼接输出绝对路径                     <<< kernel_output={output_path}/op_impl/built-in/ai_core/tbe/kernel/{soc_version}/{op_type}/
            opc编译二进制                        <<< opc py_file --main_func={op_func} --input_param={binary.json} --soc_version={soc_version} --output={kernel_output}

            json_name=算子py文件前缀.json
            拼接发布二进制总json文件    <<< json_output={output_path}/op_impl/built-in/ai_core/tbe/kernel/config/{soc_version}/{json_name}.json
            输出发布二进制总json文件    <<< python3 xxxx.py {binary.json} {kernel_output} {json_output}
```

## Script Detail

### gen_opcinfo_from_opinfo.py

```
从信息库提取算子编译函数, 包括动态算子实现文件(dynamic/add.py)和算子编译入口函数(add)
执行方法：
  python3 gen_opcinfo_from_opinfo.py ./aic-ascend910-ops-info.json ./opc_info_ascend910.csv

入参说明:
  json文件(可以多个), csv后缀的文件(输出文件)

输出csv格式:
  op_type,file_name,file_func
  Abs,dynamic/abs.py,abs
  AbsGrad,dynamic/abs_grad.py,abs_grad
```
