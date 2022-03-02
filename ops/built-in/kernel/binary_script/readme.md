# Binary Script Guide

## build binary steps

```
build_binary.sh soc_version opp_run_path output_path
    gen opcinfo.csv    <<< use gen_opcinfo_from_opinfo.py and opinfo json of {soc_version}
    for op_type in all_binary_op_list:
        bash build_binary_single_op.sh {op_type} {soc_version} {opp_run_path} {output_path}
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

### build_binary_single_op.sh

```
编译单个算子二进制kernel脚本
使用方法：
1. 在路径ops/built-in/kernel/binary_config 下添加对应平台的二进制发布json文件, 见Sample
2. 在文件ops/built-in/kernel/binary_config/binary_config.csv  添加算子对应的二进制发布json文件文件, 见Sample

执行方法：
  bash build_binary_single_op.sh {op_type} {soc_version} {opp_run_path} {output_path} {opcinfo文件}(可选, 不传内部自动生成)
    ex: bash build_binary_single_op.sh Add Ascend910 /usr/local/Ascend/opp/ ./add

入参说明
    {op_type}      算子名字, ex：Add
    {soc_version}  编译平台, ex: Ascend910
    {opp_run_path} opp包路径, 预留没使用, ex：/usr/local/Ascend/opp/
    {output_path}  二进制kernel输出路径
                     二进制文件路径路径规划 {output_path}/op_impl/built-in/ai_core/tbe/kernel/{soc_version}/{op_type}/
                     二进制config文件      {output_path}/op_impl/built-in/ai_core/tbe/kernel/config/{soc_version}/{op_type}.json
    {opcinfo文件}  可选, opc编译使用的csv文件, 通过信息库和gen_opcinfo_from_opinfo.py脚本产生, 没有传参数内部可自动生成

输出说明：
    二进制文件
        路径路径规划 {output_path}/op_impl/built-in/ai_core/tbe/kernel/{soc_version}/{xxxx}/      // {xxxx}为算子实现文件前缀, 小写加下划线
            其中路径为全小写
        ex：{output_path}/op_impl/built-in/ai_core/tbe/kernel/ascend910/add/add_0.o
            {output_path}/op_impl/built-in/ai_core/tbe/kernel/ascend910/add/add_0.json
    二进制config文件, 记录该算子二进制kernel信息
        路径规划： {output_path}/op_impl/built-in/ai_core/tbe/kernel/config/{soc_version}/{xxxx}.json  // {xxxx}为算子实现文件前缀, 小写加下划线
              ex: {output_path}/op_impl/built-in/ai_core/tbe/kernel/config/ascend910/add.json
```

### gen_opcinfo_for_socversion.sh

```
区分平台生成opc 编译算子使用的信息, 动态算子实现文件(dynamic/add.py)和算子编译入口函数(add)
内部使用函数 gen_opinfo_json_from_ini.sh + gen_opcinfo_from_opinfo.py

执行方法：
  bash gen_opcinfo_for_socversion.sh {soc_version} {out_opcinfo_csv_file}
    ex: bash gen_opcinfo_for_socversion.sh Ascend910 ./opc_info_ascend910.csv

入参说明
    {soc_version}           平台, ex: Ascend910
    {out_opcinfo_csv_file}  opcinfo保存路径, json文件后缀  ex: ./opc_info_ascend910.csv

输出说明：
    csv格式的opc编译信息, 格式如下
    op_type,file_name,file_func
    Abs,dynamic/abs.py,abs
    AbsGrad,dynamic/abs_grad.py,abs_grad
```

### gen_opinfo_json_from_ini.sh

```
生成对应平台信息库(json文件), 依赖cann仓原码
执行方法：
    bash gen_opinfo_json_from_ini.sh {soc_version} {output_json_file}
        ex: bash gen_opinfo_json_from_ini.sh Ascend910 910.json

入参说明
    {soc_version}       平台, ex: Ascend910
    {output_json_file}  信息库保存路径, json文件后缀  ex: /home/910.json

输出说明：
    json格式的算子信息库
```

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

### binary_fuzz_json.sh

```
生成对应平台的二进制发布json文件
执行方法：
1. 在binary_json_cfg.ini 中配置算子kernel发布的config, 见Addd, Cast
2. 执行 bash binary_fuzz_json.sh {op_type} {soc_version}
    ex: bash binary_fuzz_json.sh Add,Cast ascend910,ascend610

入参说明
    {op_type}      算子名字, ex：1.Add(单算子)
                                2.Add,Cast(多算子)
                                3.all(全量)
    {soc_version}  编译平台, ex: 1.Ascend910(单平台)
                                2.Ascend910,Ascend910(多平台)
                                3.all(全平台)

输出json文件说明：
    路径规划 ../binary_config/{soc_version}/{op_type}/{xxx}.json    //xxx为算子实现文件前缀, 即binary_json_cfg.ini中op_name
    ex: ../binary_config/ascend910/Add/add.json
```

### binary_mate_json.sh

```
检测输入是否匹配二进制kernel
执行方法：
    bash binary_fuzz_json.sh {op_type} {binary_file} {input_tensors}
    ex: bash binary_fuzz_json.sh Sample ${Asend}/op_impl/built-in/ai_core/tbe/kernel/config/ascend910/sample.json ./test.json

入参说明
    {op_type}        算子名字
    {binary_file}    算子kernel的总json文件
    {input_tensors}  要检测的输入(ex: binary_config/ascend910/Add/test.json)
```

### gen_output_json.py

```
获取opc工具生成的supportInfo合并到binList里生成二进制kernel总json文件
执行方法：
    python3 gen_opcinfo_from_opinfo.py {binary_config_full_path} {binary_compile_full_path} {binary_compile_json_full_path}

入参说明:
     1.二进制kernel发布场景json文件, 2.opc工具生成的supportinfo, 3.kernel的总json文件(1 + 2)
```
