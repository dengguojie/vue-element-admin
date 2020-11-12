python_cmd=$1
whl_path=.

echo ${whl_path}
cd ${whl_path}

cp ../op_gen/msopgen.py ${whl_path}
mv ${whl_path}/msopgen.py ${whl_path}/msopgen
${python_cmd} setup.py bdist_wheel
cp -r ${whl_path}/dist/* ${whl_path}/build
rm -rf ${whl_path}/op_gen.egg-info
rm -rf ${whl_path}/dist
rm -rf ${whl_path}/msopgen


