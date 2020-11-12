#!/usr/bin/env bash
PROTOC=protoc

cur_path=`pwd`
testcases_path=$1
cd $testcases_path

if [ `find proto/*.cc|wc -l` != 0 ]; then
  rm proto/*.cc
  rm proto/*.h
fi

if [ -f "proto/task.pb.h" ]; then
  echo "task pb has already protoc"
else
  $PROTOC -Iproto --cpp_out=proto proto/task.proto
  echo "protoc task pb success"
fi

if [ -f "proto/om.pb.h" ]; then
  echo "om pb has already protoc"
else
  $PROTOC -Iproto --cpp_out=proto proto/om.proto
  echo "protoc om pb success"
fi

if [ -f "proto/insert_op.pb.h" ]; then
  echo "insert_op pb has already protoc"
else
  $PROTOC -Iproto --cpp_out=proto proto/insert_op.proto
  echo "protoc insert_op pb success"
fi

if [ -f "proto/ge_ir.pb.h" ]; then
  echo "ge_ir pb has already protoc"
else
  $PROTOC -Iproto --cpp_out=proto proto/ge_ir.proto
  echo "protoc ge_ir pb success"
fi



if [ `find proto/tensorflow/*.cc | wc -l` != 0 ]; then
  rm proto/tensorflow/*.cc
  rm proto/tensorflow/*.h
fi

tf_proto_files=`ls proto/tensorflow/`
for tf_proto_file in $tf_proto_files; do
  $PROTOC -Iproto/tensorflow/ --cpp_out=proto/tensorflow/ proto/tensorflow/$tf_proto_files
  echo "protoc $tf_proto_file success "
done

cd $cur_path