# TODO add real path
export ATC_HOME=
export OP_TEST_FRAME_INSTALL_HOME=
# conifg xxx/built-in/tbe
export OPS_SOURCE_PATH=

export PYTHONPATH=$PYTHONPATH:$ATC_HOME/python/site-packages/te:$ATC_HOME/python/site-packages/topi:$OPS_SOURCE_PATH:$OP_TEST_FRAME_INSTALL_HOME
export LD_LIBRARY_PATH=$ATC_HOME/lib64
export PATH=$PATH:$ATC_HOME/ccec_compiler/bin
