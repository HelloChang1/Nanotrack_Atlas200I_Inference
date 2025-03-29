#!/bin/bash
# 作用：获取当前脚本所在的目录路径，并将其存储在变量   ScriptPath   中。
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
# 作用：定义模型文件所在的路径。
ModelPath="${ScriptPath}/../model"

function build()
{
  # 如果目录   ${ScriptPath}/../build/intermediates/host   存在，则递归删除该目录。
  if [ -d ${ScriptPath}/../build/intermediates/host ];then
    rm -rf ${ScriptPath}/../build/intermediates/host
  fi
  #  创建目录（如果不存在）并切换到该目录。
  mkdir -p ${ScriptPath}/../build/intermediates/host
  cd ${ScriptPath}/../build/intermediates/host

  # 调用 CMake 配置项目，指定 C++ 编译器为   g++  ，并跳过 RPATH 设置。
  cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
  # 如果 CMake 配置失败（  $? -ne 0  ），打印错误信息并返回 1。
  if [ $? -ne 0 ];then
    echo "[ERROR] cmake error, Please check your environment!"
    return 1
  fi
  # 构建项目。
  make
  # 如果构建失败（  $? -ne 0  ），打印错误信息并返回 1。
  if [ $? -ne 0 ];then
    echo "[ERROR] build failed, Please check your environment!"
    return 1
  fi
  # 返回到脚本最初所在的目录，并将切换目录的输出重定向到   /dev/null  ，避免输出干扰。
  cd - > /dev/null
}

function main()
{
  # 打印一条信息，表示开始准备样本。
  echo "[INFO] Sample preparation"
  # 使用   find   命令在   ${ModelPath}   目录下查找名为   resnet50.om   的文件。
  ret=`find ${ModelPath} -maxdepth 1 -name nanotrack_backbone_127.om 2> /dev/null`

  # • 如果找到模型文件（  ret   不为空），打印信息并继续。
  # • 如果未找到模型文件，打印错误信息并返回 1。
   if [[ ${ret} ]];then
      echo "[INFO] The nanotrack_backbone_127.om already exists.start buiding"
    else
      echo "[ERROR] nanotrack_backbone_127.om does not exist, please follow the readme to convert the model and place it in the correct position!"
      return 1
    fi
  # 调用   build   函数进行构建。
  build
  # 检查   build   函数的返回值，如果失败（返回值不为 0），返回 1。
  if [ $? -ne 0 ];then
    return 1
  fi

  echo "[INFO] Sample preparation is complete"
}
main