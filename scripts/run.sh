#!/bin/bash
# 作用：获取当前脚本所在的目录路径，并将其存储在变量   ScriptPath   中。
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

echo "[INFO] Nanotrack starts to run"
running_command="./main"
cd ${ScriptPath}/../out
${running_command}
if [ $? -ne 0 ];then
    echo "[INFO] The program runs failed"
else
    echo "[INFO] The program runs successfully"
fi