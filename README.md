# AscendCL Nanotrack 在Atlas 200I DK A2上部署推理
  
## 项目介绍

基于ONNX框架的Nanotrack模型，对视频进行目标跟踪，最高可达70FPS.

设置 constexpr bool INFER_ONCE = true 以进行追踪测试；否，进行视频追踪


**代码目录说明如下：**

```
|———data         // 用于存放测试视频
|———image        // 用于存放测试图片
|———inc          // 用于存放头文件
|———model        // 用于存放模型文件
|———out          // 用于存放可执行文件
|———results      // 用于存放追踪结果
|———script       // 用于存放编译、运行样例的脚本                                              
|———src          // 用于存放源码
```

## 准备环境<a name="section1835415517712"></a>

1.  安装CANN软件。

    单击[Link](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，获取最新版本的CANN软件安装指南。

    **注意：**此处还可以在页面左上侧切换版本，查看对应版本的安装指南。
    **本项目使用的版本为CANN社区版8.0.0.alpha003**

2.  设置环境变量。

    **注：**“$HOME/Ascend”请替换“Ascend-cann-toolkit”包的实际安装路径。

    ```
    # 设置CANN依赖的基础环境变量
     bash /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # 配置程序编译依赖的头文件与库文件路径
    export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest 
    export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
    ```

3.  安装OpenCV。

    执行以下命令安装opencv，**确保是3.x版本**：

    ```
    sudo apt-get install libopencv-dev
    ```

## 项目运行<a name="section012033382418"></a>

1.  **获取ONNX框架的NanoTrack模型（\*.onnx），并转换为昇腾AI处理器能识别的模型（\*.om）。**

    **注：**此处以昇腾310B4 AI处理器为例，针对其它昇腾AI处理器的模型转换，需修改atc命令中的--soc\_version参数值。

    ```
    # 在这里直接给出原始模型转换命令,可以直接拷贝执行。
    具体操作说明：https://www.hiascend.com/document/detail/zh/Atlas200IDKA2DeveloperKit/23.0.RC2/Application%20Development%20Guide/tmuacop/tmuacop_0025.html

    特别说明：  模型的输入节点名称和shape，shape的格式一般为[batch,channels,height,width]。
               一般情况下不需要使用该参数，如果要转换的模型为动态shape的ONNX模型时，需要使用该参数并填写shape。

    标准格式：atc --model=model.onnx --framework=5 --output=model --soc_version=Ascend310B4 
**Nanotrack转换指令**
```bash
     atc --model=nanotrack_backbone_127.onnx --framework=5 --output=nanotrack_backbone_127 --soc_version=Ascend310B4

     atc --model=nanotrack_backbone_255.onnx --framework=5 --output=nanotrack_backbone_255 --soc_version=Ascend310B4

     atc --model=nanotrack_head.onnx --framework=5 --output=nanotrack_head --soc_version=Ascend310B4
```

    atc命令中各参数的解释如下，详细约束说明请参见[《ATC模型转换指南》](https://hiascend.com/document/redirect/CannCommunityAtc)。

    -   --model：NanoTrack网络的模型文件的路径。
    -   --framework：原始框架类型。5表示ONNX。
    -   --nanotrack_backbone_127.om等模型文件的路径。请注意，记录保存该om模型文件的路径，后续开发应用时需要使用。
    -   --input\_shape：模型输入数据的shape。
    -   --soc\_version：昇腾AI处理器的版本。

        >**说明：** 
        >如果无法确定当前设备的soc\_version，则在安装驱动包的服务器执行**npu-smi info**命令进行查询，在查询到的“Name“前增加Ascend信息，例如“Name“对应取值为_xxxyy_，实际配置的soc\_version值为Ascend_xxxyy_。


2.  **获取测试视频/图片数据。**

    请将获取的测试视频放在data目录下，图片放在image目录下

    **注：若需更换测试图片，则需自行准备测试图片，并将测试图片放到image目录下。**

3.  **编译源码。**

    执行以下命令编译源码。

    ```
    cd ./scripts 
    bash build.sh
    ```

4.  **运行。**

    执行以下脚本运行样例：

    ```
    bash run.sh
    ```

    执行成功后，在屏幕上的关键提示信息示例如下，这些值可能会根据版本、环境有所不同，请以实际情况为准：

    ```
    [INFO]   Rectangle frame SUCCES
    now frame  is 987
    End of video or unable to read frame.
    [INFO]  ACNNDeInit SUCCES
    [INFO] The program runs successfully
    ```

    图片结果展示：
    [跟踪结果](results/INFER_ONCE_result.jpg)


## 相关操作<a name="section27901216172515"></a>

-   获取更多样例，请单击[Link](https://gitee.com/ascend/samples/tree/master/inference/modelInference)。
-   获取在线视频课程，请单击[Link](https://www.hiascend.com/edu/courses?activeTab=%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91)。
-   获取学习文档，请单击[AscendCL C&C++](https://hiascend.com/document/redirect/CannCommunityCppAclQuick)或[AscendCL Python](https://hiascend.com/document/redirect/CannCommunityPyaclQuick)，查看最新版本的AscendCL推理应用开发指南。
-   查模型的输入输出

    可使用第三方工具Netron打开网络模型，查看模型输入或输出的数据类型、Shape，便于在分析应用开发场景时使用。
    


