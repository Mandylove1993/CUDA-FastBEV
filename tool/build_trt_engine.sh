#!/bin/bash
# configure the environment
. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

# tensorrt version
# version=`trtexec | grep -m 1 TensorRT | sed -n "s/.*\[TensorRT v\([0-9]*\)\].*/\1/p"`

# resnet18/resnet18int8/resnet18int8head
base=model/$DEBUG_MODEL

# fp16/int8
precision=$DEBUG_PRECISION

# precision flags
trtexec_fp16_flags="--fp16"
trtexec_dynamic_flags="--fp16"
if [ "$precision" == "int8" ]; then
    trtexec_dynamic_flags="--fp16 --int8"
fi

function get_onnx_number_io(){

    # $1=model
    model=$1

    if [ ! -f "$model" ]; then
        echo The model [$model] not exists.
        return
    fi

    number_of_input=`python -c "import onnx;m=onnx.load('$model');print(len(m.graph.input), end='')"`
    number_of_output=`python -c "import onnx;m=onnx.load('$model');print(len(m.graph.output), end='')"`
    # echo The model [$model] has $number_of_input inputs and $number_of_output outputs.
}

function compile_trt_model(){

    # $1: name
    # $2: precision_flags
    # $3: number_of_input
    # $4: number_of_output
    name=$1
    precision_flags=$2
    number_of_input=$3
    number_of_output=$4
    need_output_flg=$5
    result_save_directory=$base/build
    onnx=$base/$name.onnx

    if [ -f "${result_save_directory}/$name.plan" ]; then
        echo Model ${result_save_directory}/$name.plan already build 🙋🙋🙋.
        return
    fi
    
    # Remove the onnx dependency
    # get_onnx_number_io $onnx
    echo $number_of_input  $number_of_output

    input_flags="--inputIOFormats="
    output_flags="--outputIOFormats="
    for i in $(seq 1 $number_of_input); do
        input_flags+=fp16:chw,
    done

    for i in $(seq 1 $number_of_output); do
        output_flags+=fp16:chw,
    done

    if [ "$need_output_flg" == "need" ]; then
        cmd="--onnx=$base/$name.onnx ${precision_flags} ${input_flags} ${output_flags} \
            --saveEngine=${result_save_directory}/$name.plan \
            --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
            --dumpProfile --separateProfileRun \
            --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"
    else
        cmd="--onnx=$base/$name.onnx ${precision_flags} ${input_flags} \
            --saveEngine=${result_save_directory}/$name.plan \
            --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
            --dumpProfile --separateProfileRun \
            --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"
    fi
    echo $cmd
    mkdir -p $result_save_directory
    echo Building the model: ${result_save_directory}/$name.plan, this will take several minutes. Wait a moment 🤗🤗🤗~.
    trtexec $cmd > ${result_save_directory}/$name.log 2>&1
}

# maybe int8 / fp16
compile_trt_model "fastbev_pre_trt" "$trtexec_dynamic_flags"  1 1 "need"
compile_trt_model "fastbev_post_trt_decode" "$trtexec_dynamic_flags"  1 3 "noneed"
