#include <iostream>
#include <fstream>
#include <utility>

#include <NvOnnxParser.h>

#include "infer.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calibrator.h"
#include "utils.h"

using namespace nvinfer1;


YoloDetector::YoloDetector(
    std::string  trtFile,
    int gpuId,
    float nmsThresh,
    float confThresh,
    int numClass
): trtFile_(std::move(trtFile)), nmsThresh_(nmsThresh), confThresh_(confThresh), numClass_(numClass)
{
    gLogger = Logger(ILogger::Severity::kERROR);
    cudaSetDevice(gpuId);

    CHECK(cudaStreamCreate(&stream));

    // load engine
    get_engine();

    context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32{4, {1, 3, kInputH, kInputW}});

    // get engine output info
    Dims32 outDims = context->getBindingDimensions(1); // [1, 84, 8400]
    OUTPUT_CANDIDATES = outDims.d[2]; // 8400
    int outputSize = 1; // 84 * 8400
    for (int i = 0; i < outDims.nbDims; i++)
    {
        outputSize *= outDims.d[i];
    }

    // prepare output data space on host
    outputData = new float[1 + kMaxNumOutputBbox * kNumBoxElement];
    // prepare input and output space on device
    vBufferD.resize(2, nullptr);
    CHECK(cudaMalloc(&vBufferD[0], 3 * kInputH * kInputW * sizeof(float)));
    CHECK(cudaMalloc(&vBufferD[1], outputSize * sizeof(float)));

    CHECK(cudaMalloc(&transposeDevice, outputSize * sizeof(float)));
    CHECK(cudaMalloc(&decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float)));
}

void YoloDetector::get_engine()
{
    if (access(trtFile_.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile_, std::ios::binary);
        long int fsize = 0;

        engineFile.seekg(0, std::ifstream::end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, std::ifstream::beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.empty())
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        std::cout << "Failed loading engine!" << std::endl;
    }
}

YoloDetector::~YoloDetector()
{
    cudaStreamDestroy(stream);

    for (int i = 0; i < 2; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }

    CHECK(cudaFree(transposeDevice));
    CHECK(cudaFree(decodeDevice));

    delete [] outputData;

    delete context;
    delete engine;
    delete runtime;
}

std::vector<Detection> YoloDetector::inference(cv::Mat& img) const
{
    if (img.empty()) return {};

    // put input on device, then letterbox、bgr to rgb、hwc to chw、normalize.
    preprocess(img, (float*)vBufferD[0], kInputH, kInputW, stream);

    // tensorrt inference
    context->enqueueV2(vBufferD.data(), stream, nullptr);

    // transpose [1 84 8400] convert to [1 8400 84]
    transpose((float*)vBufferD[1], transposeDevice, OUTPUT_CANDIDATES, numClass_ + 4, stream);
    // convert [1 8400 84] to [1 7001]
    decode(transposeDevice, decodeDevice, OUTPUT_CANDIDATES, numClass_, confThresh_, kMaxNumOutputBbox, kNumBoxElement,
           stream);
    // cuda nms
    nms(decodeDevice, nmsThresh_, kMaxNumOutputBbox, kNumBoxElement, stream);

    CHECK(cudaMemcpyAsync(outputData, decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    std::vector<Detection> vDetections;
    int count = std::min((int)outputData[0], kMaxNumOutputBbox);
    for (int i = 0; i < count; i++)
    {
        int pos = 1 + i * kNumBoxElement;
        int keepFlag = (int)outputData[pos + 6];
        if (keepFlag == 1)
        {
            Detection det{};
            memcpy(det.bbox, &outputData[pos], 4 * sizeof(float));
            det.conf = outputData[pos + 4];
            det.classId = (int)outputData[pos + 5];
            vDetections.push_back(det);
        }
    }

    for (auto& vDetection : vDetections)
    {
        scale_bbox(img, vDetection.bbox);
    }

    return vDetections;
}

void YoloDetector::draw_image(cv::Mat& img, std::vector<Detection>& inferResult)
{
    // draw inference result on image
    for (auto& i : inferResult)
    {
        cv::Scalar bboxColor(get_random_int(), get_random_int(), get_random_int());
        cv::Rect r(
            round(i.bbox[0]),
            round(i.bbox[1]),
            round(i.bbox[2] - i.bbox[0]),
            round(i.bbox[3] - i.bbox[1])
        );
        cv::rectangle(img, r, bboxColor, 2);

        const std::string& className = vClassNames[(int)i.classId];
        std::string labelStr = className + " " + std::to_string(i.conf).substr(0, 4);

        cv::Size textSize = cv::getTextSize(labelStr, cv::FONT_HERSHEY_PLAIN, 1.2, 2, nullptr);
        cv::Point topLeft(r.x, r.y - textSize.height - 3);
        cv::Point bottomRight(r.x + textSize.width, r.y);
        cv::rectangle(img, topLeft, bottomRight, bboxColor, -1);
        cv::putText(img, labelStr, cv::Point(r.x, r.y - 2), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2,
                    cv::LINE_AA);
    }
}
