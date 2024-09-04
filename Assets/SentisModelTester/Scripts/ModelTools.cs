using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Unity.Sentis;
using Microsoft.ML.OnnxRuntime;
//using Microsoft.ML.OnnxRuntime.Gpu;
using System;
using System.Linq;
using System.IO;


/// <summary>
/// ModelTools includes methods for model inference on Sentis and OnnxRuntime.
/// </summary>
public class ModelTools 
{

    // ORT inference
    public static Dictionary<string, float[]> DoOrtInference(string modelPath, float[] inputData, Vector3Int imageSize)
    {
        Dictionary<string, float[]> outTensors = new Dictionary<string, float[]>();

        try
        {
            SessionOptions sessionOptions = new SessionOptions();
#if UNITY_STANDALONE_WIN
            //sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0);  // gpuId
#endif

            using (InferenceSession session = new InferenceSession(modelPath, sessionOptions))
            {
                // setup sample input data
                var inputMeta = session.InputMetadata;
                var inputTensors = new List<NamedOnnxValue>();  // we consider only 1 input instead of 'inputMeta.Count'
                var inputNames = new List<string>();

                foreach (var name in inputMeta.Keys)
                {
                    //string shapeStr = string.Join(", ", inputMeta[name].Dimensions.ToArray());
                    //Debug.Log($"ORT inputTensor: {name} - shape: ({shapeStr})");

                    int[] inputDims = inputMeta[name].Dimensions;
                    int fi = inputDims.Length - 3;

                    if (inputDims[fi] > 0 && inputDims[fi] < 10)
                    {
                        // NCHW
                        if (inputDims[fi] < 0)
                            inputDims[fi] = imageSize.z;
                        if (inputDims[fi + 1] < 0)
                            inputDims[fi + 1] = imageSize.y;
                        if (inputDims[fi + 2] < 0)
                            inputDims[fi + 2] = imageSize.x;
                    }
                    else
                    {
                        // NHWC
                        if (inputDims[fi] < 0)
                            inputDims[fi] = imageSize.y;
                        if (inputDims[fi + 1] < 0)
                            inputDims[fi + 1] = imageSize.x;
                        if (inputDims[fi + 2] < 0)
                            inputDims[fi + 2] = imageSize.z;
                    }

                    var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(inputDims);
                    tensor.Fill(0);

                    if (inputData != null)
                    {
                        for (int i = 0; i < inputData.Length; i++)
                            tensor.SetValue(i, inputData[i]);
                    }

                    var onnxValue = NamedOnnxValue.CreateFromTensor(name, tensor);
                    inputTensors.Add(onnxValue);

                    inputNames.Add(name);
                    break;
                }

                // output tensor names
                var outputMeta = session.OutputMetadata;
                var outputNames = new List<string>(outputMeta.Count);

                foreach (var name in outputMeta.Keys)
                {
                    outputNames.Add(name);
                }

                // Run the inference
                // 'results' is an IDisposableReadOnlyCollection<OrtValue> container
                var dtStartTime = DateTime.UtcNow;
                using (var results = session.Run(inputTensors))
                {
                    var dtEndTime = DateTime.UtcNow;
                    var dtTime = (dtEndTime.Ticks - dtStartTime.Ticks) * 0.0000001f;
                    //Debug.Log($"ORT model inference took {Mathf.RoundToInt(dtTime * 1000)} ms");

                    // dump the results
                    foreach (var result in results)
                    {
                        var tensor = result.AsTensor<float>();
                        if (tensor == null)
                            continue;

                        float[] tensorData = tensor.ToArray();
                        outTensors.Add(result.Name, tensorData);

                        float fMin = tensorData.Min();
                        float fMax = tensorData.Max();

                        string shapeStr = string.Join(", ", tensor.Dimensions.ToArray());
                        string dataStr = string.Join(", ", tensorData);

                        Debug.Log($"ORT - {result.Name} - shape: ({shapeStr})\nMin: {fMin}, Max: {fMax}\n{dataStr}");
                    }
                }

            }
        }
        catch (Exception ex)
        {
            Debug.LogError("Error running ORT inferece.");
            Debug.LogException(ex);
        }

        return outTensors;
    }


    // Sentis inference
    public static Dictionary<string, float[]> DoSentisInference(BackendType backendType, string model_resource_name, float[] inputData, bool hasNormOutput,
        out List<TensorShape> outTensorShapes, out List<string> outTensorNames)
    {
        ModelAsset modelAsset = (ModelAsset)Resources.Load(model_resource_name);
        var sentisModel = ModelLoader.Load(modelAsset);

        return DoSentisInference(backendType, sentisModel, null, inputData, hasNormOutput, out outTensorShapes, out outTensorNames);
    }


    public static Dictionary<string, float[]> DoSentisInference(BackendType backendType, Model sentisModel, Model origModel, float[] inputData, bool hasNormOutput,
        out List<TensorShape> outTensorShapes, out List<string> outTensorNames)
    {
        Dictionary<string, float[]> outTensors = new Dictionary<string, float[]>();
        outTensorShapes = new List<TensorShape>();
        outTensorNames = new List<string>();

        if (sentisModel == null)
        {
            Debug.LogError("Model asset is null.");
            return outTensors;
        }

        try
        {
            // create worker
            var worker = new Worker(sentisModel, backendType);

            var inputShape = sentisModel.inputs[0].shape;
            //var tensorShape = inputShape.ToTensorShape();

            var tensorShape = TensorShape.Ones(inputShape.rank);
            for (var i = 0; i < inputShape.rank; i++)
            {
                if (inputShape.Get(i) >= 0)
                    tensorShape[i] = inputShape.Get(i);
            }

            using (var inputTensor = inputData != null ? new Tensor<float>(tensorShape, inputData) : new Tensor<float>(tensorShape))
            {
                //Debug.Log($"Sentis inputTensor: {sentisModel.inputs[0].name} - shape: {inputTensor.shape}");

                var dtStartTime = DateTime.UtcNow;
                worker.Schedule(inputTensor);

                var dtEndTime = DateTime.UtcNow;
                var dtTime = (dtEndTime.Ticks - dtStartTime.Ticks) * 0.0000001f;
                //Debug.Log($"Sentis model inference took {Mathf.RoundToInt(dtTime * 1000)} ms");

                //foreach (var output in sentisModel.outputs)
                for (int o = 0; o < sentisModel.outputs.Count; o++)
                {
                    var output = sentisModel.outputs[o];
                    string outputName = output.name;

                    var outTensorUnk = worker.PeekOutput(outputName);
                    var outTensorF = outTensorUnk as Tensor<float>;

                    string tensorProps = string.Empty;
                    float[] outTensorData = null;

                    if (outTensorF != null)
                    {
                        Tensor<float> localTensor = outTensorF.ReadbackAndClone();
                        outTensorData = localTensor.DownloadToArray();  // new float[outTensor.shape.length];
                        localTensor.Dispose();
                    }
                    else
                    {
                        // try int-tensor
                        var outTensorI = outTensorUnk as Tensor<int>;

                        Tensor<int> localTensor = outTensorI.ReadbackAndClone();
                        int[] intTensorData = localTensor.DownloadToArray();
                        localTensor.Dispose();

                        outTensorData = new float[intTensorData.Length];
                        for (int i = 0; i < intTensorData.Length; i++)
                            outTensorData[i] = intTensorData[i];
                    }

                    if (o < (sentisModel.outputs.Count - 1) || !hasNormOutput)
                    {
                        if (origModel != null && o < origModel.outputs.Count)
                        {
                            outputName = origModel.outputs[o].name;
                        }

                        outTensors.Add(outputName, outTensorData);
                        outTensorShapes.Add(outTensorUnk.shape);
                        outTensorNames.Add(outputName);
                    }

                    float fMin = outTensorData.Min();
                    float fMax = outTensorData.Max();

                    string dataStr = string.Join(", ", outTensorData);
                    Debug.Log($"Sentis - {outputName} - shape: {outTensorUnk.shape} {tensorProps}\nMin: {fMin}, Max: {fMax}:\n{dataStr}");
                }
            }

            worker.Dispose();
        }
        catch (Exception ex)
        {
            Debug.LogError("Error running Sentis inferece.");
            Debug.LogException(ex);
        }

        return outTensors;
    }

    // calculates the mean squared error between two tensors (flatten data arrays)
    public static float GetTensorDifference(float[] t1, float[] t2)
    {
        int minLen = Mathf.Min(t1.Length, t2.Length);

        float diff = 0f;
        for(int i = 0; i < minLen; i++)
        {
            float elemDiff = t1[i] - t2[i];
            diff += elemDiff * elemDiff;  // Mathf.Abs(elemDiff);  // 
        }
        
        if(t1.Length > t2.Length)
        {
            for(int i = minLen; i < t1.Length; i++)
            {
                float elemDiff = t1[i];
                diff += elemDiff * elemDiff;  // Mathf.Abs(elemDiff);  // 
            }
        }
        else if(t2.Length > t1.Length)
        {
            for (int i = minLen; i < t2.Length; i++)
            {
                float elemDiff = t2[i];
                diff += elemDiff * elemDiff;  // Mathf.Abs(elemDiff);  // 
            }
        }

        int maxLen = Mathf.Max(t1.Length, t2.Length);
        diff /= maxLen;

        if (t1.Length != t2.Length)
        {
            Debug.LogWarning($"  T1-len: {t1.Length}, T2-len: {t2.Length}, Diff: {diff}");
        }

        return diff;
    }


    // calculates Sigmoid(x)
    public static float Sigmoid(float x)
    {
        return (float)(1.0 / (1.0 + Math.Exp(-x)));
    }


}
