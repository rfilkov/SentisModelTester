using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using UnityEngine;


public class ModelTester : MonoBehaviour
{
    [Tooltip("Sentis backend type")]
    public BackendType backendType = BackendType.GPUCompute;

    [Tooltip("Image conversion")]
    public ImageConv inputConversion = ImageConv.Resize;
    public enum ImageConv { Resize, Crop, Letterbox };

    [Tooltip("Reference to the Sentis model asset")]
    public ModelAsset modelAsset = null;

    [Tooltip("Number of frames to span the inference")]
    public int framesToExectute = 1;

    [Tooltip("Index of the output tensor to be displayed as an image")]
    public int outputTensorAsImage = 0;

    [Tooltip("Whether to add layers to normalize the model output or not")]
    public bool normalizeOutput = false;

    [Space]

    [Tooltip("Raw image to display the original image")]
    public UnityEngine.UI.RawImage origImage = null;

    [Tooltip("Raw image to display the converted image")]
    public UnityEngine.UI.RawImage convImage = null;

    [Tooltip("Raw image to display the resultant image")]
    public UnityEngine.UI.RawImage outputImage = null;

    [Tooltip("Text UI to display the current inference message")]
    public UnityEngine.UI.Text infMessageText = null;

    [Tooltip("Text UI to display the output message")]
    public UnityEngine.UI.Text outMessageText = null;


    [Space]

    [Tooltip("Input image size of the model")]
    private Vector3Int inputSize = Vector3Int.zero;

    [Tooltip("Input image layout of the model")]
    private TensorLayout inputLayout = TensorLayout.NCHW;

    [Tooltip("Output message")]
    [TextArea]
    private string outMessage = string.Empty;

    [Tooltip("Total error (all tensors)")]
    private float totalError = 0f;


    // model reference
    private Model _sentisModel = null;
    private Model _origModel = null;

    // image size
    private Vector3Int _imageSize = Vector3Int.zero;

    // converted texture
    private RenderTexture _convTex;
    private Material _letterboxMat;

    // output texture
    private RenderTexture _outTex;

    // canvas
    //private Canvas canvas = null;
    private UnityEngine.UI.CanvasScaler scaler = null;

    void Awake()
    {
        // canvas
        //canvas = FindFirstObjectByType<Canvas>();
        scaler = FindFirstObjectByType<UnityEngine.UI.CanvasScaler>();

        // lbox shader
        Shader lboxTexShader = Shader.Find("Kinect/LetterboxTexShader");
        _letterboxMat = new Material(lboxTexShader);
    }


    void Start()
    {
        if (modelAsset == null)
            return;

        // load the model
        _sentisModel = ModelLoader.Load(modelAsset);

        if(outputTensorAsImage >= 0 && normalizeOutput)
        {
            _origModel = _sentisModel;
            _sentisModel = Functional.Compile(
                    (input1) =>
                    {
                        var modelOutputs = _sentisModel.Forward(input1);
                        var output = modelOutputs[outputTensorAsImage];

                        var max0 = Functional.ReduceMax(output, new[] { -1, -2, -3 });
                        var min0 = Functional.ReduceMin(output, new[] { -1, -2, -3 });

                        var outNorm = (output - min0) / (max0 - min0);
                        var outTensors = Functional.Concat(modelOutputs, 1);

                        return (outTensors, outNorm);
                    },

                    (_sentisModel.inputs[0])
                );
        }

        // init model worker
        _modelWorker = WorkerFactory.CreateWorker(backendType, _sentisModel);  // BackendType.GPUCompute
        modelLayerCount = _sentisModel.layers.Count;

        // estimate input size and tensor layout
        Texture inputImage = WebCamInput.Instance.InputImageTexture;
        _imageSize = inputImage != null ? new Vector3Int(inputImage.width, inputImage.height, 3) : Vector3Int.zero;

        if (inputSize == Vector3Int.zero && _sentisModel.inputs.Count > 0)
        {
            var inShape = _sentisModel.inputs[0].shape;
            if (inShape.rank == 3)
            {
                inputSize = new Vector3Int(inShape[0].isValue ? inShape[0].value : -1, inShape[1].isValue ? inShape[1].value : -1, inShape[2].isValue ? inShape[2].value : -1);
            }
            else if (inShape.rank == 4)
            {
                inputSize = new Vector3Int(inShape[1].isValue ? inShape[1].value : -1, inShape[2].isValue ? inShape[2].value : -1, inShape[3].isValue ? inShape[3].value : -1);
            }

            if (inputSize.x > 0 && inputSize.x < 10)
            {
                inputSize = new Vector3Int(inputSize.z, inputSize.y, inputSize.x);  // move channels last, swap h & w
                inputLayout = TensorLayout.NCHW;
            }
            else
            {
                inputSize = new Vector3Int(inputSize.y, inputSize.x, inputSize.z);  // just swap h & w
                inputLayout = TensorLayout.NHWC;
            }

            if (inputSize.x < 0)
                inputSize.x = _imageSize.y;
            if (inputSize.y < 0)
                inputSize.y = _imageSize.x;
            if (inputSize.z < 0)
                inputSize.z = _imageSize.z;

            //Debug.Log($"Detected inputSize: {inputSize}, layout: {inputLayout}");
        }
    }


    void Update()
    {
        if (_sentisModel == null)
            return;

        // convert input image to required size
        ConvertInputImage();

        // make sentis inference in parts, if needed
        DoSentisInferenceInParts();

        if (Input.GetMouseButton(0))
        {
            // make onnx and sentis inference & compare them
            DoOnnxAndSentisInference();
        }
    }

    private void OnDestroy()
    {
        // release converted texture
        _convTex?.Release();
        _convTex = null;

        _inputTensor?.Dispose();
        _modelWorker?.Dispose();
    }


    // sentis parameters
    private bool _isInferenceStarted = false;
    private IWorker _modelWorker = null;
    private IEnumerator _modelSchedule = null;
    private TensorFloat _inputTensor = null;
    private int modelLayerCount = 0;


    // do one step of sentis inference
    private void DoSentisInferenceInParts()
    {
        if (_convTex == null || framesToExectute <= 0)
            return;

        bool hasMoreWork = false;
        long dtStart = System.DateTime.UtcNow.Ticks;

        if (!_isInferenceStarted)
        {
            var texTrans = new TextureTransform().SetDimensions(inputSize.x, inputSize.y, inputSize.z).SetTensorLayout(inputLayout);

            if (_inputTensor == null)
                _inputTensor = TextureConverter.ToTensor(_convTex, texTrans);
            else
                TextureConverter.ToTensor(_convTex, _inputTensor, texTrans);

            _modelSchedule = _modelWorker.ExecuteLayerByLayer(_inputTensor);
            hasMoreWork = _modelSchedule.MoveNext();
            _isInferenceStarted = true;
        }

        int layersToRun = (modelLayerCount + framesToExectute - 1) / framesToExectute; // round up
        int runLayers = 0;

        for (int i = 0; i < layersToRun; i++)
        {
            hasMoreWork = _modelSchedule.MoveNext();
            runLayers++;

            if (!hasMoreWork)
                break;
        }

        long dtEnd = System.DateTime.UtcNow.Ticks;
        //Debug.Log($"Executed {runLayers} layers in {(dtEnd - dtStart) * 0.0001f} ms. moreWork: {hasMoreWork}");

        if(infMessageText != null)
        {
            infMessageText.text = $"Executed {runLayers} layers in {(dtEnd - dtStart) * 0.0001f:F3} ms";
        }

        if (!hasMoreWork)
        {
            _isInferenceStarted = false;

            if (outputTensorAsImage >= 0)  // && outputTensorAsImage < _sentisModel.outputs.Count)
            {
                string outputName = _sentisModel.outputs[_sentisModel.outputs.Count - 1].name;
                TensorFloat output = _modelWorker.PeekOutput(outputName) as TensorFloat;

                if(output.shape.rank < 4)
                {
                    output.Reshape(output.shape.Unsqueeze(0));
                }

                // create output texture as needed
                CreateOutputTexture(output.shape);

                //TextureTransform texTransform = new TextureTransform().SetCoordOrigin(CoordOrigin.BottomLeft);
                TextureConverter.RenderToTexture(output, _outTex);  //, texTransform);
            }
        }
    }


    // do inference on onnx-runtime and sentis, and compare them
    private void DoOnnxAndSentisInference()
    {
        float[] tensorData = null;

        // get input tensor data
        var texTrans = new TextureTransform().SetDimensions(inputSize.x, inputSize.y, inputSize.z).SetTensorLayout(inputLayout);
        using (var tensor = TextureConverter.ToTensor(_convTex, texTrans))
        {
            var localTensor = tensor.ReadbackAndClone();
            tensorData = tensor.ToReadOnlyArray();
            localTensor.Dispose();
        }

        // do ORT inference
        string modelPath = UnityEditor.AssetDatabase.GetAssetPath(modelAsset);
        Dictionary<string, float[]> ortOutData = ModelTools.DoOrtInference(modelPath, tensorData, _imageSize);  // onnxModelDir + "/" + modelAsset.name + ".onnx"

        // do Sentis inference
        bool hasNormOutput = outputTensorAsImage >= 0 && normalizeOutput;
        Dictionary<string, float[]> sentisOutData = ModelTools.DoSentisInference(backendType, _sentisModel, _origModel, tensorData, hasNormOutput,
            out List<TensorShape> outTensorShapes, out List<string> outTensorNames);

        // check the outputs
        if (ortOutData.Count != sentisOutData.Count)
        {
            outMessage = $"ORT produced {ortOutData.Count} tensors, while Sentis produced {sentisOutData.Count} tensors.";
        }
        else
        {
            string sMessage = string.Empty;
            totalError = 0f;

            for (int i = 0; i < ortOutData.Count; i++)
            {
                string tensorName = outTensorNames[i];
                float diff = ModelTools.GetTensorDifference(ortOutData[tensorName], sentisOutData[tensorName]);

                TensorShape shape = outTensorShapes[i];
                sMessage += $"T{i} - {tensorName} - {shape} - difference: {diff}\n";
                totalError += diff;
            }

            outMessage = sMessage;
            //outError = totalError;

            if(outMessageText != null)
            {
                sMessage += $"\nTotal error: {totalError}\n";
                outMessageText.text = sMessage;
            }

            //Debug.Log($"Total error: {totalError}");
        }
    }


    // returns the maximum screen size of an image, according to the screen resolution and image aspect ratio
    private Vector2 GetMaxImageScreenSize(Texture tex, float resFactor = 0.45f)
    {
        Vector2 scrHalfRes = scaler.referenceResolution * resFactor;  // new Vector2(Screen.width, Screen.height) * resFactor;

        float imgW = 0f;
        float imgH = 0f;

        if (tex.width >= tex.height)
        {
            // landscape
            imgW = scrHalfRes.x;
            imgH = imgW * tex.height / tex.width;

            if(imgH > scrHalfRes.y)
            {
                imgH = scrHalfRes.y;
                imgW = imgH * tex.width / tex.height;
            }
        }
        else
        {
            // portrait
            imgH = scrHalfRes.y;
            imgW = imgH * tex.width / tex.height;

            if(imgW > scrHalfRes.x)
            {
                imgW = scrHalfRes.x;
                imgH = imgW * tex.height / tex.width;
            }
        }

        Vector2 imgSize = new Vector2(imgW, imgH);
        //Debug.Log($"  tex: {tex.width} x {tex.height}, scr: {Screen.width} x {Screen.height}, refRes: {scaler.referenceResolution:F0}\nhalfRes: {scrHalfRes:F0}, tex: {tex.width} x {tex.height}, img: {imgSize:F0}");

        return imgSize;
    }


    // converts the input image to input texture of required size
    private void ConvertInputImage()
    {
        Texture inputImage = WebCamInput.Instance.InputImageTexture;
        if (inputImage == null)
            return;

        if (origImage != null && inputImage != null)
        {
            origImage.texture = inputImage;
            origImage.rectTransform.sizeDelta = GetMaxImageScreenSize(inputImage);
            //Debug.Log($"origImageSize: {origImage.rectTransform.sizeDelta:F2}");
        }

        if (_convTex == null || _convTex.width != inputSize.x || _convTex.height != inputSize.y)
        {
            _convTex = Utils.CreateRenderTexture(_convTex, inputSize.x, inputSize.y, RenderTextureFormat.ARGB32);
        }

        if (convImage != null && _convTex != null)
        {
            convImage.texture = _convTex;
            convImage.rectTransform.sizeDelta = GetMaxImageScreenSize(_convTex);
            //Debug.Log($"convImageSize: {convImage.rectTransform.sizeDelta:F2}");
        }

        switch (inputConversion)
        {
            case ImageConv.Resize:
                Graphics.Blit(inputImage, _convTex);
                break;

            case ImageConv.Crop:
                Utils.TransformTexture(inputImage, _convTex);
                break;

            case ImageConv.Letterbox:
                _letterboxMat.SetInt("_isLinearColorSpace", QualitySettings.activeColorSpace == ColorSpace.Linear ? 1 : 0);
                _letterboxMat.SetInt("_letterboxWidth", inputSize.x);
                _letterboxMat.SetInt("_letterboxHeight", inputSize.y);

                var scale = new Vector2(Mathf.Max((float)inputImage.height / inputImage.width, 1), Mathf.Max(1, (float)inputImage.width / inputImage.height));
                _letterboxMat.SetVector("_spadScale", scale);

                Graphics.Blit(inputImage, _convTex, _letterboxMat);
                break;
        }
    }


    // create output texture as needed
    private void CreateOutputTexture(TensorShape tensorShape)
    {
        Vector3Int tSize = Vector3Int.zero;
        //TensorLayout tLayout = TensorLayout.NCHW;

        if (tensorShape.rank == 3)
        {
            tSize = new Vector3Int(tensorShape[0], tensorShape[1], tensorShape[2]);
        }
        else if (tensorShape.rank == 4)
        {
            tSize = new Vector3Int(tensorShape[1], tensorShape[2], tensorShape[3]);
        }

        if (tSize.x < 10)
        {
            tSize = new Vector3Int(tSize.z, tSize.y, tSize.x);  // move channels last, swap h & w
            //tLayout = TensorLayout.NCHW;
        }
        else
        {
            tSize = new Vector3Int(tSize.y, tSize.x, tSize.z);  // just swap h & w
            //tLayout = TensorLayout.NHWC;
        }

        // Debug.Log($"Detected output-tex size: {tSize}, layout: {tLayout}");

        if (tSize.x > 20 && tSize.x < 2000 && tSize.y > 20 && tSize.y < 2000 && tSize.z > 0 && tSize.z < 5)
        {
            if (_outTex == null || _outTex.width != tSize.x || _outTex.height != tSize.y)
            {
                _outTex = Utils.CreateRenderTexture(_outTex, tSize.x, tSize.y, tSize.z > 1 ? RenderTextureFormat.ARGB32 : RenderTextureFormat.RFloat);
            }

            if (outputImage != null && _outTex != null)
            {
                outputImage.texture = _outTex;
                outputImage.rectTransform.sizeDelta = GetMaxImageScreenSize(_outTex, 0.6f);
                //Debug.Log($"outImageSize: {outputImage.rectTransform.sizeDelta:F2}");
            }
        }
    }


}
