using UnityEngine;


/// <summary>
/// WebCamInput controls the source device (webcam, video or texture), and manages the source texture.
/// </summary>
public class WebCamInput : MonoBehaviour
{

    [Tooltip("Static image or video render texture. If left empty, a webcam is used as source device.")]
    public Texture staticInput;

    [Tooltip("Webcam device name, if a specific webcam is requested. If empty, the default webcam is selected.")]
    public string webCamName;

    [Tooltip("Requested webcam resolution. Uses the default webcam resolution, if set to (0, 0).")]
    public Vector2 webCamResolution = new Vector2(1920, 1080);


    // singleton instance of the class
    public static WebCamInput Instance = null;


    private WebCamTexture webCamTexture;
    private RenderTexture inputRT;


    // Get input image texture.
    public Texture InputImageTexture
    {
        get
        {
            if(staticInput != null)
            {
                return staticInput;
            }

            return inputRT;
        }
    }


    void Awake()
    {
        Instance = this;
    }


    void Start()
    {
        if(staticInput == null)
        {
            var webcamDevices = WebCamTexture.devices;
            int deviceCount = webcamDevices.Length;
            System.Text.StringBuilder sbWebcams = new System.Text.StringBuilder();

            for (int i = 0; i < deviceCount; i++)
            {
                var device = webcamDevices[i];
                sbWebcams.Append(device.name).AppendLine();
            }

            Debug.Log($"Available Webcams:\n{sbWebcams}");

            // open the selected webcam
            webCamTexture = new WebCamTexture(webCamName, (int)webCamResolution.x, (int)webCamResolution.y);
            webCamTexture.Play();
        }
    }


    void Update()
    {
        if (staticInput != null)
            return;
        if(!webCamTexture.didUpdateThisFrame)
            return;

        if(inputRT == null || inputRT.width != webCamTexture.width || inputRT.height != webCamTexture.height)
        {
            inputRT = Utils.CreateRenderTexture(inputRT, webCamTexture.width, webCamTexture.height, 0);
        }

        float aspect1 = (float)webCamTexture.width / webCamTexture.height;
        float aspect2 = (float)inputRT.width / inputRT.height;
        float aspectGap = aspect2 / aspect1;

        bool vMirrored = webCamTexture.videoVerticallyMirrored;
        Vector2 scale = new Vector2(aspectGap, vMirrored ? -1 : 1);
        Vector2 offset = new Vector2((1 - aspectGap) / 2, vMirrored ? 1 : 0);

        Graphics.Blit(webCamTexture, inputRT, scale, offset);
    }


    void OnDestroy()
    {
        if (webCamTexture != null)
            Destroy(webCamTexture);
        if (inputRT != null)
            Destroy(inputRT);
    }
}
