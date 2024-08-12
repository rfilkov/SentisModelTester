using System.Collections;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;


/// <summary>
/// Utility methods.
/// </summary>
public static class Utils
{
    // creates new render texture with the given dimensions and format
    public static RenderTexture CreateRenderTexture(RenderTexture currentTex, int width, int height, RenderTextureFormat texFormat = RenderTextureFormat.Default)
    {
        if (currentTex != null)
        {
            currentTex.Release();
        }

        RenderTexture renderTex = new RenderTexture(width, height, 0, texFormat);
        renderTex.wrapMode = TextureWrapMode.Clamp;
        renderTex.filterMode = FilterMode.Point;
        renderTex.enableRandomWrite = true;

        return renderTex;
    }


    // transforms the source texture to target texture, by applying the needed resize, rotation and flip
    public static void TransformTexture(Texture srcTex, RenderTexture tgtTex, int rotationAngle = 0, bool flipX = false, bool flipY = false)
    {
        if (srcTex == null || tgtTex == null)
            return;

        int srcWidth = srcTex.width;
        int srcHeight = srcTex.height;

        bool isPortraitMode = rotationAngle == 90 || rotationAngle == 270;

        if (isPortraitMode)
        {
            // swap width and height
            (srcWidth, srcHeight) = (srcHeight, srcWidth);
        }

        float camAspect = (float)srcWidth / srcHeight;
        float tgtAspect = (float)tgtTex.width / tgtTex.height;

        //// fix bug on Android
        //if (Application.platform == RuntimePlatform.Android)
        //{
        //    rotationAngle = -rotationAngle;
        //}

        Matrix4x4 transMat = Matrix4x4.identity;
        Vector4 texST = Vector4.zero;

        if (isPortraitMode)
        {
            transMat = GetTransformMat(rotationAngle, flipY, flipX);
            texST = GetTexST(tgtAspect, camAspect);
        }
        else
        {
            transMat = GetTransformMat(rotationAngle, flipX, flipY);
            texST = GetTexST(camAspect, tgtAspect);
        }

        if (_textureMat == null)
        {
            _textureMat = new Material(Shader.Find("Kinect/TransformTexShader"));
        }

        // blit the texture
        if (_textureMat != null)
        {
            _textureMat.SetMatrix(_TransformMatParam, transMat);
            _textureMat.SetVector(_TexSTParam, texST);

            Graphics.Blit(srcTex, tgtTex, _textureMat, 0);
        }
    }


    private static Material _textureMat = null;
    private static readonly int _TransformMatParam = Shader.PropertyToID("_TransformMat");
    private static readonly int _TexSTParam = Shader.PropertyToID("_TexST");

    // returns the camera transform matrix
    private static Matrix4x4 GetTransformMat(float rotation, bool mirrorHorizontal, bool mirrorVertical)
    {
        Vector3 scale = new Vector3(mirrorHorizontal ? -1f : 1f, mirrorVertical ? -1f : 1f, 1f);
        Matrix4x4 mat = Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(0, 0, rotation), scale);

        return PUSH_MATRIX * mat * POP_MATRIX;
    }

    private static readonly Matrix4x4 PUSH_MATRIX = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
    private static readonly Matrix4x4 POP_MATRIX = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));

    // returns the camera texture offsets
    private static Vector4 GetTexST(float srcAspect, float dstAspect)
    {
        if (srcAspect > dstAspect)
        {
            float s = dstAspect / srcAspect;
            return new Vector4(s, 1, (1 - s) / 2, 0);
        }
        else
        {
            float s = srcAspect / dstAspect;
            return new Vector4(1, s, 0, (1 - s) / 2);
        }
    }


    // destroys the object of given type
    public static void Destroy(Object o)
    {
        if (o == null)
            return;

        if(o is RenderTexture)
        {
            ((RenderTexture)o).Release();
        }

        if (Application.isPlaying)
            Object.Destroy(o);
        else
            Object.DestroyImmediate(o);
    }

}

