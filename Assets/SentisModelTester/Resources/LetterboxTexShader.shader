Shader "Kinect/LetterboxTexShader"
{
    Properties
	{
		_MainTex("_MainTex", 2D) = "white" {}
	}

    SubShader
    {
		ZTest Always Cull Off ZWrite Off
		Fog { Mode off }

		Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;

            uint _isLinearColorSpace;
            uint _letterboxWidth;
            uint _letterboxHeight;
            float2 _spadScale;


            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 pos : SV_POSITION;
            };

         
            v2f vert (appdata v)
            {
                v2f o;

                o.uv = (v.uv - 0.5) * _spadScale + 0.5;  // v.uv;  // 
				o.pos = UnityObjectToClipPos(v.vertex);
             
                return o;
            }

            float4 frag (v2f i) : SV_Target
            {
                float2 uv = i.uv; 

                float2 duv_dx = float2(1.0 / _letterboxWidth * _spadScale.x, 0);
                float2 duv_dy = float2(0, 1.0 / _letterboxHeight * _spadScale.y);

                float3 rgb = tex2Dgrad(_MainTex, uv, duv_dx, duv_dy).rgb;
                rgb *= all(uv > 0) && all(uv < 1);

                if(_isLinearColorSpace) 
		            rgb = LinearToGammaSpace(rgb);

				return float4(rgb, 1);
            }

            ENDCG
        }
    }
}
