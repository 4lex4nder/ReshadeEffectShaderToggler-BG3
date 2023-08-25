#pragma once

#include "ReShade.fxh"

namespace BG3 {
    //uniform float4x4 matProj < source = "mat_Proj"; >;
    uniform float4x4 matProjInv < source = "mat_InvProj"; >;
    uniform float3 camPos < source = "vec_CameraWorldPos"; >;
	uniform float z_far < source = "var_z_far"; >;
	uniform float z_near < source = "var_z_near"; >;

    texture NormalMapTex : NORMALS;
    sampler NormalMap { Texture = NormalMapTex; MinFilter=POINT; MipFilter=POINT; MagFilter=POINT; };
	
	texture texLambert : LAMBERT;
	sampler sLambert { Texture = texLambert; };

	texture texEmission : EMISSION;
	sampler sEmission { Texture = texEmission; };

    // https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
    float2 _octWrap( float2 v )
    {
        //return ( 1.0 - abs( v.yx ) ) * ( v.xy >= 0.0 ? 1.0 : -1.0 );
        return float2((1.0 - abs( v.y ) ) * ( v.x >= 0.0 ? 1.0 : -1.0),
            (1.0 - abs( v.x ) ) * ( v.y >= 0.0 ? 1.0 : -1.0));
    }
    
    float2 _encode( float3 n )
    {
        n /= ( abs( n.x ) + abs( n.y ) + abs( n.z ) );
        n.xy = n.z >= 0.0 ? n.xy : _octWrap( n.xy );
        n.xy = n.xy * 0.5 + 0.5;
        return n.xy;
    }
    
    float3 _decode( float2 f )
    {
        f = f * 2.0 - 1.0;
        float3 n = float3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
        float t = saturate( -n.z );
        n.xy += n.xy >= 0.0 ? -t : t;
        return normalize( n );
    }
	
	float3 get_radiance(float2 uv)
	{
		float3 color = tex2D(sLambert, uv).rgb;
		float3 emission = tex2D(sEmission, uv).rgb;
	
		return (color + emission);
	}
	
	float4x4 fwd_projection()
	{
		float4x4 ret;
		ret[0][0] = 1.0/matProjInv[0][0];
		ret[1][1] = 1.0/matProjInv[1][1];
		ret[2][3] = 1.0/z_near-1.0/z_far;
		//let's ignore the jitter for now
		//ret[3][0] = -ret[0][0] * matProjInv[2][0];
		//ret[3][1] = -ret[1][1] * matProjInv[2][1];
		ret[3][2] = 1.0;
		ret[3][3] = 1.0/z_far;
		
		return ret;
	}

    float3 get_normal(float2 texcoord)
    {
        float2 normal = tex2D(NormalMap, texcoord).xy;
		return (_decode(normal) + 1.0) / 2.0;
    }
    
    float linearize_depth(float depth)
    {
		float depthLinearizeMul = matProjInv[3][2];
		float depthLinearizeAdd = matProjInv[2][2];
		
		return (depthLinearizeAdd * depth + 1.0) / (depthLinearizeMul * depth * z_far);
    }
	
	float linearize_depth(float depth, float2 coords)
    {
		return linearize_depth(depth);
    }
	
    float3 get_position_from_uv(float2 uv, float depth)
    {
		float4x4 bb = matProjInv;
		//mask jitter
		bb[2][0] = 0;
		bb[2][1] = 0;
	
        float4 pos = (float4(uv.x, uv.y, depth, 1) * float4(2, 2, 1, 1)) - float4(1, 1, 0, 0);
        float4 res = mul(bb, pos);
        res /= res.w;
        
        return res.xyz;
    }
	
	float3 get_position_from_uv_viewdepth(float2 uv, float viewdepth)
    {
		float2 CameraTanHalfFOV = float2(matProjInv[0][0], matProjInv[1][1]);
        float2 NDCToViewMul = float2(CameraTanHalfFOV.x * 2.0, CameraTanHalfFOV.y * 2.0);
		float2 NDCToViewAdd = float2(CameraTanHalfFOV.x * -1.0, CameraTanHalfFOV.y * -1.0);
		
		float3 ret;
		ret.xy = (NDCToViewMul * uv + NDCToViewAdd) * viewdepth;
		ret.z = viewdepth;
		return ret;
    }
	
    float3 get_uv_from_position(float3 pos)
    {
        float4 uv_pos = mul(fwd_projection(), float4(pos, 1));
        uv_pos /= uv_pos.w;
        uv_pos.xyz = uv_pos.xyz * float3(0.5, 0.5, 1) + float3(0.5, 0.5, 0);
        return uv_pos.xyz;
    }
}