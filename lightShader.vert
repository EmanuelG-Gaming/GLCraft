#version 100
attribute vec4 position;

varying vec4 v_mCoord;

uniform mat4 model;
uniform mat4 u_lightView;
uniform mat4 u_lightProjection;

void main()
{
	v_mCoord = position;
	gl_Position = vec4(position.xyz, 1.0) * (model * u_lightView * u_lightProjection);
	//gl_Position = (u_lightProjection * u_lightView * model) * vec4(position.xyz, 1.0);
}