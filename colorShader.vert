#version 100
attribute vec4 position;
attribute vec3 color;
attribute vec2 textureCoords;

varying vec3 v_color;
varying vec2 v_texCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	v_color = color;
	v_texCoords = textureCoords;
	gl_Position = vec4(position.xyz, 1.0) * (model * view * projection);
}