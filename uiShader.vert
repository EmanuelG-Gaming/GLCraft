#version 100
attribute vec3 a_pos;
attribute vec3 a_colors;
attribute vec2 a_tCoords;

varying vec3 v_position;
varying vec3 v_color;
varying vec2 v_textureCoords;

uniform mat4 u_model;

void main()
{
     v_position = a_pos;
     v_color = a_colors;
     v_textureCoords = a_tCoords;
     
     gl_Position = vec4(a_pos.xy, 0.0, 1.0) * u_model;
}