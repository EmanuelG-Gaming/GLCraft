#version 100
attribute vec4 a_pos;

varying vec4 v_position;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

void main()
{
     v_position = a_pos;
     
     gl_Position = vec4(a_pos.xyz, 1.0) * (u_model * u_view * u_projection);
}