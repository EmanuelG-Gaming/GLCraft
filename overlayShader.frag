#version 100
precision mediump float;

uniform float u_time;

varying vec4 v_position;

void main()
{
     float fTime = u_time * 4.0;
     float d = (sin(fTime) + 1.0) * 0.5 + 0.1;
     vec4 color = vec4(1.0, 1.0, 1.0, 1.0);
     color.xyz *= d;
     
     gl_FragColor = color;
}