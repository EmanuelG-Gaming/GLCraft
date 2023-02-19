#version 100
attribute vec4 position;
attribute vec2 textureCoords;
attribute vec3 color;

varying vec4 pos;		
varying vec4 mCoord;
varying vec2 texCoord;
varying vec3 normal;
 	
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
      
void main()
{
     mCoord = position;
     texCoord = textureCoords;
     if (mCoord.w >= 80.0) normal = vec3(0.0, 0.0, 1.0);
     else if (mCoord.w >= 64.0) normal = vec3(0.0, 0.0, -1.0);
     else if (mCoord.w >= 48.0) normal = vec3(0.0, 1.0, 0.0);
     else if (mCoord.w >= 32.0) normal = vec3(0.0, -1.0, 0.0);
     else if (mCoord.w >= 16.0) normal = vec3(1.0, 0.0, 0.0);
     else normal = vec3(-1.0, 0.0, 0.0);
     
     pos = vec4(position.xyz, 1.0) * (model);			
     gl_Position = vec4(position.xyz, 1.0) * (model * view * projection);
}