#version 100
precision mediump float;

varying vec4 v_mCoord;
uniform sampler2D theTexture;

void main()
{
	 float atlasWidth = 16.0;
	 vec2 coord;
	 
	 if (v_mCoord.w >= 32.0 && v_mCoord.w < 64.0) {
	     coord = vec2((fract(v_mCoord.x) + v_mCoord.w) / atlasWidth, v_mCoord.z);
	 } else {
	     coord = vec2((fract(v_mCoord.x + v_mCoord.z) + v_mCoord.w) / atlasWidth, -v_mCoord.y);
	 }
	 
	 vec4 color = texture2D(theTexture, coord);
	 
	 if (color.a < 0.4) {
	     discard;
	 }
	 //gl_FragColor = color;
}