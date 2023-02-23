#version 100
precision mediump float;

varying vec3 v_position;
varying vec2 v_textureCoords;
uniform sampler2D theTexture;

void main()
{
     vec2 flipped = vec2(v_textureCoords.x, -v_textureCoords.y);
     vec4 color = texture2D(theTexture, flipped);
     //vec4 color = vec4(1.0, 1.0, 0.0, 1.0);
     if (color.a < 0.4) {
         discard;
     }
     
     gl_FragColor = color;
}