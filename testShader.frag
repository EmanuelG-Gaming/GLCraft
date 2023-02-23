#version 100
precision mediump float;

varying vec4 pos;
varying vec4 mCoord;
varying vec3 normal;

uniform sampler2D theTexture;
uniform vec3 lightPosition;

const vec4 fogColor = vec4(0.47, 0.57, 0.97, 1.0);
const float fogIntensity = 0.0003;


void main()
{
     float intensity = 0.9;
     float atlasWidth = 16.0;
     
     vec2 texCoord;
     // If normal is vertical
     if (normal.y != 0.0) {
         texCoord = vec2((fract(mCoord.x) + mCoord.w) / atlasWidth, mCoord.z);
     } else {
         texCoord = vec2((fract(mCoord.x + mCoord.z) + mCoord.w) / atlasWidth, -mCoord.y);
     }
     
     vec4 color = texture2D(theTexture, texCoord);
     if (color.a < 0.4) {
         discard;
     }
     
     // Light direction
     vec3 lDir = normalize(lightPosition - pos.xyz);
     intensity *= clamp(dot(normal, lDir), 0.0, 1.0);
     
     color.xyz *= (intensity + 0.35) * vec3(1.0, 1.0, 1.0);
     
     float z = (gl_FragCoord.z / gl_FragCoord.w);
     float fog = clamp(exp(-fogIntensity * z * z), 0.2, 1.0);
    
     gl_FragColor = mix(fogColor, color, fog);
}