#version 100
precision mediump float;

varying vec4 pos;
varying vec4 mCoord;
varying vec4 lightCoord;
varying vec3 normal;

uniform sampler2D theTexture;
uniform sampler2D shadowmap;

uniform vec3 lightPosition;
uniform vec3 sourceLightPosition;
uniform vec3 viewPosition;

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
     float steepness = dot(lDir, normal);
     
     // View direction
     vec3 vDir = normalize(viewPosition - pos.xyz);
     vec3 sourceDir = normalize(sourceLightPosition - pos.xyz);
     vec3 reflectDirection = reflect(-sourceDir, normal);
     
     float specular = 0.35 * pow(max(dot(vDir, reflectDirection), 0.0), 32.0);
     
     vec3 lCoord = lightCoord.xyz;
     lCoord.z += 0.005;
     lCoord.xyz /= lightCoord.w;
     
     float bias = max(0.05 * (1.0 - steepness), 0.005);
     float visibility = 1.0;
     float closest = texture2D(shadowmap, lCoord.xy).r;
     float current = lCoord.z;
     
     //if (all(equal(lCoord.xy, clamp(lCoord.xy, 0.0, 1.0)))) {
         //visibility = 1.0;
     //}
     //if (current > 1.0) {
         //visibility = 1.0;
     //}
     if (current - bias > closest) {
         visibility = 0.5;
     }
     
     intensity *= clamp(steepness, 0.0, 1.0);
     
     //color.xyz *= (intensity + 0.35) * visibility * vec3(1.0, 1.0, 1.0);
     color.xyz *= (intensity + 0.35 + specular) * visibility * vec3(1.0, 1.0, 1.0);
     
     float z = (gl_FragCoord.z / gl_FragCoord.w);
     float fog = clamp(exp(-fogIntensity * z * z), 0.2, 1.0);
    
     gl_FragColor = mix(fogColor, color, fog);
     //gl_FragColor = vec4(visibility, visibility, visibility, 1.0);
}