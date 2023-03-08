#version 100
precision mediump float;

varying vec3 v_color;
varying vec2 v_texCoords;

uniform sampler2D theTexture;

void main() {
    vec2 flipped = vec2(v_texCoords.x, -v_texCoords.y);
    vec4 color = texture2D(theTexture, flipped) * vec4(v_color.xyz, 1.0);
    if (color.a < 0.4) {
        discard;
    }
    gl_FragColor = color;
}