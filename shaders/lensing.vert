#version 330

// Gravitational lensing vertex shader

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

out vec2 texcoord;
out vec3 worldPos;

void main() {
    texcoord = p3d_MultiTexCoord0;
    worldPos = p3d_Vertex.xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
