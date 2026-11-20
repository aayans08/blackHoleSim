#version 330

// PBR vertex shader

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;
uniform mat4 p3d_ModelMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

out vec2 texcoord;
out vec3 worldPos;
out vec3 worldNormal;
out vec3 viewPos;

void main() {
    texcoord = p3d_MultiTexCoord0;

    // World space position
    worldPos = (p3d_ModelMatrix * p3d_Vertex).xyz;

    // World space normal
    worldNormal = mat3(p3d_ModelMatrix) * p3d_Normal;

    // View space position
    viewPos = (p3d_ModelViewMatrix * p3d_Vertex).xyz;

    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
