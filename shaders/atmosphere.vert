#version 330

// Atmospheric scattering vertex shader
// Based on Sean O'Neil's GPU Gems 2 implementation

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

out vec2 texcoord;
out vec3 v_position;
out vec3 v_normal;
out vec3 v_view_dir;

void main() {
    // Pass texture coordinates
    texcoord = p3d_MultiTexCoord0;

    // Transform position and normal
    vec4 view_pos = p3d_ModelViewMatrix * p3d_Vertex;
    v_position = view_pos.xyz;
    v_normal = normalize(p3d_NormalMatrix * p3d_Normal);
    v_view_dir = normalize(-view_pos.xyz);

    // Output position
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
