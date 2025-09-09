shader_type particles;

uniform sampler2D particle_data;
uniform sampler2D particle_colors_texture;
uniform float scale = .08;


void process() {
	ivec2 pos_image_size = textureSize(particle_data, 0);
	ivec2 texel_pos = ivec2(
		int(mod(float(INDEX), float(pos_image_size.x))), 
		int(float(INDEX)/float(pos_image_size.x))
	);

	vec4 particle_pos = texelFetch(particle_data, texel_pos, 0);

	int color_index = int(particle_pos.b);
	vec4 particle_col = texelFetch(particle_colors_texture, ivec2(color_index, 0), 0);

	mat2 scale_mat = mat2(
		vec2(1, 0),
		vec2(0, 1)
	);

	scale_mat *= scale;

	// Rotation and scale
	TRANSFORM[0].xy = scale_mat[0];
	TRANSFORM[1].xy = scale_mat[1];
	// Translation
	TRANSFORM[3].xy = vec2(particle_pos.r, particle_pos.g);

	COLOR.rgb = particle_col.rgb*255.;
}
