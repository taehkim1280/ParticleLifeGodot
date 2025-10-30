#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

// A binding to the buffer we create in our script
layout(set = 0, binding = 0, std430) restrict buffer Position {
    vec2 data[];
} particle_pos;

layout(set = 0, binding = 1, std430) restrict buffer LastPosition {
    vec2 data[];
} particle_last_pos;

layout(set = 0, binding = 2, std430) restrict buffer WrapDisplacement {
    vec2 data[];
} wrap_displacement;

// rule assignment for each particle
layout(set = 0, binding = 3, std430) restrict buffer Rule {
    int data[];
} particle_rule;

layout(set = 0, binding = 4, std430) restrict buffer Params {
    float num_particles;
    float num_rules;
    float image_size;

    float max_force;

    float attraction_radius;
    float attraction_force;

    float collision_edge_outer;
    float collision_edge_inner;
    float collision_neg_exp;
    float collision_force;

    float rule_radius;
    float rule_radius_inner;
    float rule_force;
    float rule_force_exp;

    float global_friction;
    float air_friction;
    float collision_ratio;
    
    float viewport_x;
    float viewport_y;
    float delta_time;
} params;

layout(set = 0, binding = 5, std430) restrict buffer Rules {
    float data[];
} particle_rules;

// output buffer
layout(rgba16f, binding = 6) uniform image2D particle_data;


void main() {
    int my_index = int(gl_GlobalInvocationID.x);
    int num_particles = int(params.num_particles);

    if (my_index >= num_particles ) return;

    vec2 cur_pos = particle_pos.data[my_index];
    vec2 last_pos = particle_last_pos.data[my_index];
    int my_rule = particle_rule.data[my_index];

    ivec2 quadrant = ivec2(int(cur_pos.x > params.viewport_x/2.), int(cur_pos.y > params.viewport_y/2.));

    // save last pos
    particle_last_pos.data[my_index] = cur_pos;

    vec2 total_force = vec2(0.0,0.0);

    //////////// LOOP THROUGH ALL PARTICLES ////////////
    
    for (int i = 0; i < num_particles; i++) {
        if (i != my_index) {
            vec2 other_pos_m = particle_pos.data[i];

            //////////// WRAPPING FROM ALL 8 SIDES ////////////
            vec2 w = vec2(params.viewport_x, 0.);
            vec2 h = vec2(0., params.viewport_y);

            vec2 other_pos_tl = other_pos_m + h - w;
            vec2 other_pos_t = other_pos_m + h;
            vec2 other_pos_tr = other_pos_m + h + w;

            vec2 other_pos_l = other_pos_m - w;
            vec2 other_pos_r = other_pos_m + w;

            vec2 other_pos_bl = other_pos_m - h - w;
            vec2 other_pos_b = other_pos_m - h;
            vec2 other_pos_br = other_pos_m - h + w;

            vec2 other_pos_full[9];
            other_pos_full = vec2[9](
                other_pos_m,
                other_pos_tl,
                other_pos_t ,
                other_pos_tr,
                other_pos_l ,
                other_pos_r ,
                other_pos_bl,
                other_pos_b ,
                other_pos_br
            );
            vec2 other_poss[4];
            {
                if (quadrant == ivec2(0, 0)) { // top left
                    other_poss = vec2[4](
                        other_pos_tl,
                        other_pos_t,
                        other_pos_l,
                        other_pos_m
                    );
                }
                else if (quadrant == ivec2(1, 0)) { // top right
                    other_poss = vec2[4](
                        other_pos_t,
                        other_pos_tr,
                        other_pos_m,
                        other_pos_r
                    );
                }
                else if (quadrant == ivec2(0, 1)) { // bottom left
                    other_poss = vec2[4](
                        other_pos_l,
                        other_pos_m,
                        other_pos_bl,
                        other_pos_b
                    );
                }
                else { // bottom right
                    other_poss = vec2[4](
                        other_pos_m,
                        other_pos_r,
                        other_pos_b,
                        other_pos_br
                    );
                }
            }

            for (int j = 0; j < 9; j++) {
                vec2 other_pos = other_pos_full[j];

                vec2 other_v = cur_pos - other_pos;
                vec2 other_n = normalize(other_v);
                float dist = max(0.01, length(other_v));


                //////////// FORCES FROM RULESET ////////////
                if (dist < params.rule_radius) {
                    // neg: attraction
                    // pos: repulsion
                    float other_to_my_force = 0.0;

                    int other_rule = particle_rule.data[i];
                    other_to_my_force += particle_rules.data[int(params.num_rules)*my_rule + other_rule];

                    float dist_coefficient = (params.rule_radius-dist)/params.rule_radius;
                    float dist_coefficient_max = (params.rule_radius-params.rule_radius_inner)/params.rule_radius;

                    other_to_my_force *= (1000.0/params.num_particles)*params.rule_force*pow(min(dist_coefficient_max, dist_coefficient), params.rule_force_exp);

                    total_force += other_n*other_to_my_force;
                }

                //////////// GLOBAL ATTRACTION ////////////
                // if (dist < params.attraction_radius){
                //     total_force += normalize(other_pos - cur_pos)*(2.0*params.attraction_force)/(dist*dist);
                // }

                //////////// COLLISION REPULSION ////////////
                
                if (dist < params.collision_edge_outer) {
                    // This force gets very strong as dist -> 0

                    float repulsionmax = params.collision_force * (params.collision_edge_outer - params.collision_edge_inner) / pow(params.collision_edge_inner, params.collision_neg_exp);
                    float repulsion = params.collision_force * (params.collision_edge_outer - dist) / pow(dist, params.collision_neg_exp);
                    total_force += other_n * min(repulsionmax, repulsion);
                }

                // float total_force_m = length(total_force);
                // total_force_m = min(params.max_force, total_force_m);
                // total_force = normalize(total_force)*total_force_m;
                // weaken force if more particles
                //total_force /= params.num_particles/params.global_friction;
                //break;
            }
        }
    }

    //////////// INTERGRATE ////////////
    // verlet integration
    vec2 cur_displacement = (cur_pos - (last_pos + wrap_displacement.data[my_index]))*params.air_friction + total_force*params.delta_time*params.delta_time;
    cur_pos += cur_displacement;

    // Wrap around screen
    vec2 wrapped_pos = vec2(mod(cur_pos.x, params.viewport_x), mod(cur_pos.y, params.viewport_y));
    // store wrap displacement
    wrap_displacement.data[my_index] = wrapped_pos - cur_pos;
    cur_pos = wrapped_pos;
    
    // save cur_pos n+1
    particle_pos.data[my_index] = cur_pos;

    // store to buffer for particle shader to render
    ivec2 pixel_pos = ivec2(int(mod(float(my_index), params.image_size)), int(float(my_index)/params.image_size));
    imageStore(particle_data, pixel_pos, vec4(cur_pos.x, cur_pos.y, float(my_rule), 0.0));
}