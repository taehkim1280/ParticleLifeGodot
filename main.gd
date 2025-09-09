extends Node2D


@export_category("Particle Params")
@export_range(0, 10000) var NUM_PARTICLES := 3000

@export_range(0, 10000) var max_force := 5000.0

@export_range(0, 50) var attraction_radius := 25.0
@export_range(0, 50) var attraction_force := 10.0
@export_range(0, 30) var collision_edge_outer := 30.0
@export_range(0, 2) var collision_neg_exp := 0.8
@export_range(0, 4000) var collision_force := 3100.0

@export_range(0, 2000) var rule_radius := 1500.0
@export_range(0, 10) var rule_force := 1.0
@export_range(0, 50) var rule_force_falloff := 1.0

@export_range(0, 5000) var global_friction := 500.0
@export_range(0.5, 1) var air_friction := 0.97
@export_range(0, 50) var collision_ratio := 1.0


@export_range(2, 20) var NUM_COLORS := 5
class Particle_Rule:
	var color: Color
	var forces: Array[float]

# negative values means attraction
var particle_rules_a := [
	[-80.0,	-10.0, 	-10.0, 	-3.0, 	-3.0], 	# red reacts to _ in this way..
	[-10.0,	-80.0, 	-10.0, 	-3.0, 	-3.0], 	# green
	[-10.0,	-10.0, 	-20.0, 	-3.0, 	-3.0], 	# blue
	[-03.0, 	-03.0, 	-03.0, 	-10.0, 	-03.0], 	# indigo
	[-3.0, 	-3.0, 	-3.0, 	-3.0, 	-40.0] 	# cyan
]
var particle_rules_b := [
	[-40.0,	30.0, 	-10.0, 	-3.0, 	-3.0], 	# red reacts to _ in this way..
	[-30.0,	-30.0, 	-10.0, 	-3.0, 	-3.0], 	# green
	[-10.0,	40.0, 	-20.0, 	-3.0, 	-3.0], 	# blue
	[33.0, 	-03.0, 	-03.0, 	-10.0, 	-03.0], 	# indigo
	[-3.0, 	-3.0, 	-3.0, 	23.0, 	-5.0] 	# cyan
]
var particle_rules_c := [
	[-40.0,	30.0, 	0.0, 	-0.0, 	-0.0], 	# red reacts to _ in this way..
	[-20.0,	-40.0, 	30.0, 	-0.0, 	-0.0], 	# green
	[00.0,	-20.0, 	-40.0, 	30.0, 	-3.0], 	# blue
	[0.0, 	-00.0, 	-20.0, 	-40.0, 	30.0], 	# indigo
	[0.0, 	-0.0, 	-0.0, 	-20.0, 	-40.0] 	# cyan
]

var particle_colors := [
	Color.RED, 
	Color.GREEN, 
	Color.BLUE, 
	Color.INDIGO,
	Color.CYAN
]

var particle_pos : Array[Vector2] = []
var particle_last_pos : Array[Vector2] = []
var particle_wrapped_displacement : Array[Vector2] = []
var particle_type : Array[int] = [] # index of rule in particle_rules

var IMAGE_SIZE := int(ceil(sqrt(NUM_PARTICLES)))
var particle_data : Image
var particle_data_texture : ImageTexture
var particle_colors_image : Image
var particle_colors_texture : ImageTexture

# GPU Variables
var rd : RenderingDevice
var particle_compute_shader : RID
var particle_pipeline : RID
var bindings : Array
var uniform_set : RID

var particle_pos_buffer : RID
var particle_last_pos_buffer : RID
var particle_wrapped_displacement_buffer : RID
var particle_rule_buffer : RID
var params_buffer: RID
var particle_rules_buffer : RID
var params_uniform : RDUniform
var particle_data_buffer : RID

func _generate_particles():
	for i in NUM_PARTICLES:
		particle_pos.append(Vector2(randf() * get_viewport_rect().size.x, randf() * get_viewport_rect().size.y))
		particle_type.append(randi_range(0, 1))
	particle_last_pos = particle_pos.duplicate(true)
	particle_wrapped_displacement = particle_pos.duplicate(true)

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	_generate_particles()
	print(NUM_COLORS)
	print(NUM_PARTICLES)
	print(IMAGE_SIZE)
	print(int(ceil(sqrt(NUM_PARTICLES))))
	particle_data = Image.create(IMAGE_SIZE, IMAGE_SIZE, false, Image.FORMAT_RGBAH)
	particle_data_texture = ImageTexture.create_from_image(particle_data)
	particle_colors_image = Image.create(NUM_COLORS, 1, false, Image.FORMAT_RGBAH)
	particle_colors_texture = ImageTexture.create_from_image(particle_colors_image)
	for i in NUM_PARTICLES:
		var pixel_pos = Vector2(int(float(i) / IMAGE_SIZE), int(i % IMAGE_SIZE))
		particle_data.set_pixel(pixel_pos.x, pixel_pos.y, Color(particle_pos[i].x, particle_pos[i].y, float(particle_type[i]), 0))
	for i in NUM_COLORS:
		particle_colors_image.set_pixel(i, 0, particle_colors[i])
	particle_data_texture.update(particle_data)
	particle_colors_texture.update(particle_colors_image)


	$ParticleSim.amount = NUM_PARTICLES
	$ParticleSim.process_material.set_shader_parameter("particle_data", particle_data_texture)
	$ParticleSim.process_material.set_shader_parameter("particle_colors_texture", particle_colors_texture)

	_setup_compute_shader()
	_update_particles_gpu(0.001)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	get_window().title = "Workgroups: " + str(ceil(NUM_PARTICLES/1024.)) + " / Particles: " + str(NUM_PARTICLES) + " / FPS: " + str(Engine.get_frames_per_second())

	_sync_particles_gpu()

	# //////////////////////////////////////////////////////////////////////////////////////
	# //////////////     GET DATA FROM GPU TO SEND TO PARTICLE SHADER      //////////////////
	# //////////////////////////////////////////////////////////////////////////////////////
	var particle_data_image_data := rd.texture_get_data(particle_data_buffer, 0)
	particle_data.set_data(IMAGE_SIZE, IMAGE_SIZE, false, Image.FORMAT_RGBAH, particle_data_image_data)
	particle_data_texture.update(particle_data)

	_update_particles_gpu(delta)

# //////////////////////////////////////////////////////////////////////
# /////////////////////     GPU FUNCTIONS    /////////////////////////
# //////////////////////////////////////////////////////////////////////

func _setup_compute_shader():

	# //////////////////////////////////////////////////////////
	# //////////////////     SETUP      ///////////////////////
	# //////////////////////////////////////////////////////////

	rd = RenderingServer.create_local_rendering_device()

	# load shader file
	var shader_file := load("res://particle_sim_gpu.glsl")
	var shader_spirv: RDShaderSPIRV = shader_file.get_spirv()
	particle_compute_shader = rd.shader_create_from_spirv(shader_spirv)

	# set member with pipeline
	particle_pipeline = rd.compute_pipeline_create(particle_compute_shader)
	
	# //////////////////////////////////////////////////////////
	# //////////////////     INPUTS      ///////////////////////
	# //////////////////////////////////////////////////////////

	# pos buffer
	particle_pos_buffer = _generate_vec2_buffer(particle_pos)
	var particle_pos_uniform = _generate_uniform(particle_pos_buffer, RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 0)
	# last pos buffer
	particle_last_pos_buffer = _generate_vec2_buffer(particle_last_pos)
	var particle_last_pos_uniform = _generate_uniform(particle_last_pos_buffer, RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 1)
	# wrap displacement buffer
	particle_wrapped_displacement_buffer = _generate_vec2_buffer(particle_wrapped_displacement)
	var particle_wrapped_displacement_uniform = _generate_uniform(particle_wrapped_displacement_buffer, RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 2)
	# rule buffer
	particle_rule_buffer = _generate_int_buffer(particle_type)
	var particle_type_uniform = _generate_uniform(particle_rule_buffer, RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 3)
	# params buffer
	params_buffer = _generate_parameter_buffer(0.001)
	params_uniform = _generate_uniform(params_buffer, RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 4)
	# rules buffer
	var particle_rules_concat : Array[float] = []
	for i in range(NUM_COLORS):
		for j in range(NUM_COLORS):
			particle_rules_concat.append(particle_rules_c[i][j])
	particle_rules_buffer = _generate_float_buffer(particle_rules_concat)
	var particle_rules_uniform = _generate_uniform(particle_rules_buffer, RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 5)

	######## particle_data buffer (compute shader output) ########
	# sets the texture format
	var fmt := RDTextureFormat.new()
	fmt.width = IMAGE_SIZE
	fmt.height = IMAGE_SIZE
	fmt.format = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.usage_bits = (RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT 
		| RenderingDevice.TEXTURE_USAGE_STORAGE_BIT
		| RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT)
	# uses texture format to make texture containing particle_data image
	var view := RDTextureView.new()
	particle_data_buffer = rd.texture_create(fmt, view, [particle_data.get_data()])
	var particle_data_buffer_uniform = _generate_uniform(particle_data_buffer, RenderingDevice.UNIFORM_TYPE_IMAGE, 6)


	# set bindings for uniform set
	bindings = [
		particle_pos_uniform, 
		particle_last_pos_uniform,
		particle_wrapped_displacement_uniform,
		particle_type_uniform,
		params_uniform,
		particle_rules_uniform,
		particle_data_buffer_uniform
	]
	uniform_set = rd.uniform_set_create(bindings, particle_compute_shader, 0)

func _update_particles_gpu(delta : float):
	rd.free_rid(params_buffer)
	params_buffer = _generate_parameter_buffer(delta)
	params_uniform.clear_ids()
	params_uniform.add_id(params_buffer)
	uniform_set = rd.uniform_set_create(bindings, particle_compute_shader, 0)

	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, particle_pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)

	# //////////////////////////////////////////////////////////
	# //////////////////     EXECUTE      ///////////////////////
	# //////////////////////////////////////////////////////////

	rd.compute_list_dispatch(compute_list, ceil(NUM_PARTICLES/1024.), 1, 1)
	rd.compute_list_end()
	rd.submit()

func _sync_particles_gpu():
	rd.sync()

func _exit_tree():
	_sync_particles_gpu()
	rd.free_rid(uniform_set)
	rd.free_rid(particle_data_buffer)
	rd.free_rid(params_buffer)
	rd.free_rid(particle_pos_buffer)
	rd.free_rid(particle_last_pos_buffer)
	rd.free_rid(particle_wrapped_displacement_buffer)
	rd.free_rid(particle_rule_buffer)
	rd.free_rid(particle_rules_buffer)
	rd.free_rid(particle_pipeline)
	rd.free_rid(particle_compute_shader)

	rd.free()



# //////////////////////////////////////////////////////////////////////
# /////////////////////     HELPER FUNCTIONS    /////////////////////////
# //////////////////////////////////////////////////////////////////////

func _generate_vec2_buffer(data):
	var data_buffer_bytes := PackedVector2Array(data).to_byte_array()
	var data_buffer = rd.storage_buffer_create(data_buffer_bytes.size(), data_buffer_bytes)
	return data_buffer

func _generate_int_buffer(data : Array[int]):
	var data_buffer_bytes = PackedInt32Array(data).to_byte_array()
	var data_buffer = rd.storage_buffer_create(data_buffer_bytes.size(), data_buffer_bytes)
	return data_buffer

func _generate_float_buffer(data : Array[float]):
	var data_buffer_bytes = PackedFloat32Array(data).to_byte_array()
	var data_buffer = rd.storage_buffer_create(data_buffer_bytes.size(), data_buffer_bytes)
	return data_buffer
	
func _generate_uniform(data_buffer, type, binding):
	var data_uniform = RDUniform.new()
	data_uniform.uniform_type = type
	data_uniform.binding = binding
	data_uniform.add_id(data_buffer)
	return data_uniform


func _generate_parameter_buffer(delta: float):
	var params_buffer_bytes : PackedByteArray = PackedFloat32Array([
		float(NUM_PARTICLES),
		float(NUM_COLORS),
		float(IMAGE_SIZE), 

		max_force,

		attraction_radius,
		attraction_force,
		collision_edge_outer,
		collision_neg_exp,
		collision_force,

		rule_radius,
		rule_force,

		global_friction,
		air_friction,
		collision_ratio,

		float(get_viewport_rect().size.x),
		float(get_viewport_rect().size.y),
		delta
	]).to_byte_array()
	
	return rd.storage_buffer_create(params_buffer_bytes.size(), params_buffer_bytes)
