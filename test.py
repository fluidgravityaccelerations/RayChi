import raychi

# Render an image using the default configuration
image = raychi.render(
    light_pos=[0.0, 5.4, -1.0],
    ambient_color=[0.2, 0.2, 0.4],
    light_color=[10.0, 10.0, 10.0],
    image_width=640,
    image_height=360,
    ENABLE_AO=True,
    ENABLE_DIRECT_LIGHTING=True,
    RR_prob=0.80,
    max_ray_pool=8000000,
    max_ambient_requests=1000000,
    samples_per_pixel=256,
    max_depth=8,
    num_AO_samples=32,
    max_AO_distance=2.0,
    vup=[0.0, 1.0, 0.0],
    theta=60.0
)
