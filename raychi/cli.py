import argparse
from .renderer import render

def main():
    parser = argparse.ArgumentParser(description="RayChi: A Taichi-based Ray Tracer")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--disable-ao", action='store_false', dest='enable_ao')
    parser.add_argument("--disable-direct-lighting", action='store_false', dest='enable_direct_lighting')
    parser.add_argument("--rr-prob", type=float, default=0.8)
    parser.add_argument("--max-ray-pool", type=int, default=8000000)
    parser.add_argument("--max-ambient-requests", type=int, default=1000000)
    # New arguments for light position and config file
    parser.add_argument("--light-pos", nargs=3, type=float, help="Light position as X Y Z")
    parser.add_argument("--light-color", nargs=3, type=float, help="Light color as R G B")
    parser.add_argument("--ambient-color", nargs=3, type=float, help="Ambienti color as R G B")
    parser.add_argument("--cam-origin", nargs=3, type=float, help="Camera origin as X Y Z")
    parser.add_argument("--lookat", nargs=3, type=float, help="Lookat target as X Y Z")
    parser.add_argument("--vup", nargs=3, type=float, help="View-up vector as X Y Z")
    parser.add_argument("--theta", type=float, help="Vertical FOV in degrees")
    parser.add_argument("--objects-config", type=str, help="Path to separate objects config file")
    parser.add_argument("--config", type=str, default='raychi_config.json', 
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    render(
        image_width=args.width,
        image_height=args.height,
        ENABLE_AO=args.enable_ao,
        ENABLE_DIRECT_LIGHTING=args.enable_direct_lighting,
        RR_prob=args.rr_prob,
        max_ray_pool=args.max_ray_pool,
        max_ambient_requests=args.max_ambient_requests,
        light_pos=args.light_pos,  # Pass light_pos from CLI
        config_file=args.config,    # Pass config file path
        cam_origin=[0.0, 1.0, -5.0],
        lookat=[0.0, 1.0, 0.0],
        ambient_color=[0.2, 0.2, 0.4],
        light_color=[10.0, 10.0, 10.0]
)

if __name__ == "__main__":
    main()
