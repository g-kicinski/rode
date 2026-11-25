from networkx import config
import yaml
import argparse

def load_config():
    # CLI overrides
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    # Load YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))

    return config

