"""
Compose a CALVIN env render config (.hydra/merged_config.yaml) from calvin_env's
own conf, so HulcWrapper.get_env can create the PyBullet sim env WITHOUT the
official dataset's .hydra/merged_config.yaml (the VyoJ subset doesn't ship it).

It reproduces the standard CALVIN recording config: env=play_table_env,
scene=calvin_scene_D, robot=panda_longer_finger, cameras=static_and_gripper.

用法 / Usage:
    python scripts/make_calvin_merged_config.py \
      --out_dirs /root/autodl-tmp/CALVIN-datasets/calvin_vyoj/training \
                 /root/autodl-tmp/CALVIN-datasets/calvin_vyoj/validation
"""
import argparse
from pathlib import Path

import calvin_env
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dirs", nargs="+", default=[
        "/root/autodl-tmp/CALVIN-datasets/calvin_vyoj/training",
        "/root/autodl-tmp/CALVIN-datasets/calvin_vyoj/validation",
    ])
    ap.add_argument("--cameras", default="static_and_gripper")
    ap.add_argument("--scene", default="calvin_scene_D")
    args = ap.parse_args()

    conf_dir = str(Path(calvin_env.__file__).parents[1] / "conf")
    print("calvin_env conf dir:", conf_dir)

    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config_data_collection",
            overrides=[f"cameras={args.cameras}", f"scene={args.scene}"],
        )

    # get_env() reads render_conf.env / render_conf.cameras / render_conf.scene
    assert "env" in cfg and "cameras" in cfg and "scene" in cfg, "composed cfg missing keys"

    for d in args.out_dirs:
        hd = Path(d) / ".hydra"
        hd.mkdir(parents=True, exist_ok=True)
        out = hd / "merged_config.yaml"
        OmegaConf.save(cfg, out)
        print("wrote", out)
    print("done.")


if __name__ == "__main__":
    main()
