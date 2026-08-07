import os
import glob
import re

files = glob.glob("src/**/*.py", recursive=True)

for file in files:
    with open(file, "r") as f:
        content = f.read()

    modified = False

    # Replace standard project_root definitions
    if (
        'project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))\n'
        in content
    ):
        content = content.replace(
            'project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))\n', ""
        )
        modified = True

    if 'output_dir = os.path.join(project_root, "output")\n' in content:
        content = content.replace(
            'output_dir = os.path.join(project_root, "output")\n', 'output_dir = "output"\n'
        )
        modified = True

    # src/demo/01_bio_blade_engine.py
    if (
        'FILE_CONTROL = os.path.join(project_root, "data", "ephys", "Drug_2953_control.raw.h5")'
        in content
    ):
        content = content.replace(
            'os.path.join(project_root, "data", "ephys", "Drug_2953_control.raw.h5")',
            '"data/ephys/Drug_2953_control.raw.h5"',
        )
        content = content.replace(
            'os.path.join(project_root, "data", "ephys", "Drug_2953_50uM.raw.h5")',
            '"data/ephys/Drug_2953_50uM.raw.h5"',
        )
        modified = True

    if (
        'output_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output")), "01_bio_blade_engine.png")'
        in content
    ):
        content = content.replace(
            'os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output")), "01_bio_blade_engine.png")',
            '"output/01_bio_blade_engine.png"',
        )
        modified = True

    # src/demo/raw/2_hssm_standing_wave.py
    if (
        "workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
        in content
    ):
        content = content.replace(
            "workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n",
            "",
        )
        content = content.replace(
            "os.path.join(workspace_root, 'output', '2_hssm_standing_wave.png')",
            '"output/2_hssm_standing_wave.png"',
        )
        modified = True

    # src/demo/raw/8_ephys_demo.py
    if "current_dir = os.path.dirname(os.path.abspath(__file__))\n" in content:
        content = content.replace("current_dir = os.path.dirname(os.path.abspath(__file__))\n", "")
        content = content.replace(
            'repo_root = os.path.abspath(os.path.join(current_dir, "../../.."))\n', ""
        )
        content = content.replace(
            'os.path.join(repo_root, "output", "raw", "8_ephys_demo.png")',
            '"output/raw/8_ephys_demo.png"',
        )
        content = content.replace(
            'os.path.join(repo_root, "data", "ephys", "example.brw")', '"data/ephys/example.brw"'
        )
        modified = True

    # src/metrics/spd_interpreter.py
    if 'default_config = os.path.join(project_root, "configs", "spd_mamba_config.yaml")' in content:
        content = content.replace(
            'os.path.join(project_root, "configs", "spd_mamba_config.yaml")',
            '"configs/spd_mamba_config.yaml"',
        )
        modified = True

    # src/pipeline/ephys/brw_dataloader.py
    if 'repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))\n' in content:
        content = content.replace(
            'repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))\n', ""
        )
        content = content.replace(
            'default_dir = os.path.join(repo_root, "data", "ephys")\n',
            'default_dir = "data/ephys"\n',
        )
        modified = True

    if modified:
        with open(file, "w") as f:
            f.write(content)
        print(f"Refactored {file}")
