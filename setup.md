- 환경 복사

```bash
git clone https://github.com/NaCl-1374/inspire_hand_ws.git
```

- inspire_hand_ws 진입, venv_x86 압축 해제 후, 코드 실행

```bash
python -m venv venv  # or  Unzip venv_x86.tar.xz, and place the.venv in inspire_hand_ws/.venv

# Then execute the script to modify venv:
python update_venv_path.py .venv
python update_bin_files.py .venv 

source venv/bin/activate  # Activate the virtual environment for Linux/MacOS
```

- Initialize and update submodules:

```bash
git submodule init  # Initialize submodules
git submodule update  # Update submodules to the latest version
```

- Install the two SDKs:

```bash
cd unitree_sdk2_python
pip install -e .

cd ../inspire_hand_sdk
pip install -e .
```

- install pytorch

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

- unitree_rl_gym 설치 (safetics 브랜치)

```bash
cd unitree_rl_gym
pip install -e .
```

- 이후 deploy

```bash
python deploy/deploy_real/deploy_real.py {net_interface(ex:enp3s0)} g1.yaml
```
