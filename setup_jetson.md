- 환경 복사

```bash
git clone https://github.com/NaCl-1374/inspire_hand_ws.git
```

- Initialize and update submodules:

```bash
cd inspire_hand_ws
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
python3 -m pip install --upgrade pip; python3 -m pip install numpy==1.22.1; python3 -m pip install onnx; python3 -m pip install --no-cache $TORCH_INSTALL
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
