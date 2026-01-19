import tensorrt as trt
from pathlib import Path
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR

def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_size_bytes: int = 1 << 30,  # 1GB
):
    logger = trt.Logger(trt.Logger.WARNING)
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    # EXPLICIT_BATCH 네트워크 생성
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(logger) as builder, \
         builder.create_network(flags=flags) as network, \
         trt.OnnxParser(network, logger) as parser:

        # 1) ONNX 파싱
        with open(onnx_path, "rb") as f:
            model_bytes = f.read()
        if not parser.parse(model_bytes):
            print("[TensorRT] ONNX parsing failed")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

        # 2) BuilderConfig 생성
        config = builder.create_builder_config()
        # 워크스페이스 메모리 한도 설정 (TRT10 방식)
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_size_bytes
        )

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # (동적 shape가 있으면 여기서 optimization profile 추가)

        # 3) build_serialized_network 사용 (TRT10)
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build serialized TensorRT engine")

        # 4) 바로 파일로 저장
        engine_path.write_bytes(bytes(serialized_engine))
        print(f"[TensorRT] Saved engine to: {engine_path}")

class TrtPolicyRunner:
    def __init__(self, engine_path: str, num_layers: int, hidden_size: int):
        self.logger = trt.Logger(trt.Logger.WARNING)

        # 1) 엔진 로드
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        # 2) 실행 컨텍스트
        self.context = self.engine.create_execution_context()

        # 3) I/O 텐서 이름 모으기
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # 기대: ['obs','h_in','c_in'], ['actions','h_out','c_out']
        required_inputs = {"obs", "h_in", "c_in"}
        required_outputs = {"actions", "h_out", "c_out"}
        if set(self.input_names) != required_inputs or set(self.output_names) != required_outputs:
            raise RuntimeError(
                f"Unexpected IO tensors: inputs={self.input_names}, outputs={self.output_names}"
            )

        # 이름을 고정 순서로 보관
        self.obs_name = "obs"
        self.h_in_name = "h_in"
        self.c_in_name = "c_in"
        self.actions_name = "actions"
        self.h_out_name = "h_out"
        self.c_out_name = "c_out"

        # 4) 입력 shape 설정
        #    obs: (1, 96) 고정, h/c: (num_layers, 1, hidden_size) 고정이라고 가정
        self.obs_shape = (1, 96)
        self.h_obs = cuda.pagelocked_empty(
            shape=self.obs_shape, dtype=np.float32
        )
        self.h_shape = (num_layers, 1, hidden_size)
        self.c_shape = (num_layers, 1, hidden_size)

        self.context.set_input_shape(self.obs_name, self.obs_shape)
        self.context.set_input_shape(self.h_in_name, self.h_shape)
        self.context.set_input_shape(self.c_in_name, self.c_shape)

        # 실제 런타임 shape 확인
        self.obs_shape = tuple(self.context.get_tensor_shape(self.obs_name))
        self.h_shape = tuple(self.context.get_tensor_shape(self.h_in_name))
        self.c_shape = tuple(self.context.get_tensor_shape(self.c_in_name))

        self.actions_shape = tuple(self.context.get_tensor_shape(self.actions_name))
        self.h_out_shape = tuple(self.context.get_tensor_shape(self.h_out_name))
        self.c_out_shape = tuple(self.context.get_tensor_shape(self.c_out_name))

        # 5) 크기 계산
        self.obs_size = int(np.prod(self.obs_shape))
        self.h_size = int(np.prod(self.h_shape))
        self.c_size = int(np.prod(self.c_shape))
        self.actions_size = int(np.prod(self.actions_shape))
        self.h_out_size = int(np.prod(self.h_out_shape))
        self.c_out_size = int(np.prod(self.c_out_shape))

        # 6) GPU 메모리 할당
        self.in_total_size = self.obs_size + self.h_size + self.c_size
        self.out_total_size = self.actions_size + self.h_out_size + self.c_out_size

        self.d_in = cuda.mem_alloc(self.in_total_size * np.float32().nbytes)
        self.d_out = cuda.mem_alloc(self.out_total_size * np.float32().nbytes)

        # 개별 텐서의 디바이스 주소 계산 (offset)
        float_bytes = np.float32().nbytes
        obs_offset = 0
        h_offset = obs_offset + self.obs_size * float_bytes
        c_offset = h_offset + self.h_size * float_bytes

        actions_offset = 0
        h_out_offset = actions_offset + self.actions_size * float_bytes
        c_out_offset = h_out_offset + self.h_out_size * float_bytes

        self.d_obs    = int(self.d_in)  + obs_offset
        self.d_h_in   = int(self.d_in)  + h_offset
        self.d_c_in   = int(self.d_in)  + c_offset
        self.d_actions = int(self.d_out) + actions_offset
        self.d_h_out   = int(self.d_out) + h_out_offset
        self.d_c_out   = int(self.d_out) + c_out_offset

        # 7) 호스트 버퍼: in/out 통합 버퍼 + view
        self.h_in_flat = cuda.pagelocked_empty(
            shape=(self.in_total_size,), dtype=np.float32
        )
        self.h_out_flat = cuda.pagelocked_empty(
            shape=(self.out_total_size,), dtype=np.float32
        )

        # numpy view로 개별 텐서 모양 매핑
        offset = 0
        self.h_obs = self.h_in_flat[offset:offset+self.obs_size].reshape(self.obs_shape)
        offset += self.obs_size
        self.h_h = self.h_in_flat[offset:offset+self.h_size].reshape(self.h_shape)
        offset += self.h_size
        self.h_c = self.h_in_flat[offset:offset+self.c_size].reshape(self.c_shape)

        offset = 0
        self.h_actions = self.h_out_flat[offset:offset+self.actions_size].reshape(self.actions_shape)
        offset += self.actions_size
        self.h_h_out = self.h_out_flat[offset:offset+self.h_out_size].reshape(self.h_out_shape)
        offset += self.h_out_size
        self.h_c_out = self.h_out_flat[offset:offset+self.c_out_size].reshape(self.c_out_shape)

        self.h_h.fill(0.0)
        self.h_c.fill(0.0)

        # 8) 텐서 주소 바인딩
        self.context.set_tensor_address(self.obs_name, self.d_obs)
        self.context.set_tensor_address(self.h_in_name, self.d_h_in)
        self.context.set_tensor_address(self.c_in_name, self.d_c_in)

        self.context.set_tensor_address(self.actions_name, self.d_actions)
        self.context.set_tensor_address(self.h_out_name, self.d_h_out)
        self.context.set_tensor_address(self.c_out_name, self.d_c_out)

        # CUDA 스트림
        self.stream = cuda.Stream()

    def reset(self):
        """에피소드 시작 시 hidden/cell state 리셋."""
        self.h_h.fill(0.0)
        self.h_c.fill(0.0)

    def infer(self, obs_np: np.ndarray) -> np.ndarray:
        """
        obs_np: shape (96,) 또는 (1,96) float32.
        내부적으로 h_h, h_c를 유지하면서 매 스텝 업데이트.
        """
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(self.obs_shape)
        elif obs_np.shape != self.obs_shape:
            raise ValueError(f"Expected obs shape {self.obs_shape}, got {obs_np.shape}")
        np.copyto(self.h_obs, obs_np.astype(np.float32, copy=False))

        # H2D 한 번
        cuda.memcpy_htod_async(self.d_in, self.h_in_flat, self.stream)

        # 실행
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # D2H 한 번
        cuda.memcpy_dtoh_async(self.h_out_flat, self.d_out, self.stream)
        self.stream.synchronize()

        # hidden state 업데이트 (view이므로 copy 불필요)
        np.copyto(self.h_h, self.h_h_out)
        np.copyto(self.h_c, self.h_c_out)

        return np.asarray(self.h_actions).squeeze()
    
    def __del__(self):
        """pycuda로 할당한 GPU 리소스 정리."""
        try:
            # 스트림
            if hasattr(self, "stream") and self.stream is not None:
                self.stream.synchronize()
                self.stream = None

            # 디바이스 메모리들
            for name in [
                "d_obs", "d_h_in", "d_c_in",
                "d_actions", "d_h_out", "d_c_out",
            ]:
                buf = getattr(self, name, None)
                if buf is not None:
                    buf.free()
                    setattr(self, name, None)
            # 스트림 삭제
            if hasattr(self, "stream"):
                del self.stream
            # TensorRT 객체들 (Python 객체만 끊어 주면 됨)
            if hasattr(self, "context"):
                self.context = None
            if hasattr(self, "engine"):
                self.engine = None

        except Exception:
            # 소멸자에서 예외가 올라가지 않게 방어
            pass

if __name__ == "__main__":
    onnx_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/policy/g1/exported/policies/policy.onnx"
    engine_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/policy/g1/exported/policies/policy.engine"
    build_engine_from_onnx(onnx_path, engine_path, fp16=True)