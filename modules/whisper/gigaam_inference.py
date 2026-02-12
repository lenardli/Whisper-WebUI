"""
GigaAM-v3 (ai-sage/GigaAM-v3) inference for transcription.
Uses Hugging Face Transformers with trust_remote_code; supports file and microphone input.
"""
import os
import time
import tempfile
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import numpy as np
import gradio as gr

from modules.utils.paths import DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR
from modules.whisper.data_classes import Segment, WhisperParams
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.utils.logger import get_logger

logger = get_logger()

GIGAAM_MODEL_ID = "ai-sage/GigaAM-v3"
GIGAAM_REVISION = "e2e_rnnt"


class GigaAMInference(BaseTranscriptionPipeline):
    """Transcription pipeline using ai-sage/GigaAM-v3 (Hugging Face)."""

    def __init__(self,
                 model_dir: str = None,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        # GigaAM doesn't use local model_dir for weights (uses HF cache)
        if model_dir is None:
            model_dir = os.path.join(output_dir, "gigaam_models")
        super().__init__(
            model_dir=model_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir,
            output_dir=output_dir
        )
        self._gigaam_model = None
        self.available_models = [GIGAAM_MODEL_ID]
        self.current_model_size = None
        self.current_compute_type = "float32"

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ) -> Tuple[List[Segment], float]:
        start_time = time.time()
        params = WhisperParams.from_list(list(whisper_params))

        if self._gigaam_model is None or self.current_model_size != params.model_size:
            self.update_model(params.model_size, params.compute_type or "float32", progress)

        progress(0, desc="Loading audio..")
        audio_path, cleanup_temp = self._audio_to_path(audio)
        if audio_path is None:
            return [Segment()], 0.0

        try:
            if hasattr(self._gigaam_model, "transcribe_longform"):
                utterances = self._gigaam_model.transcribe_longform(audio_path)
                segments_result = []
                for i, utt in enumerate(utterances):
                    text = utt.get("transcription", "")
                    boundaries = utt.get("boundaries", (0.0, 0.0))
                    start_s, end_s = boundaries if len(boundaries) >= 2 else (0.0, 0.0)
                    segments_result.append(Segment(
                        id=i,
                        start=start_s,
                        end=end_s,
                        text=text.strip() if text else None
                    ))
                    if progress_callback is not None and len(utterances) > 0:
                        progress_callback((i + 1) / len(utterances))
                    progress((i + 1) / max(len(utterances), 1), desc="Transcribing..")
            else:
                text = self._gigaam_model.transcribe(audio_path)
                if isinstance(text, str) and text.strip():
                    segments_result = [Segment(start=0.0, end=0.0, text=text.strip())]
                else:
                    segments_result = [Segment()]
                if progress_callback is not None:
                    progress_callback(1.0)
                progress(1.0, desc="Transcribing..")
        except Exception as e:
            logger.exception("GigaAM transcription failed: %s", e)
            segments_result = [Segment()]
        finally:
            if cleanup_temp and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        elapsed = time.time() - start_time
        return segments_result, elapsed

    def _audio_to_path(self, audio: Union[str, BinaryIO, np.ndarray]) -> Tuple[Optional[str], bool]:
        """Convert audio input to a file path for GigaAM (it expects path). Returns (path, cleanup_temp)."""
        if isinstance(audio, str) and os.path.isfile(audio):
            return audio, False
        if isinstance(audio, np.ndarray):
            import soundfile as sf
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                sf.write(path, audio, 16000)
                return path, True
            except Exception as e:
                logger.warning("Failed to write temp audio for GigaAM: %s", e)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                return None, False
        return None, False

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()):
        if model_size != GIGAAM_MODEL_ID:
            return
        progress(0, desc="Loading GigaAM-v3..")
        try:
            from transformers import AutoModel
            import torch
            # GigaAM longform/VAD pipeline ожидает HF_TOKEN; ставим безопасное
            # значение по умолчанию, если пользователь явно не задал токен.
            os.environ.setdefault("HF_TOKEN", "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")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._gigaam_model = AutoModel.from_pretrained(
                GIGAAM_MODEL_ID,
                revision=GIGAAM_REVISION,
                trust_remote_code=True,
            ).to(device)
            self._gigaam_model = self._gigaam_model.float()
            self._gigaam_model.eval()
            self.current_model_size = GIGAAM_MODEL_ID
            self.current_compute_type = compute_type
            logger.info("GigaAM-v3 loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load GigaAM-v3: %s", e)
            raise RuntimeError(f"Failed to load GigaAM-v3: {e}") from e

    def offload(self):
        """Offload the model and free memory."""
        if self._gigaam_model is not None:
            del self._gigaam_model
            self._gigaam_model = None
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
