"""
Composite transcription pipeline: GigaAM-v3 + Faster-Whisper in one interface.
Dispatches by model_size so GigaAM is selectable from the same Model dropdown.
"""
from typing import List, Union, Optional, Callable, Tuple
import numpy as np
import gradio as gr

from modules.whisper.data_classes import TranscriptionPipelineParams, WhisperParams, Segment
from modules.whisper.gigaam_inference import GigaAMInference, GIGAAM_MODEL_ID
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline


class CompositeTranscriptionPipeline(BaseTranscriptionPipeline):
    """Pipeline that uses GigaAM-v3 or Faster-Whisper based on selected model."""

    def __init__(self, faster_whisper_inf: BaseTranscriptionPipeline, gigaam_inf: GigaAMInference):
        self.faster_whisper_inf = faster_whisper_inf
        self.gigaam_inf = gigaam_inf
        # GigaAM first so it is the default in the dropdown
        fw_models = list(faster_whisper_inf.available_models)
        if GIGAAM_MODEL_ID not in fw_models:
            self.available_models = [GIGAAM_MODEL_ID] + fw_models
        else:
            self.available_models = fw_models
        self.available_langs = faster_whisper_inf.available_langs
        self.device = faster_whisper_inf.device
        self.diarizer = faster_whisper_inf.diarizer
        self.music_separator = faster_whisper_inf.music_separator
        self.vad = faster_whisper_inf.vad
        self.model_dir = faster_whisper_inf.model_dir
        self.output_dir = faster_whisper_inf.output_dir
        self.available_compute_types = faster_whisper_inf.available_compute_types
        self.current_compute_type = getattr(faster_whisper_inf, "current_compute_type", "float16")

    def _use_gigaam(self, pipeline_params: list) -> bool:
        params = TranscriptionPipelineParams.from_list(pipeline_params)
        return params.whisper.model_size == GIGAAM_MODEL_ID

    def transcribe(self,
                   audio: Union[str, bytes, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ) -> Tuple[List[Segment], float]:
        """Delegate to GigaAM or Faster-Whisper based on model_size in whisper_params."""
        params = WhisperParams.from_list(list(whisper_params))
        if params.model_size == GIGAAM_MODEL_ID:
            return self.gigaam_inf.transcribe(
                audio, progress, progress_callback, *whisper_params
            )
        return self.faster_whisper_inf.transcribe(
            audio, progress, progress_callback, *whisper_params
        )

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()):
        """Delegate to the backend that owns the given model."""
        if model_size == GIGAAM_MODEL_ID:
            self.gigaam_inf.update_model(model_size, compute_type, progress)
        else:
            self.faster_whisper_inf.update_model(model_size, compute_type, progress)

    def run(self,
            audio: Union[str, bytes, list],
            progress: gr.Progress = gr.Progress(),
            file_format: str = "SRT",
            add_timestamp: bool = True,
            progress_callback: Optional[Callable] = None,
            *pipeline_params,
            ):
        pipeline_list = list(pipeline_params)
        if self._use_gigaam(pipeline_list):
            return self.gigaam_inf.run(
                audio, progress, file_format, add_timestamp, progress_callback, *pipeline_params
            )
        return self.faster_whisper_inf.run(
            audio, progress, file_format, add_timestamp, progress_callback, *pipeline_params
        )

    def transcribe_file(self,
                       files=None,
                       input_folder_path=None,
                       include_subdirectory=None,
                       save_same_dir=None,
                       file_format: str = "SRT",
                       add_timestamp: bool = True,
                       progress=gr.Progress(),
                       *pipeline_params,
                       ):
        pipeline_list = list(pipeline_params)
        if self._use_gigaam(pipeline_list):
            return self.gigaam_inf.transcribe_file(
                files, input_folder_path, include_subdirectory, save_same_dir,
                file_format, add_timestamp, progress, *pipeline_params
            )
        return self.faster_whisper_inf.transcribe_file(
            files, input_folder_path, include_subdirectory, save_same_dir,
            file_format, add_timestamp, progress, *pipeline_params
        )

    def transcribe_mic(self,
                      mic_audio: str,
                      file_format: str = "SRT",
                      add_timestamp: bool = True,
                      progress=gr.Progress(),
                      *pipeline_params,
                      ):
        pipeline_list = list(pipeline_params)
        if self._use_gigaam(pipeline_list):
            return self.gigaam_inf.transcribe_mic(
                mic_audio, file_format, add_timestamp, progress, *pipeline_params
            )
        return self.faster_whisper_inf.transcribe_mic(
            mic_audio, file_format, add_timestamp, progress, *pipeline_params
        )

    def transcribe_youtube(self,
                           youtube_link: str,
                           file_format: str = "SRT",
                           add_timestamp: bool = True,
                           progress=gr.Progress(),
                           *pipeline_params,
                           ):
        pipeline_list = list(pipeline_params)
        if self._use_gigaam(pipeline_list):
            return self.gigaam_inf.transcribe_youtube(
                youtube_link, file_format, add_timestamp, progress, *pipeline_params
            )
        return self.faster_whisper_inf.transcribe_youtube(
            youtube_link, file_format, add_timestamp, progress, *pipeline_params
        )
