
"""pyttsx3 TTS adapter.



pyttsx3 wraps the operating system's native TTS engine -- SAPI5 on

Windows, espeak/espeak-ng on Linux, NSSpeechSynthesizer on macOS. No model

download needed, which fits this project's local-first philosophy and

lets a TTS benchmark run out of the box on any machine with zero setup.

"""

from __future__ import annotations



from pathlib import Path

from typing import Any



from voice_benchmark.adapters.base import TTSModel

from voice_benchmark.core.exceptions import InferenceError, ModelLoadError

from voice_benchmark.utils.audio import (

    AudioLoadError,

    compute_silence_ratio,

    get_audio_duration_sec,

    load_audio,

)

from voice_benchmark.utils.timing import Timer, compute_rtf





class Pyttsx3Adapter(TTSModel):

    """Adapter for pyttsx3 (OS-native TTS engine).



    Known pyttsx3 quirk: reusing one engine instance across multiple

    save_to_file() calls in the same process is unreliable -- later calls

    can silently produce an empty (header-only) wav file. The documented

    workaround is to create a fresh engine per synthesis call, which is

    what synthesize() does below; load() only validates the import and

    that init() succeeds at all.

    """



    def __init__(self, rate: int = 175, voice_id: str | None = None) -> None:

        self.name = "pyttsx3"

        self.rate = rate

        self.voice_id = voice_id

        self._ready = False



    def load(self) -> None:

        try:

            import pyttsx3

        except ImportError as exc:

            raise ModelLoadError("pyttsx3 is not installed. Run: uv add pyttsx3") from exc



        try:

            engine = pyttsx3.init()

            engine.stop()

        except Exception as exc:  # noqa: BLE001

            raise ModelLoadError(f"Failed to initialize pyttsx3 engine: {exc}") from exc



        self._ready = True



    def synthesize(self, text: str, output_path: str, **kwargs: Any) -> dict[str, Any]:

        if not self._ready:

            raise ModelLoadError(f"{self.name}: engine not loaded -- call load() first")



        Path(output_path).parent.mkdir(parents=True, exist_ok=True)



        try:

            import pyttsx3



            with Timer() as timer:

                engine = pyttsx3.init()

                engine.setProperty("rate", self.rate)

                if self.voice_id:

                    engine.setProperty("voice", self.voice_id)

                engine.save_to_file(text, output_path)

                engine.runAndWait()

                engine.stop()

        except Exception as exc:  # noqa: BLE001

            raise InferenceError(f"pyttsx3 synthesis failed: {exc}", model=self.name) from exc



        try:

            output_duration_sec = get_audio_duration_sec(output_path)

            if output_duration_sec <= 0:

                raise InferenceError(

                    f"pyttsx3 produced empty audio (0 duration) at {output_path}", model=self.name

                )

            samples, sr = load_audio(output_path)

        except AudioLoadError as exc:

            raise InferenceError(

                f"pyttsx3 produced no readable audio at {output_path}: {exc}", model=self.name

            ) from exc



        rtf = compute_rtf(timer.elapsed_ms / 1000.0, output_duration_sec)

        word_count = len(text.split())

        speech_rate_wpm = (

            (word_count / output_duration_sec) * 60.0 if output_duration_sec > 0 else None

        )

        silence_ratio = compute_silence_ratio(samples, sr)



        return {

            "output_path": output_path,

            "generation_latency_ms": timer.elapsed_ms,

            "output_duration_sec": output_duration_sec,

            "rtf": rtf,

            "speech_rate_wpm": speech_rate_wpm,

            "sample_rate": sr,

            "channels": 1,

            "silence_ratio": silence_ratio,

        }



    def unload(self) -> None:

        self._ready = False

