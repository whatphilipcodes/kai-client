# kai-client

### Manual Voice Cloning

1. Get a recording of the target voice reading out a reference text (`assets/voices/ref-long.txt` or `assets/voices/ref-short.txt`). The clip should be 10-20 seconds long ideally recorded with medium to high quality (ie. NOT recorded on the integrated microphone array on the Alienware as the recording dictates the quality of the voice clone). The file needs to be of the following format:
    
    ```
    Waveform audio (.wav)
    Audio Channels: Mono
    Sample Rate: 48kHz
    Bits per Sample: 16
    ```
    *To convert into the correct format a tool like [Audacity](https://www.audacityteam.org/) (free download) might come in handy.*

2. Place the recording in `assets/voices/source-<name>.wav`. Then edit `config.toml` to use the new file in the clients requests:

    ```toml
    [tts]
    (...)
    ref_audio = "assets/voices/source-<name>.wav"
    ref_text = "assets/voices/ref-<long|short>.txt"
    (...)
    ```
    *Make sure to pass the same reference text used in the recording to `ref_text`.*

3. Restart the client application by closing it if already running and then either via the Desktop shortcut or in a terminal using:
    ```sh
    cd <repository root>
    uv run kai-client
    ```

### Distortion & Misuse

At the moment exclusively controlled via the `distortion & misuse` block in the `config.toml`. With the generation parameters the generated repossess can be affected. When modifying, generation is becoming very unpredictable quickly and even small changes destabilize the output entirely. The commented out example config strikes a middle ground with occasional utterance of random languages and word particles. This becomes increasingly noticeable with longer conversations (history limit is set to 8 turns). The first response usually comes out legible and then onwards the responses derail slightly until there are only fragments generated.

- `top_k`: Set to `0` for strong deviations and rambling in random languages. At anything above `0 (default: 64)` the model appears to be stable albeit occasionally constructing non-words shortly loosing context (co-depends mostly on temperature)
- `temperature`: Range `0.0` `2.0` (default: `1.0`). From `1.5` upwards output get more unstable, especially with `top_k = 0` and `min_p = 0`.
- `top_p`: Nucleus sampling, keeps only the most likely tokens up to this cumulative probability (default: `0.95`). `1.0` disables it and makes the tail of the distribution reachable.
- `min_p`: Drops tokens below this fraction of the top token's probability (default: `0.0`, off). Around `0.05` keeps high temperatures usable; at `0.0` there is no safety net.
- `presence_penalty`: Range `-2.0` to `2.0` (default: `0.0`). Penalizes every token that has already appeared. Positive values push towards new words, negative values towards repetition.
- `frequency_penalty`: Range `-2.0` to `2.0` (default: `0.0`). Like `presence_penalty`, but scaled by how often a token was used. Near `2.0` even common words get suppressed, leading to invented words and fragments.

*These options can be found in the `config.toml`*


### License Overview

All saved in `licenses.csv`. To update run:
```sh
uv run pip-licenses --format=csv > licenses.csv
```