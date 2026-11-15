# kai-client

<!-- to-do: readme -->

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