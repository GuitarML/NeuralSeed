# NeuralSeed

Neural Seed uses neural networks to emulate amps/pedals on the Daisy Seed hardware by [Electro-Smith](https://www.electro-smith.com/), and the Terrarium
pedal by [PedalPCB](https://www.pedalpcb.com/product/pcb351/). Models are trained on audio recordings from your amp or pedal, and can be uploaded to the Daisy Seed
as firmware. NeuralSeed includes several built in amp/pedal models, as well as controls for input/output level, wet/dry mix, and up to 3 parameterized knobs
for the selected neural model. Effective for Amps/PreAmps (direct out, no cab), Distortion/Overdrive/Boost pedals (non-time based, no Reverb/Delay/Flange/Phaser).

View the Demo on [YouTube](https://youtu.be/c3zJRMWr_Ls)

Read the tech article published in [Towards Data Science](https://medium.com/p/ea9cdad2a29c)

Download the neuralseed.bin for Daisy Seed from the [Releases](https://github.com/GuitarML/NeuralSeed/releases) page. See release notes for a list of built in amp/pedal models.

Start training models for NeuralSeed the easy way using the [Colab Script](https://colab.research.google.com/github/GuitarML/Automated-GuitarAmpModelling/blob/ns-capture/ProteusCapture.ipynb)!

![app](https://github.com/GuitarML/NeuralSeed/blob/main/neuralseed.jpg)

The Daisy Seed is a powerful ARM Cortex-M7 powered board intended for audio effects. The Terrarium pcb provides input and output buffers, as well as connections
to the Daisy Seed for up to 6 potentiometer knobs, 4 switches, in/out jacks, and 9v power supply. NeuralSeed is the software developed by GuitarML to run on this hardware. 
Total material cost was less than $100 to build a fully capable digital guitar multi-effect.

For information on compiling NeuralSeed in the Daisy Ecosystem see the [Daisy Wiki](https://github.com/electro-smith/DaisyWiki/wiki).

## Technical Info
In comparison to other GuitarML plugins, Neural Seed is very minimal, running only a GRU size 10 (by comparison
[Proteus](https://github.com/GuitarML/Proteus) uses LSTM size 40). Using this size model, may not be able to accurately capture certain devices, especially
high gain amps, but should work decently well for distortion/overdrive pedals and low gain amps (direct out, not
from a microphone). This is due to limited processing power on the M7 microcontroller. The current code processes on
float32 audio data, so this could be optimized for the M7 chip using quantized int16 data. The Daisy Seed
supports the addition of a display so this could be a future modification, as well as an encoder knob to switch models.

The compiled binary and model data fits into Flash Memory, which is limited to 128KB. The [RTNeural](https://github.com/jatinchowdhury18/RTNeural)
engine is used for fast inferencing of the neural models with a very tiny footprint.  It is possible to add more models utilizing other data storage 
areas on the Daisy Seed.

Utilizing only the 128KB Flash memory, NeuralSeed can hold around 15 different models. The initial release includes 11 built in models,
and there is room to add more, however the lack of a display means it's up to you to keep track of what model you're on! For live shows one 
could choose a subset of their favorite models by editing the ```NeuralSeed/all_model_data.h``` file to make this more manageable.

## Getting started
Build the daisy libraries and NeuralSeed with (after installing the Daisy Toolchain):
```
make -C DaisySP
make -C libDaisy
cd NeuralSeed
make
```

Then flash your terrarium with the following commands (or use the [Electrosmith Web Programmer](https://electro-smith.github.io/Programmer/))
```
cd your_pedal
# using USB (after entering bootloader mode)
make program-dfu
# using JTAG/SWD adaptor (like STLink)
make program
```

# Control

| Control | Description | Comment |
| --- | --- | --- |
| Ctrl 1 | Input Level | Adjusts the input level to the neural net model (0 to 3x volume) |
| Ctrl 2 | Mix | Set the dry/wet amount (full left for dry, full right for only neural net output |
| Ctrl 3 | Volume | Sets the output volume of the pedal |
| Ctrl 4 | Neural Param 1 | Adjusts the first model parameter, such as gain/tone (only applies to 1-conditioned models) |
| Ctrl 5 | Neural Param 2 | Adjusts the second model parameter, such as gain/tone (only applies to 2-conditioned models)  |
| Ctrl 6 | Neural Param 3 | Adjusts the third model parameter, such as gain/tone (only applies to 3-conditioned models) |
| SW 1 - 4 | 4Band EQ Boost | Bass, Low Mid, High Mid, Treble Boost (Band Pass) |
| FS 1 | Bypass/Active | Bypass / effect engaged |
| FS 2 | Cycle Neural Model | Loads the next available Neural Model, starts at beginning after the last in the list (i.e. next pedal/amp) |
| LED 1 | Bypass/Active Indicator |Illuminated when effect is set to Active |
| LED 2 | Brightness based on selected model parameterization | Off = Snapshot, Dim = 1 Knob, Brighter = 2 Knobs, Brightest = 3 Knobs |
| Audio In 1 | Audio input | Mono only for Terrarium |
| Audio Out 1 | Mix Out | Mono only for Terrarium |

## Training your own Models (Amp/Pedal Capture)

You can train models for Neural Seed using the GuitarML fork (ns-capture branch) of [Automated-GuitarAmpModelling](https://github.com/GuitarML/Automated-GuitarAmpModelling/tree/ns-capture) code.
Run "scripts/convert_json_to_c_header.py" on the resulting JSON model to generate a .h file, which you can copy and paste the weights into the "all_model_data.h" header before
compiling "neuralseed.bin". Only GRU size 10 models (snapshot, 1-param) or GRU size 8 (2-param, 3-param) models are currently compatible. Train models using 48kHz to match Daisy Seed processing 
(note that this is different from most other GuitarML plugins). A 48kHz input wav file based on the Proteus input wav is available [here](https://github.com/GuitarML/Automated-GuitarAmpModelling/blob/ns-capture/Data/Proteus_Capture_48k.wav).

Keep in mind that capture with such small models is hit or miss, and may take some massaging to get a proper capture, or may not be captured at all. Loss values over 0.08 probably won't sound good,
or may contain harmonic noise. Ensure to start with volume low when testing a new model on the pedal.

## Adding your model to NeuralSeed
To add your models to a custom build of Neural Seed, run the converter script on your json model. <br>
```python convert_json_to_c_header_gru.py <model_file.json>```<br>
Then copy and paste the information from the generated c-header into the ```all_model_data.h``` file 
before compiling. Ensure to modify the "Model" weights to a unique identifier, and add to the initialization list a the top and the "model_collection" vector at the bottom.

Email either the .json or .h model file to smartguitarml@gmail.com and it may be included in the official repo.
