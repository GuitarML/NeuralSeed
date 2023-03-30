# NeuralSeed
## Author

Keith Bloemer (GuitarML)

## Description

A description for your pedal for the [terrarium](https://www.pedalpcb.com/product/pcb351/) from [PedalPCB](https://www.pedalpcb.com).

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