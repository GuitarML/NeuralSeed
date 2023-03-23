#include "daisy_petal.h"
#include "daisysp.h"
#include "terrarium.h"
#include "../RTNeural/RTNeural/RTNeural.h"

using namespace daisy;
using namespace daisysp;
using namespace terrarium;

// Declare a local daisy_petal for hardware access
DaisyPetal hw;

bool      bypass;

Led led1, led2;

RTNeural::ModelT<float, 1, 1,
    RTNeural::LSTMLayerT<float, 1, 4>,
    RTNeural::DenseT<float, 4, 1>> model;



// This runs at a fixed rate, to prepare audio samples
static void AudioCallback(AudioHandle::InputBuffer  in,
                          AudioHandle::OutputBuffer out,
                          size_t                    size)
{
    //hw.ProcessAllControls();
    hw.ProcessAnalogControls();
    hw.ProcessDigitalControls();
    led1.Update();
    led2.Update();

    // (De-)Activate bypass and toggle LED when left footswitch is pressed
    if(hw.switches[Terrarium::FOOTSWITCH_1].RisingEdge())
    {
        bypass = !bypass;
        led1.Set(bypass ? 0.0f : 1.0f);
    }

    for(size_t i = 0; i < size; i++)
    {
        //for(int chn = 0; chn < 2; chn++)
        //{
        float input = in[0][i];
        float wet   = input;
        //dryl  = in[i];
        //dryr  = in[i + 1];

        // Process your signal here
        if(bypass)
        {
            out[0][i] = in[0][i];
        }
        else
        {
            float input_arr[] = { input };
            out[0][i] = model.forward (input_arr) + in[0][i];  // Run Model and add Skip Connection
        }

        // Copy left channel to right channel (see how well mono processing works then try stereo)
        for(size_t i = 0; i < size; i++)
        {
            out[1][i] = out[0][i];
        }
        
    }
}

int main(void)
{
    float samplerate;

    hw.Init();
    samplerate = hw.AudioSampleRate();
    //hw.SetAudioBlockSize(4);

    // Initialize your knobs here like so:
    // parameter.Init(hw.knob[Terrarium::KNOB_1], 0.6f, 0.999f, Parameter::LOGARITHMIC);

    // Set samplerate for your processing like so:
    // verb.Init(samplerate);

    // Initialize Neural Net
    model.reset();

    // Init the LEDs and set activate bypass
    led1.Init(hw.seed.GetPin(Terrarium::LED_1),false);
    led1.Update();
    bypass = true;

    hw.StartAdc();
    hw.StartAudio(AudioCallback);
    while(1)
    {
        // Do Stuff Infinitely Here
        System::Delay(10);
    }
}