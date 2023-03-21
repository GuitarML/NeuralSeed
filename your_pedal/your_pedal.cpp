#include "daisy_petal.h"
#include "daisysp.h"
#include "terrarium.h"

using namespace daisy;
using namespace daisysp;
using namespace terrarium;

// Declare a local daisy_petal for hardware access
DaisyPetal hw;

bool      bypass;

Led led1, led2;

// This runs at a fixed rate, to prepare audio samples
void callback(float *in, float *out, size_t size)
{
    hw.ProcessAllControls();
    led1.Update();
    led2.Update();

    // (De-)Activate bypass and toggle LED when left footswitch is pressed
    if(hw.switches[Terrarium::FOOTSWITCH_1].RisingEdge())
    {
        bypass = !bypass;
        led1.Set(bypass ? 0.0f : 1.0f);
    }

    for(size_t i = 0; i < size; i += 2)
    {
        float dryl, dryr;
        dryl  = in[i];
        dryr  = in[i + 1];

        // Process your signal here

        if(bypass)
        {
            out[i]     = in[i];     // left
            out[i + 1] = in[i + 1]; // right
        }
        else
        {
            out[i]     = in[i]; // Replace in[i] with your left processed signal
            out[i + 1] = in[i + 1]; // Replace in[i + 1] with your right processed signal
        }
    }
}

int main(void)
{
    float samplerate;

    hw.Init();
    samplerate = hw.AudioSampleRate();

    // Initialize your knobs here like so:
    // parameter.Init(hw.knob[Terrarium::KNOB_1], 0.6f, 0.999f, Parameter::LOGARITHMIC);

    // Set samplerate for your processing like so:
    // verb.Init(samplerate);


    // Init the LEDs and set activate bypass
    led1.Init(hw.seed.GetPin(Terrarium::LED_1),false);
    led1.Update();
    bypass = true;

    hw.StartAdc();
    hw.StartAudio(callback);
    while(1)
    {
        // Do Stuff Infinitely Here
        System::Delay(10);
    }
}