#include "daisy_petal.h"
#include "daisysp.h"
#include "terrarium.h"
#include <RTNeural/RTNeural.h>

// Model Weights
#include "ts9_lstm7_is2_gainKnob.h"

using namespace daisy;
using namespace daisysp;
using namespace terrarium;

// Declare a local daisy_petal for hardware access
DaisyPetal hw;
Parameter inLevel, modelParam, outLevel, wetDryMix;
bool      bypass;

Led led1, led2;

//RTNeural::ModelT<float, 1, 1,
//    RTNeural::LSTMLayerT<float, 1, 10>,
//    RTNeural::DenseT<float, 10, 1>> model;


//RTNeural::ModelT<float, 1, 1,
//    RTNeural::GRULayerT<float, 1, 8>,
//    RTNeural::DenseT<float, 8, 1>> model;

RTNeural::ModelT<float, 2, 1,
      RTNeural::LSTMLayerT<float, 2, 7>,
      RTNeural::DenseT<float, 7, 1>> model;

// Notes: With default settings, LSTM 8 is max size
//        Parameterized LSTM 8 is too much (1 knob), 7 works
//  Each LSTM 7 parameterized model takes up 3008 bytes

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

    float in_level = inLevel.Process();
    float model_param = modelParam.Process();
    float out_level = outLevel.Process(); 
    float wet_dry_mix = wetDryMix.Process();

    float input_arr[2] = { 0.0, 0.0 };

    // (De-)Activate bypass and toggle LED when left footswitch is pressed
    if(hw.switches[Terrarium::FOOTSWITCH_1].RisingEdge())
    {
        bypass = !bypass;
        led1.Set(bypass ? 0.0f : 1.0f);
    }

    // Cycle available models
    //if(hw.switches[Terrarium::FOOTSWITCH_2].RisingEdge())
    //{  
        
    //}

    for(size_t i = 0; i < size; i++)
    {
        float input = in[0][i];
        float wet   = input;

        // Process your signal here
        if(bypass)
        {
            out[0][i] = in[0][i];
        }
        else
        {
            //float input_arr[] = { input * in_level };     // Set input array with input level adjustment
            //wet = model.forward (input_arr) + input;    // Run Model and add Skip Connection

            //float input_arr[2] = { input * in_level, model_param  };
            input_arr[0] = input * in_level;
            input_arr[1] = model_param;
            wet = model.forward (input_arr) + input;  // Run Parameterized Model and add Skip Connection

            wet = wet * wet_dry_mix  + input * (1 - wet_dry_mix);  // Set Wet/Dry Mix

            out[0][i] = wet * out_level;                       // Set output level
        }

        // Copy left channel to right channel (see how well mono processing works then try stereo)
	// Not needed for Terrarium, mono only (left channel)
        //for(size_t i = 0; i < size; i++)
        //{
        //    out[1][i] = out[0][i];
        //}
        
    }
}

int main(void)
{
    float samplerate;

    hw.Init();
    samplerate = hw.AudioSampleRate();
    //hw.SetAudioBlockSize(4);

    // Initialize your knobs here like so:
    // parameter.Init(hw.knob[Terrarium::KNOB_1], 0.0f, 1.0f, Parameter::LOGARITHMIC);

    inLevel.Init(hw.knob[Terrarium::KNOB_1], 0.0f, 2.0f, Parameter::LINEAR);
    modelParam.Init(hw.knob[Terrarium::KNOB_2], 0.0f, 1.0f, Parameter::LINEAR);
    outLevel.Init(hw.knob[Terrarium::KNOB_3], 0.0f, 1.5f, Parameter::LINEAR);
    wetDryMix.Init(hw.knob[Terrarium::KNOB_4], 0.0f, 1.0f, Parameter::LINEAR);

    // Set samplerate for your processing like so:
    // verb.Init(samplerate);

    // Initialize the correct model
    auto& lstm = (model).template get<0>();
    auto& dense = (model).template get<1>();

    lstm.setWVals(rec_weight_ih_l0);
    lstm.setUVals(rec_weight_hh_l0);
    lstm.setBVals(lstm_bias_sum);

    dense.setWeights(lin_weight);
    dense.setBias(lin_bias.data());

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
