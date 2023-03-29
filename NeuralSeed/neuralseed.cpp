#include "daisy_petal.h"
#include "daisysp.h"
#include "terrarium.h"
#include <RTNeural/RTNeural.h>

// Model Weights
#include "all_model_data.h"
//#include "lstm6_test.h"
//#include "ts9_lstm7_is2_gainKnob.h"
//#include "prince_is3_lstm7.h"

using namespace daisy;
using namespace daisysp;
using namespace terrarium;

// Declare a local daisy_petal for hardware access
DaisyPetal hw;
Parameter inLevel, modelParam, modelParam2, outLevel, wetDryMix;
bool      bypass;
int       modelInSize;
unsigned int       modelIndex;

Led led1, led2;

RTNeural::ModelT<float, 1, 1,
    RTNeural::LSTMLayerT<float, 1, 6>,
    RTNeural::DenseT<float, 6, 1>> model;

//RTNeural::ModelT<float, 1, 1,
//    RTNeural::GRULayerT<float, 1, 8>,
//    RTNeural::DenseT<float, 8, 1>> model;

RTNeural::ModelT<float, 2, 1,
      RTNeural::LSTMLayerT<float, 2, 7>,
      RTNeural::DenseT<float, 7, 1>> model2;
	  
RTNeural::ModelT<float, 3, 1,
      RTNeural::LSTMLayerT<float, 3, 7>,
      RTNeural::DenseT<float, 7, 1>> model3;

// Notes: With default settings, LSTM 8 is max size (7 to be safe)
//        Parameterized LSTM 8 is too much (1 knob), 7 works
//  Each LSTM 7 parameterized model takes up 3008 bytes


void changeModel()
{
    if (bypass) {
       return;
    }
    if (modelIndex == model_collection.size() - 1) {
        modelIndex = 0;
    } else {
        modelIndex += 1;
    }

    if (model_collection[modelIndex].rec_weight_ih_l0.size() == 2) {
      auto& lstm = (model2).template get<0>();
      auto& dense = (model2).template get<1>();
      modelInSize = 2;
      lstm.setWVals(model_collection[modelIndex].rec_weight_ih_l0);
      lstm.setUVals(model_collection[modelIndex].rec_weight_hh_l0);
      lstm.setBVals(model_collection[modelIndex].lstm_bias_sum);
      dense.setWeights(model_collection[modelIndex].lin_weight);
      dense.setBias(model_collection[modelIndex].lin_bias.data());
      led2.Set(0.4f);

    } else if (model_collection[modelIndex].rec_weight_ih_l0.size() == 3) {
      auto& lstm = (model3).template get<0>();
      auto& dense = (model3).template get<1>();
      modelInSize = 3;
      lstm.setWVals(model_collection[modelIndex].rec_weight_ih_l0);
      lstm.setUVals(model_collection[modelIndex].rec_weight_hh_l0);
      lstm.setBVals(model_collection[modelIndex].lstm_bias_sum);
      dense.setWeights(model_collection[modelIndex].lin_weight);
      dense.setBias(model_collection[modelIndex].lin_bias.data());
      led2.Set(1.0f);

    } else {
      auto& lstm = (model).template get<0>();
      auto& dense = (model).template get<1>();
      modelInSize = 1;
      lstm.setWVals(model_collection[modelIndex].rec_weight_ih_l0);
      lstm.setUVals(model_collection[modelIndex].rec_weight_hh_l0);
      lstm.setBVals(model_collection[modelIndex].lstm_bias_sum);
      dense.setWeights(model_collection[modelIndex].lin_weight);
      dense.setBias(model_collection[modelIndex].lin_bias.data());
      led2.Set(0.0f);
    }

    // Initialize Neural Net
    if (modelInSize == 1) {
        model.reset();
    } else if (modelInSize == 2) {
        model2.reset();
    } else {
        model3.reset();
    }
}

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
    float model_param2 = modelParam2.Process();
    float out_level = outLevel.Process(); 
    float wet_dry_mix = wetDryMix.Process();

    //float input_arr1[1] = { 0.0 };
    //float input_arr2[2] = { 0.0, 0.0 };
    float input_arr[3] = { 0.0, 0.0, 0.0 };

    // (De-)Activate bypass and toggle LED when left footswitch is pressed
    if(hw.switches[Terrarium::FOOTSWITCH_1].RisingEdge())
    {
        bypass = !bypass;
        led1.Set(bypass ? 0.0f : 1.0f);
    }

    // Cycle available models
    if(hw.switches[Terrarium::FOOTSWITCH_2].RisingEdge())
    {  
        changeModel();
    }

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
            if (modelInSize == 2) {
                input_arr[0] = input * in_level;
                input_arr[1] = model_param;
                wet = model2.forward (input_arr) + input;  // Run Parameterized Model and add Skip Connection
                
                
            } else if (modelInSize == 3) {
                input_arr[0] = input * in_level;
                input_arr[1] = model_param;
		        input_arr[2] = model_param2;
                wet = model3.forward (input_arr) + input;  // Run Parameterized Model and add Skip Connection

            } else {
                input_arr[0] = input * in_level;           // Set input array with input level adjustment
                wet = model.forward (input_arr) + input;   // Run Model and add Skip Connection
            }

            wet = wet * wet_dry_mix  + input * (1 - wet_dry_mix);  // Set Wet/Dry Mix

            out[0][i] = wet * out_level;                           // Set output level
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

    setupWeights();
    //hw.SetAudioBlockSize(4);

    // Initialize your knobs here like so:
    // parameter.Init(hw.knob[Terrarium::KNOB_1], 0.0f, 1.0f, Parameter::LOGARITHMIC);

    inLevel.Init(hw.knob[Terrarium::KNOB_1], 0.0f, 3.0f, Parameter::LINEAR);
    wetDryMix.Init(hw.knob[Terrarium::KNOB_2], 0.0f, 1.0f, Parameter::LINEAR);
    outLevel.Init(hw.knob[Terrarium::KNOB_3], 0.0f, 1.0f, Parameter::LINEAR);
    modelParam.Init(hw.knob[Terrarium::KNOB_4], 0.0f, 1.0f, Parameter::LINEAR);
    modelParam2.Init(hw.knob[Terrarium::KNOB_5], 0.0f, 1.0f, Parameter::LINEAR);
    //modelParam3.Init(hw.knob[Terrarium::KNOB_6], 0.0f, 1.0f, Parameter::LINEAR);

    // Set samplerate for your processing like so:
    // verb.Init(samplerate);

    // Initialize the correct model
    modelIndex = -1;
    changeModel();

    // Init the LEDs and set activate bypass
    led1.Init(hw.seed.GetPin(Terrarium::LED_1),false);
    led1.Update();
    bypass = true;

    led2.Init(hw.seed.GetPin(Terrarium::LED_2),false);
    led2.Update();

    // TODO: How to flash LED when cycling models?
    //led2.Init(hw.seed.GetPin(Terrarium::LED_2),false);
    //led2.Update();

    hw.StartAdc();
    hw.StartAudio(AudioCallback);
    while(1)
    {
        // Do Stuff Infinitely Here
        System::Delay(10);
    }
}
