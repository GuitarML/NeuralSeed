#include "daisy_petal.h"
#include "daisysp.h"
#include "terrarium.h"
#include <RTNeural/RTNeural.h>

using namespace daisy;
using namespace daisysp;
using namespace terrarium;

// Declare a local daisy_petal for hardware access
DaisyPetal hw;
Parameter inLevel, modelParam, outLevel, wetDryMix;
bool      bypass;

Led led1, led2;

//RTNeural::ModelT<float, 1, 1,
//    RTNeural::LSTMLayerT<float, 1, 8>,
//    RTNeural::DenseT<float, 8, 1>> model;


//RTNeural::ModelT<float, 1, 1,
//    RTNeural::GRULayerT<float, 1, 8>,
//    RTNeural::DenseT<float, 8, 1>> model;

RTNeural::ModelT<float, 2, 1,
      RTNeural::LSTMLayerT<float, 2, 6>,
      RTNeural::DenseT<float, 6, 1>> model;

// Notes: With default settings, LSTM 8 is max size
//        Parameterized LSTM 8 is too much (1 knob), 6 works


// Model Data (TODO: Move to separate .h file, allocate to SDRAM)
std::vector<std::vector<float>> rec_weight_ih_l0 = {{0.04492504522204399, 0.13859952986240387, -0.09087981283664703, 0.11267407983541489, 0.03184181824326515, -0.03356875851750374, -0.041874226182699203, 0.06536576896905899, 0.011693708598613739, -0.1161300465464592, -0.030807098373770714, -0.02685493975877762, 0.21289752423763275, -1.1547725200653076, 0.4279938340187073, -0.14268267154693604, 0.21674244105815887, -1.276771068572998, -0.009402019903063774, 0.02146889641880989, 0.0850922092795372, -0.027131281793117523, 0.04600798711180687, -0.009451695717871189}, 
                                                    { 1.7534617185592651, -0.1595017910003662, -0.16887572407722473, -0.5121852159500122, 0.8458209037780762, 0.07799922674894333, 0.10656397044658661, -0.5112974047660828, -0.1894082874059677, 0.7241670489311218, 0.7841776013374329, -0.42301422357559204, -0.3452242314815521, 0.07354798913002014, -0.01648104004561901, -0.10752736777067184, -0.07004299759864807, 0.025338606908917427, 0.787774920463562, -0.342814177274704, 0.16810399293899536, 0.3394525647163391, 0.8285894989967346, 0.14417822659015656}}; 

std::vector<std::vector<float>> rec_weight_hh_l0 = {{-0.14109712839126587, 0.750989556312561, -0.05166666954755783, -0.0056837392039597034, 0.016959182918071747, -0.06759186834096909, 0.06546922773122787, 0.011802091263234615, 0.03337809443473816, -0.08747854828834534, 0.11178101599216461, -0.024870410561561584, 1.3579564094543457, -0.31638866662979126, 0.6407556533813477, 0.25555840134620667, 0.052156995981931686, 0.02845791168510914, -0.06254027038812637, -0.7458108067512512, -0.03669289872050285, 0.037154704332351685, 0.0036812517791986465, 0.013653039000928402}, 
                                                    { 0.043547097593545914, -0.21753451228141785, 0.019742053002119064, -0.1968538761138916, -0.007471416611224413, 0.03152463957667351, -0.06896892189979553, -0.0383739247918129, -0.18550029397010803, 0.05748968571424484, 0.0012710774317383766, -0.03241857886314392, -0.40814340114593506, -0.057983797043561935, 0.042467378079891205, -0.2506653964519501, -1.6184028387069702, -0.503940761089325, -0.060017842799425125, 0.0658622533082962, -0.005728655494749546, -0.06270314007997513, -0.022478951141238213, -0.00332927075214684}, 
                                                    { 0.15782582759857178, 0.3767658770084381, -0.03295834735035896, 0.017468789592385292, -0.00547043327242136, -0.042702481150627136, 0.03938255086541176, 0.05369069427251816, -0.1560838222503662, 0.09205376356840134, -0.016548017039895058, -0.1079864576458931, -0.21931509673595428, 0.3941478431224823, 0.3050908148288727, -0.0885302871465683, -0.2666374444961548, -0.41678130626678467, 0.13890540599822998, -0.3410876989364624, -0.028263559564948082, -0.027031434699892998, -0.009549877606332302, -0.025702545419335365}, 
                                                    { -0.027064254507422447, -0.20807035267353058, 0.005652260966598988, 0.08988684415817261, 0.025916198268532753, 0.018056465312838554, -0.14808273315429688, -0.07751591503620148, 0.0490245558321476, 0.07638760656118393, 0.01788109913468361, 0.011086731217801571, -0.2409871220588684, 0.050763748586177826, 0.06379454582929611, 0.21936899423599243, 1.4699606895446777, -1.0873404741287231, -0.08714527636766434, -0.05603647604584694, 0.09442127496004105, -0.02293451502919197, 0.01940147392451763, 0.015472915023565292}, 
                                                    { -0.025227515026926994, 0.21952548623085022, 0.002134189009666443, 0.15564438700675964, -0.04863731935620308, -0.004889289382845163, -0.06264127045869827, -0.036292653530836105, 0.03938187658786774, 0.023098228499293327, -0.11761640757322311, -0.084817074239254, -3.0599677562713623, 0.8476256728172302, -0.23333430290222168, -0.9431977868080139, 0.2334945946931839, 0.25588294863700867, 0.0022656405344605446, -0.2920939028263092, 0.018304673954844475, 0.007316294126212597, -0.0459890253841877, -0.011405552737414837}, 
                                                    { -0.07076916098594666, -0.050609953701496124, -0.013246467337012291, -0.037303242832422256, -0.023028194904327393, 0.013678017072379589, 0.0795641839504242, -0.039864130318164825, 0.12798388302326202, 0.000398938893340528, 0.014727553352713585, 0.04371066391468048, -0.10369338095188141, -0.5950339436531067, 0.7538284063339233, -0.06191335991024971, -1.661817193031311, -0.5543313026428223, -0.031945787370204926, 0.06762751936912537, -0.14314891397953033, 0.07986963540315628, -0.01572539657354355, 0.021279102191329002}}; 

std::vector<std::vector<float>> lin_weight = {{0.011624181643128395, -0.2012190818786621, -1.009621500968933, 0.5557087659835815, -0.14826491475105286, 1.2326418161392212}}; 

std::vector<float> lin_bias = {-0.0005421611131168902}; 

std::vector<float> lstm_bias_sum = {0.14448755234479904, 0.9701187610626221, -2.006475567817688, -2.690411686897278, 1.4193756580352783, 1.4930198788642883, -0.09002597071230412, -0.5597395896911621, 2.457030177116394, 3.853874683380127, -0.4321446716785431, -0.0176589866168797, 0.24603769183158875, -0.05389595031738281, 0.008824095129966736, 0.07581812143325806, 0.05998519994318485, -0.0018917284905910492, 0.8196917474269867, 1.1030535697937012, 0.8081759214401245, 1.332111418247223, 1.5972752571105957, 1.7163652181625366}; 




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

