####################################################################################
# This script converts a PyTorch json model (LSTM or GRU) to a c style header file.
# The purpose is to compress the model data to include in compiled c++ programs.
#
# Tested with LSTM and GRU models trained from the Automated-GuitarAmpModelling tool. 
#
# usage: python convert_json_to_c_header.py <json_model>
#
# Generates a .h file with the same name as the json model.
#
####################################################################################

import json
import argparse
import sys
import numpy as np

def add_layer(layer_name, layer_data, header_data):
  """
  Adds layer weights and biases to header_data

  Inputs:
  layer_name (string) : The name of the layer as labelled by Pytorch (i.e. "rec.weight_ih_l0")
  layer_data (list)   : Either a 1D or 2D list of weights loaded from json file.
  header_data (list)  : The cumulative list of lines to write to the header file.
  """

  line = ""
  # Use appropriate declaration for 1D or 2D vector
  if layer_name == "rec.weight_ih_l0" or layer_name == "rec.weight_hh_l0" or layer_name == "lin.weight":
    var_declaration = "  Model." + layer_name.replace(".", "_") + " = "
  else: 
    var_declaration = "  Model." + layer_name.replace(".", "_") + " = "
  line += var_declaration

  line += "{"
  c = 0        # Counter to determine last item in list

  # If reading a 2d array
  if len(np.asarray(layer_data).shape) > 1:
    for i in layer_data:
      c += 1
      if c == 1:
        line += "{"
      else:
        line += " "*len(var_declaration) + " { "
      c2 = 0   # 2nd counter to determine last item in list
      for j in i:
        c2 += 1
        if c2 == len(i):
          line += str(j)
        else:
          line += str(j) + ", "
      if c == len(layer_data):
        line += "}}; "
      else:
        line += "}, "
      header_data.append(line)
      line = ""
  # Else if reading a 1d array
  else:
    for i in layer_data:
      c += 1
      line += str(i)
      if c == len(layer_data):
        line += "}; "
      else:
        line += ", "

    header_data.append(line)

  header_data.append("")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("json_model", type=str, default="", help="Path to json file, including filename")
  args = parser.parse_args()

  # Enter target json file here
  f = open(args.json_model)
  data = json.load(f)

  header_data = []
  header_data.append("//========================================================================")
  header_data.append("//" + args.json_model.split(".json")[0])
  # Read Model Data from Pytorch Json model to include in .h as comment
  header_data.append("/*")
  for item in data['model_data'].keys():
    header_data.append(item + " : " + str(data['model_data'][item]))
  header_data.append("*/\n")

  # Read the state dict from Pytorch Json model and reorganize data into c header format
  # Sum the rec.bias layers, skip adding individually
  for layer_name in data['state_dict'].keys():
    if layer_name.startswith("rec.bias_") == True:
      continue
    if layer_name == "rec.weight_ih_l0" or layer_name == "rec.weight_hh_l0":
      add_layer(layer_name, np.array(data['state_dict'][layer_name]).T, header_data) # Transpose 2D arrays (Pytorch->Tensorflow/Keras)
    else:
      add_layer(layer_name, data['state_dict'][layer_name], header_data)

  bias_sum = np.array(data['state_dict']['rec.bias_ih_l0']) + np.array(data['state_dict']['rec.bias_hh_l0'])
  add_layer('lstm_bias_sum', bias_sum, header_data)

  # Write data to .h file
  new_filename = args.json_model.split(".json")[0] + '.h'
  file = open(new_filename,'w')
  for item in header_data:
    file.write(item+"\n")
  file.close()
  print("Finished generating header file: "+ new_filename)
