import yaml
import torch

INT_MAX = 2 ** 30

import onnx

from onnx_coreml import convert

model_name = 'gnet5'
onnx_model = onnx.load(model_name+'.onnx')

##print(onnx.shape_inference.infer_shapes(onnx_model))
#infer_shapes(onnx_model)


up_counter = 0
resize_cout = 0
def _convert_resize(builder, node, graph, err):
    '''
    convert to CoreML Upsample or Resize Bilinear Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L2139
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L2178
    '''
    mode = node.attrs.get('mode', 'nearest')
    if node.inputs[1] not in node.input_tensors:
        return err.unsupported_op_configuration(builder, node, graph,
                                                "Scaling factor unknown!! CoreML does not support dynamic scaling for Resize")

    mode = 'NN' if mode == 'nearest' else 'BILINEAR'
    scale = node.input_tensors[node.inputs[2]]
    print('Resize {}-{}'.format(node.name, scale))
    global resize_cout
    #if resize_cout == 0:
    #    builder.add_resize_bilinear(node.name, node.inputs[0], node.outputs[0], target_height=64, target_width=64)
    #   resize_cout+=1
    #    return
    resize_cout +=1
    if len(scale) == 0:
        global up_counter
        print('{} - counter {}'.format(node.name,up_counter))
        if up_counter == 0:
            builder.add_resize_bilinear(node.name, node.inputs[0], node.outputs[0], target_height=128, target_width=128)
            up_counter+=1
            return
        if up_counter == 1:
            builder.add_resize_bilinear(node.name, node.inputs[0], node.outputs[0], target_height=256, target_width=256)
            up_counter+=1
            return


        print('no scale')

    builder.add_upsample(
        name=node.name,
        scaling_factor_h=scale[-2],
        scaling_factor_w=scale[-1],
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode=mode
    )

def load_input_constants(builder, node, graph, err):
    for i in range(len(node.inputs)):
        if node.inputs[i] in node.input_tensors and node.inputs[i] not in graph.constants_loaded:
            value = node.input_tensors[node.inputs[i]]
            builder.add_load_constant_nd(
                name=node.name + '_load_constant_' + str(i),
                output_name=node.inputs[i],
                constant_value=value,
                shape=[1] if value.shape == () else value.shape
            )
            graph.constants_loaded.add(node.inputs[i])

def _convert_tile(builder, node, graph, err):
    '''
    convert to CoreML Tile Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L5117
    '''
    load_input_constants(builder, node, graph, err)
    if node.inputs[1] not in node.input_tensors:
        err.unsupported_op_configuration(builder, node, graph, "CoreML Tile layer does not support dynamic 'reps'. 'reps' should be known statically")
    builder.add_tile(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        reps=node.input_tensors[node.inputs[1]].astype(np.int32).tolist()
    )

def _convert_pad(builder, node, graph, err):
    '''
    convert to CoreML Padding / ConstantPadding Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L4397
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L1822
    '''
    mode = node.attrs.get('mode', 'constant')

    try:
        mode = mode.decode()
    except (UnicodeDecodeError, AttributeError):
        pass

    if mode == 'constant':
        pads = node.attrs.get('pads', [])
        value = node.attrs.get('value', 0.0)
        if len(pads)==0:
            if node.inputs[1] in node.input_tensors:
                input_names = [node.inputs[0]]
                pads = node.input_tensors[node.inputs[1]]
                pads = pads.tolist()
            else:
                input_names = [node.inputs[0], node.inputs[1]]
        else:
            input_names = [node.inputs[0]]
        builder.add_constant_pad(
            name=node.name,
            input_names=input_names,
            output_name=node.outputs[0],
            value=value,
            pad_to_given_output_size_mode=False,
            pad_amounts=pads
        )


def _convert_slice(builder, node, graph, err):
    '''
    convert to CoreML Slice Static Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L5082
    '''
    if len(node.inputs) == 1:
        import onnx_coreml._operators_nd as original
        return original._convert_slice_ir4v9(builder, node, graph, err)

    if node.inputs[0] not in graph.shape_dict:
        # err.unsupported_op_configuration(builder, node, graph, "Input shape not available")
        for p in node.parents:
            if p.name == node.inputs[0]:
                if p.op_type == 'Shape':
                    if p.inputs[0] in graph.shape_dict:
                        data_shape = [graph.shape_dict[p.inputs[0]]]
                        break
                    else:
                        data_shape = [4]
    else:
        data_shape = graph.shape_dict[node.inputs[0]]
    len_of_data = len(data_shape)
    begin_masks = [True] * len_of_data
    end_masks = [True] * len_of_data

    default_axes = list(range(len_of_data))

    add_static_slice_layer = False
    if node.inputs[1] in node.input_tensors and node.inputs[2] in node.input_tensors:
        if len(node.inputs) > 3:
            if node.inputs[3] in node.input_tensors:
                if len(node.inputs) > 4:
                    if node.inputs[4] in node.input_tensors:
                        add_static_slice_layer = True
                else:
                    add_static_slice_layer = True
        else:
            add_static_slice_layer = True

    if add_static_slice_layer:
        ip_starts = node.input_tensors[node.inputs[1]]
        ip_ends = node.input_tensors[node.inputs[2]]
        axes = node.input_tensors[node.inputs[3]] if len(node.inputs) > 3 else default_axes
        ip_steps = node.input_tensors[node.inputs[4]] if len(node.inputs) > 4 else None

        starts = [0] * len_of_data
        ends = [0] * len_of_data
        steps = [1] * len_of_data

        for i in range(len(axes)):
            current_axes = axes[i]
            starts[current_axes] = ip_starts[i]
            ends[current_axes] = ip_ends[i]
            # n <= end <= INT_MAX implies end is -1, hence end_mask should be True
            # otherwise end_mask should be False
            if ends[current_axes] < data_shape[current_axes]:
                # this means end is not -1
                end_masks[current_axes] = False

            if starts[current_axes] != 0:
                begin_masks[current_axes] = False

            if isinstance(ip_steps, list):
                steps[current_axes] = ip_steps[i]

        print('Add Slice: {}'.format(node.name))
        builder.add_slice_static(
            name=node.name,
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            begin_ids=starts,
            end_ids=ends,
            strides=steps,
            begin_masks=begin_masks,
            end_masks=end_masks
        )
    else:
        err.unsupported_op_configuration(builder, node, graph,
                                         "CoreML does not support Dynamic Slice with unknown axes. Please provide Custom Function/Layer")


custom_conversion_functions={"Resize": _convert_resize,'Tile':_convert_tile,'Pad':_convert_pad,'Slice':_convert_slice}

mlmodel = convert(model=model_name+'.onnx',custom_conversion_functions=custom_conversion_functions,add_custom_layers=True,minimum_ios_deployment_target='13')
# Save converted CoreML model
mlmodel.save(model_name+'.mlmodel')
