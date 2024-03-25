# A collection of functions designed to convert a model made up of several nested sequentials into partitions
# that facilitates model sharding. Initially based on the model formatting required for 
# GPipe (https://torchgpipe.readthedocs.io/en/stable/)

# Flattens all models which have ResNet like staging (stage1, stage2,...; here called layer1, layer2,...)
# Considers the average pooling, flatten and fully connected layers at the end to be one stage (final stage)
# Considers an 'input branch', which contains the input stage downsampling (7x7 conv, see vanilla ResNet)

# Returns a torch.nn.Sequential object consisting of the 'flattened' ResNet-like model
def FlattenResnet(resnet_model): 
    modules = []

    # Input block
    modules.append(nn.Sequential(*[resnet_model.input_branch]))

    # Process each layer's blocks, where the first block is usually the downsampling one
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(resnet_model, layer_name)
        for i, bottleneck in enumerate(layer):
            # Each bottleneck block as a separate sequential
            modules.append(nn.Sequential(bottleneck))

    # Final stage combined into one block
    final_stage = nn.Sequential(
        resnet_model.avgpool,
        nn.Flatten(1),
        resnet_model.fc
    )
    modules.append(final_stage)

    return nn.Sequential(*modules)
