class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    preset = "deeplab_v3_plus_resnet50_pascalvoc"  
    image_size = [224, 224] 
    seg_image_size = [28,28]
    epochs = 25 
    batch_size = 10 
    drop_remainder = True  
    num_classes = 1 
    cache = True 