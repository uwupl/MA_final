import torch
import torch.nn.functional as F
import numpy as np
import numba as nb
# import time

def embedding_concat_frame(embeddings, cuda_active):
    '''
    framework for concatenating more than two features or less than two
    '''
    no_of_embeddings = len(embeddings)
    if no_of_embeddings == int(1):
        embeddings_result = embeddings[0].cpu()
    elif no_of_embeddings == int(2):
        embeddings_result = embedding_concat(embeddings[0], embeddings[1])
    elif no_of_embeddings > int(2):
        for k in range(no_of_embeddings - 1):
            if k == int(0):
                embeddings_result = embedding_concat(embeddings[0], embeddings[1]) # default
                pass
            else:
                if torch.cuda.is_available() and cuda_active:
                    embeddings_result = embedding_concat(embeddings_result.cuda(), embeddings[k+1])
                else:
                    embeddings_result = embedding_concat(embeddings_result, embeddings[k+1].cpu())
    return embeddings_result

def embedding_concat(x, y):
    '''
    alligns dimensions
    
    TODO: numba version plus lightweight version
    
    from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    '''
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def reshape_embedding_old(embedding):
    '''
    flattens spatial dimensions and concatenates channels. Results in 1D-Vector
    
    TODO: numba or numpy version! 
    '''
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return np.array(embedding_list)

@nb.njit
def reshape_embedding(embedding):
    '''
    flattens spatial dimensions and concatenates channels. Results in 1D-Vector
    '''
    # embeddings = np.empty((embedding.shape[0]*embedding.shape[2]*embedding.shape[3], embedding.shape[1]))
    # out = np.reshape(embedding, (embedding.shape[0]*embedding.shape[2]*embedding.shape[3], embedding.shape[1]))
    out = np.empty(shape=(embedding.shape[0]*embedding.shape[2]*embedding.shape[3], embedding.shape[1]), dtype=np.float32) # TODO: dtype?
    counter = int(0)
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                out[counter, :] = embedding[k, :, i, j]
                counter += 1
    return out

def _feature_extraction(images, forward_modules, device):
        # if not evaluation and self.train_backbone:
    #     self.forward_modules["feature_aggregator"].train()
    #     features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
    # else:
    # t_0 = time.perf_counter()
    forward_modules["backbone"].eval() #forward_modules["feature_aggregator"] = 
    # get device of backbone
    # device = next(forward_modules["backbone"].parameters()).device
    # print("device backbone: ", device)
    # get device of images
    # device_images = images.device
    # print(device)
    # in case there is a mismatch, move images to device of backbone. This is especially the case for a quantized backbone
    # if device != device_images:
    images = images.to(device)
    forward_modules["backbone"] = forward_modules["backbone"].to(device)
        # print("moved images to device of backbone")
    # print("device images: ", device_images)
    with torch.no_grad():
        features = forward_modules["backbone"](images) # type dict
        # print("features intern 1: ", features[0].shape)
    # features = [features[layer] for layer in self.layers_to_extract_from] # list of tensors like usual: WRN50 L2: (1, 512, 28, 28), L3: (1, 1024, 14, 14), L4: (1, 2048, 7, 7)
    for k in range(len(features)):
        features[k] = features[k] / 10
    # features[1] = features[1] / 10
    # print('backbone: ', next(forward_modules["backbone"].parameters()).device)f
    return features

def _embed(features, forward_modules, patch_maker, provide_patch_shapes=False):#, evaluation=False):
    """Returns feature embeddings for images."""
    
    # apply patchify
    # t_1 = time.perf_counter()
    # print('_embed')
    # print("features intern 1: mean ", features[0].mean())
    # print("features intern 1: std ", features[0].std())  
    # print("features intern 1: min ", features[0].min())
    # print("features intern 1: max ", features[0].max()) 
    
    # print("features intern 2: mean ", features[1].mean())
    # print("features intern 2: std ", features[1].std())  
    # print("features intern 2: min ", features[1].min())
    # print("features intern 2: max ", features[1].max())
    features = [
        patch_maker.patchify(x, return_spatial_info=True) for x in features
    ]

    features, patch_shapes = interpolate_bilinear_after_patchify(features=features)
    # print("features intern 3: ", features[0].shape)
    # As different feature backbones & patching provide differently
    # sized features, these are brought into the correct form here.
    features = forward_modules["preprocessing"](features) # pooling each feature to same channel and stack together; torch.Size([784, 2, 1536])
    # print("features intern 4: ", features.shape)
    features = forward_modules["preadapt_aggregator"](features) # further pooling; torch.Size([784, 1536]) --> same shape as the patchcore method
    # print("features intern 5: ", features.shape)
    if provide_patch_shapes:
        return features, patch_shapes
    else:
        return features


### helper functions
def interpolate_bilinear_after_patchify(features):
    """Interpolates features to the same size.
    Upsamples all feature maps to the size of the largest feature map.
    Uses bilinear interpolation.
    
    Input: list of tuples. One tuple for each layer. Each tuple contains the features and the patch shape of the layer. Features itself have shape (1, C, W, H)
    """
    patch_shapes = [x[1] for x in features]
    features = [x[0] for x in features]
    ref_num_patches = patch_shapes[0] # size of the larger grid, e.g. L2: 28x28; this will later be taken to upsample features from Layers deeper in the net

    for i in range(1, len(features)):
        _features = features[i]
        patch_dims = patch_shapes[i]

        # TODO(pgehler): Add comments
        _features = _features.reshape(
            _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:] #torch.Size([1, 14, 14, 1024, 3, 3])
        ) # shape: (1, 14, 14, 1024, 3, 3)
        _features = _features.permute(0, -3, -2, -1, 1, 2) # torch.Size([1, 1024, 3, 3, 14, 14])
        perm_base_shape = _features.shape
        _features = _features.reshape(-1, *_features.shape[-2:]) #([9216, 14, 14])
        _features = F.interpolate(
            _features.unsqueeze(1), # torch.Size([9216, 1, 14, 14]) 9126 = 1024 * 3 * 3 --> Why?
            
            size=(ref_num_patches[0], ref_num_patches[1]),
            mode="bilinear",
            align_corners=False,
        )
        _features = _features.squeeze(1) # torch.Size([9216, 28, 28])
        _features = _features.reshape(
            *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
        ) # torch.Size([1, 1024, 3, 3, 28, 28]) --> back in patches again
        _features = _features.permute(0, -2, -1, 1, 2, 3)
        _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
        features[i] = _features
    features = [x.reshape(-1, *x.shape[-3:]) for x in features]
    
    return features, patch_shapes


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=1):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        # was_numpy = False
        # if isinstance(x, np.ndarray):
        #     was_numpy = True
        #     x = torch.from_numpy(x)
        # while x.ndim > 2:
        #     x = torch.max(x, dim=-1).values
        # if x.ndim == 2:
        #     if self.top_k > 1:
        #         x = torch.topk(x, self.top_k, dim=1).values.mean(1)
        #     else:
        #         x = torch.max(x, dim=1).values
        # if was_numpy:
        #     return x.numpy()
        # return x
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
    
def alternative_pooling(features, batch_size):
        '''
        TODO
        '''
        selected_features = []
        
        for _, feature in enumerate(features):
            ####
            # insert dim reduction here TODO 
            # before pooling
            ####
            # pooled_features = adaptive_pooling(feature, self.pooling_strategy)#torch.nn.AvgPool2d(3, 1, 1)(feature) # TODO replace with adaptive pooling
            # if type(self.pooling_strategy) == list:
                # for strategy in self.pooling_strategy:
                    # pooled_feature = adaptive_pooling(feature, strategy)
                    # selected_features.append(pooled_feature)
            # else:
            pooled_feature = torch.nn.AvgPool2d(3, 1, 1)(feature)#adaptive_pooling(feature, 'default')
            selected_features.append(pooled_feature)
            ####
            # insert dim reduction here TODO 
            # after pooling
            ####
            # selected_features.append(pooled_features)
        
        concatenated_features = embedding_concat_frame(embeddings=selected_features, cuda_active=False)
        
        # if self.pool_depth[0]:
            # print
        # print("concatenated_features: ", concatenated_features.shape)
        batch_size = concatenated_features.shape[0]
        if batch_size == 1:
            flattened_features = np.array(reshape_embedding(np.array(concatenated_features)))
        else:
            flattened_features = np.array([np.array(reshape_embedding(np.array(concatenated_features[k,...].unsqueeze(0)))) for k in range(batch_size)])
        # print("flattened_features: ", flattened_features.shape)
        return torch.from_numpy(flattened_features)