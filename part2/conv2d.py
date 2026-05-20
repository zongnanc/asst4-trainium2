import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    # assert (
    #     in_channels_ == in_channels and out_channels_ == out_channels
    # ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # assert in_channels % 128 == out_channels % 128 == 0
    # assert nl.tile_size.gemm_moving_fmax >= out_width

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    c_in_pmax = 128
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_in_pmax

    # Pre-load and pre-transpose all weight tiles into w_all
    n_w_tiles = n_tiles_c_out * n_tiles_c_in * filter_height * filter_width
    w_all = nl.ndarray((c_in_pmax, n_w_tiles, c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)

    # Strides for w_all indexing
    w_stride_oc = n_tiles_c_in * filter_height * filter_width
    w_stride_ic = filter_height * filter_width
    w_stride_i = filter_width

    # Pre-load bias
    bias_all = nl.ndarray((c_in_pmax, n_tiles_c_out), dtype=nl.float32, buffer=nl.sbuf)

    for oc in nl.affine_range(n_tiles_c_out):
        # Load bias for this oc tile alongside weights
        bias_raw = nl.ndarray((c_in_pmax, 1), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=bias[oc*c_in_pmax:(oc+1)*c_in_pmax], dst=bias_raw)
        bias_f32 = nl.copy(bias_raw, dtype=nl.float32)
        bias_all[:, oc:oc+1] = bias_f32

        for i in nl.affine_range(filter_height):
            for j in nl.affine_range(filter_width):
                # Load all in_channels for this (oc, i, j) in one DMA: (128, in_channels)
                w_full = nl.ndarray((c_in_pmax, in_channels), dtype=W.dtype, buffer=nl.sbuf)
                nisa.dma_copy(src=W[oc*c_in_pmax:(oc+1)*c_in_pmax, :, i, j], dst=w_full)

                # Transpose each 128x128 sub-tile
                for ic in nl.affine_range(n_tiles_c_in):
                    w_psum = nisa.nc_transpose(w_full[:, ic*c_in_pmax:(ic+1)*c_in_pmax])
                    w_t_sbuf = nisa.tensor_copy(w_psum)
                    w_idx = oc * w_stride_oc + ic * w_stride_ic + i * w_stride_i + j
                    w_all[:, w_idx, :] = w_t_sbuf

    # Multi-row tiling
    TILE_H = 512 // out_width
    if TILE_H < 1:
        TILE_H = 1
    # Ensure TILE_H is divisible by pool_size for maxpool
    while out_height % TILE_H != 0 or TILE_H % pool_size != 0:
        TILE_H -= 1
    TILE_HW = TILE_H * out_width
    n_row_tiles = out_height // TILE_H

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for row_tile in nl.affine_range(n_row_tiles):
            # Pre-load input rows for this spatial tile
            input_tile_height = TILE_H + filter_height - 1
            X_tile = nl.ndarray((c_in_pmax, n_tiles_c_in, input_tile_height, input_width),
                                dtype=X.dtype, buffer=nl.sbuf)
            for ic in nl.affine_range(n_tiles_c_in):
                nisa.dma_copy(src=X[b, ic*c_in_pmax:(ic+1)*c_in_pmax,
                                   row_tile * TILE_H : row_tile * TILE_H + input_tile_height,
                                   0:input_width],
                             dst=X_tile[:, ic, 0:input_tile_height, 0:input_width])

            for oc in nl.affine_range(n_tiles_c_out):
                conv_psum = nl.zeros((c_in_pmax, TILE_HW), dtype=nl.float32, buffer=nl.psum)

                for ic in nl.affine_range(n_tiles_c_in):
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            input_patch = nl.ndarray((c_in_pmax, TILE_HW), dtype=X.dtype, buffer=nl.sbuf)
                            for h in nl.affine_range(TILE_H):
                                input_patch[:, h * out_width : (h + 1) * out_width] = X_tile[:, ic, h + i, j:j + out_width]

                            w_idx = oc * w_stride_oc + ic * w_stride_ic + i * w_stride_i + j
                            conv_psum += nisa.nc_matmul(w_all[:, w_idx, :], input_patch)

                conv_sbuf = nisa.tensor_copy(conv_psum)
                conv_bias = nisa.tensor_scalar(conv_sbuf, nl.add, bias_all[:, oc:oc+1])
                conv_out = nl.copy(conv_bias, dtype=X.dtype)

                if pool_size == 1:
                    # No pooling — store directly
                    for h in nl.affine_range(TILE_H):
                        nisa.dma_copy(src=conv_out[:, h * out_width : (h + 1) * out_width],
                                     dst=X_out[b, oc*c_in_pmax:(oc+1)*c_in_pmax,
                                               row_tile * TILE_H + h, :])
                else:
                    # Maxpool with pool_size=2
                    # conv_out shape: (128, TILE_H * out_width) = (128, TILE_H, out_width) conceptually
                    tile_pool_h = TILE_H // pool_size

                    # Step 1: Row-wise max (max of row pairs)
                    row_pooled = nl.ndarray((c_in_pmax, tile_pool_h, out_width), dtype=X.dtype, buffer=nl.sbuf)
                    for ph in nl.affine_range(tile_pool_h):
                        row0 = conv_out[:, (ph * 2) * out_width : (ph * 2 + 1) * out_width]
                        row1 = conv_out[:, (ph * 2 + 1) * out_width : (ph * 2 + 2) * out_width]
                        row_pooled[:, ph, :] = nisa.tensor_tensor(row0, row1, op=nl.maximum)

                    # Step 2: Column-wise max (max of column pairs)
                    # row_pooled[:, ph, :] has shape (128, out_width)
                    # Take max of even/odd columns
                    pooled = nl.ndarray((c_in_pmax, tile_pool_h, out_pool_width), dtype=X.dtype, buffer=nl.sbuf)
                    for ph in nl.affine_range(tile_pool_h):
                        # Reshape row to (128, out_pool_width, 2) and reduce
                        row_data = row_pooled[:, ph, :]  # (128, out_width)
                        col_pairs = nl.ndarray((c_in_pmax, out_pool_width, pool_size), dtype=X.dtype, buffer=nl.sbuf)
                        for pw in nl.affine_range(out_pool_width):
                            col_pairs[:, pw, 0:1] = row_data[:, pw*2 : pw*2+1]
                            col_pairs[:, pw, 1:2] = row_data[:, pw*2+1 : pw*2+2]
                        pooled[:, ph, :] = nisa.tensor_reduce(data=col_pairs, op=nl.maximum, axis=(2,))

                    # Store pooled result
                    for ph in nl.affine_range(tile_pool_h):
                        nisa.dma_copy(src=pooled[:, ph, :],
                                     dst=X_out[b, oc*c_in_pmax:(oc+1)*c_in_pmax,
                                               row_tile * tile_pool_h + ph, :])

    return X_out







    # # Various tiling dimensions (You may want to define more of them)
    # c_in_pmax = nl.tile_size.pmax
    # n_tiles_c_in = in_channels // c_in_pmax

    # #X_re = X.reshape((batch_size, in_channels, out_height * out_width))

    # # 1. Input shape is [batch][input_c][input_h][input_w], iterate by batch, then pick size of image same as output size, and reshape to [input_c][out_height * out_width]
    
    
    # # 2. Filter shape is [output_c][input_c][filter_h][filter_w], we do iterate by [filter_h][filter_w]

    # # matmul [output_c][input_c] * [input_c][shifted: out_height * out_width] = [output_c][out_height * out_width], then with 2. we get X_conv_out

    # # preload bias to sbuf
    # bias_sbuf = nl.ndarray((128, out_channels // 128), dtype=bias.dtype, buffer=nl.sbuf)
    # for m in nl.affine_range(out_channels // 128):
    #     nisa.dma_copy(src=bias[m*128 : (m+1)*128], dst=bias_sbuf)

    # W_sbuf = nl.ndarray((c_in_pmax, out_channels // 128, in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    # for oc in nl.affine_range(out_channels // 128):
    #     for ic in nl.affine_range(in_channels):
    #         nisa.dma_copy(src=W[oc*c_in_pmax:(oc+1)*c_in_pmax, ic, :, :], dst=W_sbuf[:, oc, ic, :, :])


    # # Process the images in batches
    # for b in nl.affine_range(batch_size):
    #     # tiling by engine constraint
    #     for oc in nl.affine_range(out_channels // c_in_pmax):
    #         # Load bias tile to SBUF
    #         bias_sbuf = nl.ndarray((c_in_pmax, 1), dtype=bias.dtype, buffer=nl.sbuf)
    #         nisa.dma_copy(src=bias[oc*c_in_pmax:(oc+1)*c_in_pmax], dst=bias_sbuf)
    #         bias_f32 = nl.copy(bias_sbuf, dtype=nl.float32)
    #         for row in nl.affine_range(out_height):
    #             # allocate psum for conv_out
    #             conv_psum = nl.zeros((c_in_pmax, out_width), dtype=nl.float32, buffer=nl.psum)
    #             # Process the b_th image by filter
    #             for filter_i in nl.affine_range(filter_height):
    #                 for filter_j in nl.affine_range(filter_width):
    #                     for ic in nl.affine_range(in_channels // c_in_pmax):
    #                         # allocate sbuf for input_row: [128-input_channels][out_width]
    #                         input_sbuf = nl.ndarray((c_in_pmax, out_width), dtype=X.dtype, buffer=nl.sbuf)
    #                         # load hbm to sbuf
    #                         nisa.dma_copy(src=X[b, ic*c_in_pmax : (ic+1)*c_in_pmax, row + filter_i, filter_j : filter_j + out_width], dst=input_sbuf)

    #                         # allocate sbuf for filter: 128 * 128
    #                         filter_sbuf = nl.ndarray((c_in_pmax, c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
    #                         nisa.dma_copy(src=W[oc*c_in_pmax : (oc + 1)*c_in_pmax, ic*c_in_pmax : (ic+1)*c_in_pmax, filter_i, filter_j], dst=filter_sbuf)
    #                         filter_psum = nisa.nc_transpose(filter_sbuf)
    #                         filter_t_sbuf = nisa.tensor_copy(filter_psum)

    #                         # matmul -> 128 oc * out_width
    #                         conv_psum += nisa.nc_matmul(filter_t_sbuf, input_sbuf)

    #             # copy conv to sbuf
    #             conv_sbuf = nisa.tensor_copy(conv_psum)

    #             # add bias
    #             conv_bias = nisa.tensor_scalar(conv_sbuf, nl.add, bias_f32)

    #             # copy result back to HBM
    #             nisa.dma_copy(src=conv_bias, dst=X_out[b, oc*c_in_pmax : (oc+1)*c_in_pmax, row, :])

    #     # raise RuntimeError("Please fill your implementation of computing convolution"
    #     #                    " of X[b] with the weights W and bias b, followed by a"
    #     #                    " maxpool and store the result in X_out[b]")

    # # X_conv_out, shape: [(Input_H - Weight_H + 1) * (Input_W - Weight_W + 1)] * Output_Channels
    # # each batch image
    # #   X_conv_out[xth][:]
    # #   each Weight_H[i]W[j] - shape: W[h][w][channel_i][channel_o]
    # #       Input_Channeled: shape: (Input_H * Input_W) * Output_Channels
    # #       Input_Channeled = (Input_H * Input_W) * Input_Channels -- matmul - Input_Channels * Output_Channels = (Input_H * Input_W) * Output_Channels
    # #       X_conv_out[xth][:] += weight
    # #        

    # # X_conv_out = nl.ndarray(shape=(batch_size, out_channels, input_height - filter_height + 1, input_width - filter_width + 1)..)

    # return X_out

