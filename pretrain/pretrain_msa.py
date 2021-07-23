# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, Knowledge Engineering Group (KEG), Tsinghua University
# Modified by Jiezhong Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain MSA"""

import torch
import torch.nn.functional as F
from megatron import get_args, get_tokenizer
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.msa_dataset import build_train_valid_test_datasets
from megatron.model import BertModel, BertModelFirstStage, BertModelIntermediateStage, BertModelLastStage
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import get_tape_masks_and_position_ids
from megatron.model.bert_model import bert_extended_attention_mask

SPLIT_TOKEN_ID = 32
EOM_TOKEN_ID = 33

# IS_FAKE = True
IS_FAKE = False

def msa_preprocess(msa_string):
    pass


def model_provider():
    """Build the model."""

    print_rank_0('building TAPE model ...')

    args = get_args()
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        # Determine model based on position of stage in pipeline.
        if mpu.is_pipeline_first_stage():
            model = BertModelFirstStage(
                num_tokentypes=0)
        elif mpu.is_pipeline_last_stage():
            model = BertModelLastStage(
                num_tokentypes=0,
                add_binary_head=False,
                parallel_output=True)
        else:
            model = BertModelIntermediateStage(
                num_tokentypes=0)
    else:
        model = BertModel(
            num_tokentypes=0,
            add_binary_head=False,
            parallel_output=True)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    tokenizer = get_tokenizer()
    # Items and their type.

    # keys = ['text', 'labels', 'loss_mask', 'padding_mask']
    # NOTE: add msa_depth and msa_length
    keys = ['text', 'labels', 'loss_mask', 'padding_mask']
    # keys = ['text', 'labels', 'loss_mask', 'padding_mask', 'msa_depth', 'msa_length']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    # print(data_iterator)

    data_b = mpu.broadcast_data(keys, data, datatype)




    # start: fake tokens
    if IS_FAKE:
        # Unpack.
        tokens = data_b['text'].long()
        # print("tokens in pretrain msa")
        # tokens = tokens[0]
        # print(tokens, tokens.shape)
        # print(tokens[:257], len(tokens[:257]))
        # print(tokens[-513:], len(tokens[-513:]))
        # print(data_b)

        # TODO: I found that the sequences are concatenated, and seperated by [CLS]
        # And I discussed with jiayun, this part data_iter should be modified

        # print('tokens [CLS] count', sum(tokens[0] == tokenizer.cls))
        # tokens[CLS] count tensor(51, device='cuda:0')

        args = get_args()
        import random
        # if random.randint(0 ,1) == 0:
        if True:
            msa_depth = args.msa_depth
            msa_length = args.msa_length
        else:
            msa_depth = args.msa_depth * 2
            msa_length = args.msa_length // 2

        loss_mask = data_b['loss_mask'].float()
        lm_labels = data_b['labels'].long()
        padding_mask = data_b['padding_mask'].long()
        # print(padding_mask[0])
        FAKE_TOKEN_ID = 16
        device = tokens.device

        tokens = torch.ones((msa_depth, msa_length), device=device) * FAKE_TOKEN_ID

        tokens[:, 0] = tokenizer.cls
        tokens = tokens.long()
        # tensor([[2., 16., 16., ..., 16., 16., 16.],
        #         [2., 16., 16., ..., 16., 16., 16.],
        #         [2., 16., 16., ..., 16., 16., 16.],
        #         ...,
        #         [2., 16., 16., ..., 16., 16., 16.],
        #         [2., 16., 16., ..., 16., 16., 16.],
        #         [2., 16., 16., ..., 16., 16., 16.]])
        msa_shape = tokens.shape
        masking_bool_mat = torch.zeros(msa_shape, dtype=torch.bool)
        for i in range(len(tokens)):
            for j in range(1, len(tokens[0])):
                if torch.rand(1).item() < 0.15:
                    masking_bool_mat[i][j] = True
        # print('masked bool mat', masking_bool_mat)
        # masked bool mat
        # tensor([[False, False, False, ..., False, False, False],
        #         [False, False, False, ..., False, False, False],
        #         [False, True, False, ..., False, False, False],
        #         ...,
        #         [False, False, False, ..., False, False, False],
        #         [False, True, False, ..., False, False, False],
        #         [False, False, False, ..., False, False, False]])

        # print('masked ', sum(sum(masking_bool_mat.long())), ', total ', masking_bool_mat.numel()) # 2-dim, sum twice
        # masked tensor(2368), total 16384

        loss_mask = masking_bool_mat.float()
        lm_labels = -torch.ones(msa_shape).long()
        lm_labels[masking_bool_mat] = FAKE_TOKEN_ID
        padding_mask = torch.ones(tokens.shape)

        # print(tokens)
        # print(loss_mask)
        # print(lm_labels)
        # print(padding_mask)

        # Get the masks and position ids.
        attention_mask, position_ids = get_tape_masks_and_position_ids(
            tokens,
            tokenizer.cls,
            reset_position_ids=True,
            reset_attention_mask=True)

        attention_mask = torch.ones(msa_shape).bool()
        # tensor([[True, True, True, ..., True, True, True],
        #         [True, True, True, ..., True, True, True],
        #         [True, True, True, ..., True, True, True],
        #         ...,
        #         [True, True, True, ..., True, True, True],
        #         [True, True, True, ..., True, True, True],
        #         [True, True, True, ..., True, True, True]])
        # position_ids = torch.arange(msa_length).repeat(msa_depth, 1)
        position_ids = torch.arange(msa_length).repeat(msa_depth, 1)
        # tensor([[0, 1, 2, ..., 509, 510, 511],
        #         [0, 1, 2, ..., 509, 510, 511],
        #         [0, 1, 2, ..., 509, 510, 511],
        #         ...,
        #         [0, 1, 2, ..., 509, 510, 511],
        #         [0, 1, 2, ..., 509, 510, 511],
        #         [0, 1, 2, ..., 509, 510, 511]])

        # note, Change in batch, to fit model input, no longer support batch
        # msa_shape = (-1, msa_depth, msa_length)
        # tokens = tokens.reshape(msa_shape)
        # loss_mask = loss_mask.reshape(msa_shape)
        # loss_mask = loss_mask.reshape(msa_shape)
        # lm_labels = lm_labels.reshape(msa_shape)
        # padding_mask = padding_mask.reshape(msa_shape)
        device = tokens.device
        # print('tokens.shape ', tokens.shape)
        for it in [tokens, loss_mask, lm_labels, padding_mask, attention_mask, position_ids]:
            print(it.shape)
        
        return tokens.to(device), loss_mask.to(device), lm_labels.to(device), padding_mask.to(device), attention_mask.to(device), position_ids.to(device)
        # return tokens, loss_mask, lm_labels, padding_mask, attention_mask, position_ids
    else:
        # Unpack.
        tokens = data_b['text'].long()
        loss_mask = data_b['loss_mask'].float()
        lm_labels = data_b['labels'].long()
        padding_mask = data_b['padding_mask'].long()

        # NOTE: msa_xxx are tensors
        msa_length = data['msa_length'].long().item() // 2
        msa_depth = data['msa_depth'].long().item() * 2
        # print(msa_length, msa_depth)
        # Get the masks and postition ids.
        # attention_mask, position_ids = get_tape_masks_and_position_ids(
        #     tokens,
        #     tokenizer.cls,
        #     reset_position_ids=True,
        #     reset_attention_mask=True)
        # attention_mask = attention_mask.squeeze(0).squeeze(0)
        # attention_mask = torch.zeros((msa_depth[0], msa_length[0]))


        attention_mask = torch.zeros((msa_depth, msa_length)).to(tokens.device)
        position_ids = torch.arange(msa_length).repeat(msa_depth, 1).to(tokens.device)
        tokens = tokens.reshape((msa_depth, msa_length))
        loss_mask =  loss_mask.reshape((msa_depth, msa_length))
        lm_labels =  lm_labels.reshape((msa_depth, msa_length))
        padding_mask =  padding_mask.reshape((msa_depth, msa_length))
        attention_mask =  attention_mask.reshape((msa_depth, msa_length))
        position_ids =  position_ids.reshape((msa_depth, msa_length))

        # for it in [tokens, loss_mask, lm_labels, padding_mask, attention_mask, position_ids]:
        #     print(it.shape)
        return tokens, loss_mask, lm_labels, padding_mask, attention_mask, position_ids

# def get_batch(data_iterator):
#     """Build the batch."""

#     tokenizer = get_tokenizer()
#     # Items and their type.
#     keys = ['text', 'labels', 'loss_mask', 'padding_mask']
#     datatype = torch.int64

#     # Broadcast data.
#     if data_iterator is not None:
#         data = next(data_iterator)
#     else:
#         data = None
#     data_b = mpu.broadcast_data(keys, data, datatype)

#     # Unpack.
#     tokens = data_b['text'].long()
#     loss_mask = data_b['loss_mask'].float()
#     lm_labels = data_b['labels'].long()
#     padding_mask = data_b['padding_mask'].long()

#     print(tokens, tokens.shape)

#     # Get the masks and postition ids.
#     attention_mask, position_ids = get_tape_masks_and_position_ids(
#         tokens,
#         tokenizer.cls,
#         reset_position_ids=True,
#         reset_attention_mask=True)
#     attention_mask = attention_mask.squeeze(0).squeeze(0)
#     return tokens, loss_mask, lm_labels, padding_mask, attention_mask, position_ids


def forward_step(data_iterator, model, input_tensor):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, loss_mask, lm_labels, padding_mask, attention_mask, position_ids \
        = get_batch(data_iterator)
    timers('batch-generator').stop()
    # print('forward step: shapes')
    # for item in [tokens, loss_mask, lm_labels, padding_mask, attention_mask, position_ids]:
    #     print(item.device)
    # all output: torch.Size([32, 512])

    # TODO: restore extended
    # extended_attention_mask = bert_extended_attention_mask(padding_mask) + attention_mask
    extended_attention_mask = attention_mask

    # Forward pass through the model.
    if mpu.is_pipeline_first_stage():
        assert input_tensor is None
        if mpu.is_pipeline_last_stage():
            output_tensor = model(tokens, extended_attention_mask, tokentype_ids=None,
                                  lm_labels=lm_labels, position_ids=position_ids)
        else:
            output_tensor = model(tokens, extended_attention_mask, tokentype_ids=None)
    elif mpu.is_pipeline_last_stage():
        assert input_tensor is not None
        output_tensor = model(input_tensor, extended_attention_mask, lm_labels=lm_labels)
    else:
        assert input_tensor is not None
        output_tensor = model(input_tensor, extended_attention_mask, position_ids=position_ids)

    if mpu.is_pipeline_last_stage():
        lm_loss_, _ = output_tensor

        lm_loss_ = lm_loss_.float()
        loss_mask = loss_mask.float()
        lm_loss = torch.sum(
            lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

        loss = lm_loss

        averaged_losses = average_losses_across_data_parallel_group([lm_loss,])

        return loss, {'lm loss': averaged_losses[0]}
    return output_tensor


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for TAPE ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating MSA datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    print('starting')
    import os

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
