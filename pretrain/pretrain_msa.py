# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, Knowledge Engineering Group (KEG), Tsinghua University
# Modified by Yijia Xiao
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
import os

import torch
import torch.nn.functional as F
from megatron import get_args, get_tokenizer
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.msa_dataset import build_train_valid_test_datasets
# from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import BertModel, BertModelFirstStage, BertModelIntermediateStage, BertModelLastStage
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import get_tape_masks_and_position_ids
from megatron.model.bert_model import bert_extended_attention_mask
import torch.nn.functional as F

# SPLIT_TOKEN_ID = 32
# EOM_TOKEN_ID = 33

# IS_FAKE = True
IS_FAKE = False


def msa_preprocess(msa_string):
    pass


class Counter(object):
    __count = 0

    @classmethod
    def get_count(cls):
        return cls.__count

    @classmethod
    def add_count(cls):
        cls.__count += 1


def model_provider():
    """Build the model."""

    print_rank_0('building MSA model ...')

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

    keys = ['text', 'labels', 'loss_mask', 'padding_mask', 'actual_msa_depth', 'actual_msa_length']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # start: fake tokens
    if IS_FAKE:
        tokens = data_b['text'].long()
        from megatron import get_args
        args = get_args()
        import random

        if random.randint(0 ,1) == 0:
            msa_depth = args.msa_depth
            msa_length = args.msa_length
        else:
            msa_depth = 8
            msa_length = 32

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
        device = tokens.device
        return tokens.to(device), loss_mask.to(device), lm_labels.to(device), padding_mask.to(device), attention_mask.to(device), position_ids.to(device)

    else:
        # Unpack.
        tokens = data_b['text'].long()
        loss_mask = data_b['loss_mask'].float()
        lm_labels = data_b['labels'].long()
        padding_mask = data_b['padding_mask'].long()
        actual_msa_length = data_b['actual_msa_length'].long().item()
        actual_msa_depth = data_b['actual_msa_depth'].long().item()

        from megatron import get_args
        args = get_args()
        MAX_MSA_DEPTH = args.msa_depth
        MAX_MSA_LENGTH = args.msa_length

        tokens = tokens.reshape((actual_msa_depth, actual_msa_length))
        # print('tokens.shape1', tokens.shape)
        # if actual_msa_depth < MAX_MSA_DEPTH:
        #     actual_msa_depth = MAX_MSA_DEPTH
        #     tokens = tokens[0].repeat((MAX_MSA_DEPTH, 1))
        # print('tokens.shape2', tokens.shape)

        tokens = F.pad(tokens, (0, MAX_MSA_LENGTH - actual_msa_length, 0,
                                MAX_MSA_DEPTH - actual_msa_depth), 'constant', 0)

        """
        In[31]: t4d
        Out[31]:
        tensor([[7.6094e+31, 4.5649e-41, 8.0253e+31],
                [4.5649e-41, 8.0256e+31, 4.5649e-41],
                [8.0250e+31, 4.5649e-41, 8.0251e+31]])

        In[32]: F.pad(t4d, (0, 1, 0, 2), "constant", 0)
        Out[32]:
        tensor([[7.6094e+31, 4.5649e-41, 8.0253e+31, 0.0000e+00],
                [4.5649e-41, 8.0256e+31, 4.5649e-41, 0.0000e+00],
                [8.0250e+31, 4.5649e-41, 8.0251e+31, 0.0000e+00],
                [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])
        """

        loss_mask = F.pad(loss_mask.reshape((actual_msa_depth, actual_msa_length)),
                          (0, MAX_MSA_LENGTH - actual_msa_length, 0, MAX_MSA_DEPTH - actual_msa_depth), 'constant', 0)
        lm_labels = F.pad(lm_labels.reshape((actual_msa_depth, actual_msa_length)),
                          (0, MAX_MSA_LENGTH - actual_msa_length, 0, MAX_MSA_DEPTH - actual_msa_depth), 'constant', -1)


        # MSA can see all tokens
        row_attention_mask = torch.zeros((MAX_MSA_DEPTH, MAX_MSA_LENGTH), device=tokens.device)
        row_attention_mask[actual_msa_depth:, :] = 1
        row_attention_mask[:, actual_msa_length:] = 1

        col_attention_mask = torch.zeros((MAX_MSA_LENGTH, MAX_MSA_DEPTH), device=tokens.device)
        col_attention_mask[actual_msa_length:, :] = 1
        col_attention_mask[:, actual_msa_depth:] = 1

        position_ids = torch.arange(MAX_MSA_LENGTH, device=tokens.device).repeat(MAX_MSA_DEPTH, 1)

        return tokens, loss_mask, lm_labels, row_attention_mask.bool(), col_attention_mask.bool(), position_ids


def forward_step(data_iterator, model, input_tensor):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    if IS_FAKE:
        # Get the batch.
        timers('batch-generator').start()
        tokens, loss_mask, lm_labels, padding_mask, attention_mask, position_ids \
            = get_batch(data_iterator)
        timers('batch-generator').stop()

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
    else:
        # Get the batch.
        timers('batch-generator').start()
        tokens, loss_mask, lm_labels, row_attention_mask, col_attention_mask, position_ids \
            = get_batch(data_iterator)
        timers('batch-generator').stop()

        # TODO: restore extended
        # extended_attention_mask = bert_extended_attention_mask(padding_mask) + attention_mask
        # extended_attention_mask = row_attention_mask

        # row_attention_mask = bert_extended_attention_mask(row_attention_mask_)
        # col_attention_mask = bert_extended_attention_mask(col_attention_mask_)

        # Forward pass through the model.
        if mpu.is_pipeline_first_stage():
            assert input_tensor is None
            if mpu.is_pipeline_last_stage():
                output_tensor = model(tokens, row_attention_mask, col_attention_mask, tokentype_ids=None,
                                    lm_labels=lm_labels, position_ids=position_ids)
            else:
                output_tensor = model(tokens, row_attention_mask, col_attention_mask, tokentype_ids=None)
        elif mpu.is_pipeline_last_stage():
            assert input_tensor is not None
            output_tensor = model(input_tensor, row_attention_mask, col_attention_mask, lm_labels=lm_labels)
        else:
            assert input_tensor is not None
            output_tensor = model(input_tensor, row_attention_mask, col_attention_mask, position_ids=position_ids)

        if mpu.is_pipeline_last_stage():


            # TODO: add attention weights
            # REMOVE BEGIN
            # lm_loss_, _ = output_tensor
            # REMOVE END

            # ADD BEGIN
            # import os
            # print('pid = ', str(os.getpid()) * 10)
            _post_language_model_processing, attn_layers_dict = output_tensor
            lm_loss_, _ = _post_language_model_processing

            attn_path = get_args().attention_save
            attn_save_freq = get_args().attn_save_freq
            # lm_output, attn_layers = self.language_model(*args, **kwargs)
            idx = Counter.get_count()
            pid = os.getpid()
            # os.system(f'mkdir -p {attn_path}/{pid}')
            if idx % attn_save_freq == 0:
                print_rank_0(f'saving to {attn_path}/attn_weights_{idx:09d}.pt')
                torch.save(attn_layers_dict, f'{attn_path}/attn_weights_{idx:09d}.pt')
            Counter.add_count()
            # ADD END


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
                 'for MSA ...')
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
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
