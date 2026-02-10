#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


class LoRALayer:
    def __init__(
        self,
        r: int,
        nr_lora: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha

        self.nr_lora = nr_lora

        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=merge_weights,
        )
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x,
                    self.lora_A.transpose(0, 1),
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (
                    self.lora_dropout(x)
                    @ self.lora_A.transpose(0, 1)
                    @ self.lora_B.transpose(0, 1)
                ) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        nr_lora=8,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            nr_lora=nr_lora,
        )
        assert (
            out_features % len(enable_lora) == 0
        ), "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.ParameterList()
            self.lora_B = nn.ParameterList()
            self.scaling = list()

            me_in_features = in_features // self.nr_lora
            me_out_features = out_features // self.nr_lora
            me_r = r // self.nr_lora

            self.me_in_features = me_in_features
            self.me_out_features = me_out_features
            self.me_r = me_r

            for _ in range(self.nr_lora):
                # for melora
                self.lora_A.append(
                    nn.Parameter(
                        self.weight.new_zeros((me_r * sum(enable_lora), me_in_features))
                    )
                )
                self.lora_B.append(
                    nn.Parameter(
                        self.weight.new_zeros(
                            (
                                me_out_features // len(enable_lora) * sum(enable_lora),
                                me_r,
                            )
                        )
                    )
                )
                self.scaling.append(self.lora_alpha / me_r)

            # self.lora_A = nn.Parameter(
            #     self.weight.new_zeros((r * sum(enable_lora), in_features))
            # )
            # self.lora_B = nn.Parameter(
            #     self.weight.new_zeros(
            #         (out_features // len(enable_lora) * sum(enable_lora), r)
            #     )
            # )  # weights for Conv1D with groups=sum(enable_lora)
            # self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(
                len(enable_lora), -1
            )  # (3d,) -> (3,d)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)  # (3d,)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.nr_lora):
                nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[i])
            # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        # x -> (B,N,2d)
        # print(x.shape)
        result = x.new_zeros((*x.shape[:-1], self.out_features))  # result -> B,N,3d
        # result = x.new_zeros((self.out_features, *x.shape[1:]))
        result = result.view(-1, self.out_features)  # BN, 3d
        # result = result.view(self.out_features, -1)
        # print(result.shape)
        # print(self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )  # (BN,2d)
        # result[self.lora_ind, :] = x.reshape(
        #      self.out_features // len(self.enable_lora) * sum(self.enable_lora), -1
        # )  # (BN,2d)
        # print(result.shape)
        return result.view((*x.shape[:-1], self.out_features))

        # print(sum(x - result[self.lora_ind,:]))
        # print(result[1,:])
        # return result

    def zero_pad_weight(self, x):
        # x -> (B,N,2d)
        # print(x.shape)
        # result = x.new_zeros((*x.shape[:-1], self.out_features)) # result -> B,N,3d
        result = x.new_zeros((self.out_features, *x.shape[1:]))
        # result = result.view(-1, self.out_features)  # BN, 3d
        result = result.view(self.out_features, -1)
        # print(result.shape)
        # print(self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        # result[:, self.lora_ind] = x.reshape(
        #     -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        # )  # (BN,2d)
        result[self.lora_ind, :] = x.reshape(
            self.out_features // len(self.enable_lora) * sum(self.enable_lora), -1
        )  # (BN,2d)
        # print(result.shape)
        # return result.view((*x.shape[:-1], self.out_features))

        # print(sum(x - result[self.lora_ind,:]))
        # print(result[1,:])
        return result

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    delta_w = list()
                    for i in range(self.nr_lora):
                        delta_w.append(
                            F.conv1d(
                                self.lora_A[i].data.unsqueeze(0),
                                self.lora_B[i].data.unsqueeze(-1),
                                groups=sum(self.enable_lora),
                            ).squeeze(0)
                            * self.scaling[i]
                        )
                    # delta_w = F.conv1d(
                    #     self.lora_A.data.unsqueeze(0),
                    #     self.lora_B.data.unsqueeze(-1),
                    #     groups=sum(self.enable_lora),
                    # ).squeeze(0)
                    delta_w = torch.cat(delta_w, dim=1)
                    # self.weight.data -= self.zero_pad_weight(T(delta_w * self.scaling))
                    self.weight.data -= self.zero_pad_weight(T(delta_w))
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    # delta_w = list()
                    nr_enable = sum(self.enable_lora)

                    delta_w = torch.zeros(
                        self.out_features
                        // len(self.enable_lora)
                        * sum(self.enable_lora),
                        self.in_features,
                        dtype=self.lora_A[0].dtype,
                        device=self.lora_A[0].device,
                    )
                    for i in range(self.nr_lora):
                        tmp_dw = (
                            F.conv1d(
                                self.lora_A[i].data.unsqueeze(0),
                                self.lora_B[i].data.unsqueeze(-1),
                                groups=sum(self.enable_lora),
                            ).squeeze(0)
                            * self.scaling[i]
                        )

                        for j in range(nr_enable):
                            delta_w[
                                i * (tmp_dw.shape[0] // nr_enable)
                                + j
                                * delta_w.shape[0]
                                // nr_enable : (i + 1)
                                * (tmp_dw.shape[0] // nr_enable)
                                + j * delta_w.shape[0] // nr_enable,
                                i * self.me_in_features : (i + 1) * self.me_in_features,
                            ] = tmp_dw[: tmp_dw.shape[0] // nr_enable, :]

                        # delta_w.append(
                        #     F.conv1d(
                        #         self.lora_A[i].data.unsqueeze(0),
                        #         self.lora_B[i].data.unsqueeze(-1),
                        #         groups=sum(self.enable_lora),
                        #     ).squeeze(0)
                        #     * self.scaling[i]
                        # )
                    # delta_w = F.conv1d(
                    #     self.lora_A.data.unsqueeze(0),
                    #     self.lora_B.data.unsqueeze(-1),
                    #     groups=sum(self.enable_lora),
                    # ).squeeze(0)
                    # delta_w = torch.cat(delta_w, dim=1)
                    # self.weight.data += self.zero_pad_weight(T(delta_w * self.scaling))

                    self.weight.data += self.zero_pad_weight(T(delta_w))

                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                x = torch.chunk(x, self.nr_lora, dim=-1)
                delta_result = list()
                for _ in range(sum(self.enable_lora)):
                    delta_result.append(list())

                for i in range(self.nr_lora):
                    tmp_delta_result = F.linear(x[i], self.lora_A[i])
                    tmp_delta_result = F.conv1d(
                        tmp_delta_result.transpose(-2, -1),
                        self.lora_B[i].unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    ).transpose(-2, -1)
                    tmp_delta_result = tmp_delta_result.chunk(
                        sum(self.enable_lora), dim=-1
                    )
                    for i, tmp_r in enumerate(tmp_delta_result):
                        delta_result[i].append(tmp_r)
                    # delta_result.append(tmp_delta_result * self.scaling[i])

                delta_result = [torch.cat(dr, dim=-1) for dr in delta_result]
                delta_result = torch.cat(delta_result, dim=-1)

                # after_A = F.linear(self.lora_dropout(x), self.lora_A)
                # after_B = F.conv1d(
                #     after_A.transpose(-2, -1),
                #     self.lora_B.unsqueeze(-1),
                #     groups=sum(self.enable_lora),
                # ).transpose(-2, -1)
                # result += self.zero_pad(after_B) * self.scaling
                result = result + self.zero_pad(delta_result)

            return result


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(
        self,
        conv_module,
        in_channels,
        out_channels,
        kernel_size,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        merge_weights=True,
        **kwargs
    ):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros(
                    (out_channels // self.conv.groups * kernel_size, r * kernel_size)
                )
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.conv.weight.data -= (self.lora_B @ self.lora_A).view(
                    self.conv.weight.shape
                ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.conv.weight.data += (self.lora_B @ self.lora_A).view(
                    self.conv.weight.shape
                ) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight
                + (self.lora_B @ self.lora_A).view(self.conv.weight.shape)
                * self.scaling,
                self.conv.bias,
            )
        return self.conv(x)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)


# Can Extend to other ones like this


class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)
