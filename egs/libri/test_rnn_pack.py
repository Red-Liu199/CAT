import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Union, Literal, Sequence, Tuple, List


class PackedSequence():
    def __init__(self, xs: Sequence[torch.Tensor] = None) -> None:
        if xs is None:
            return
        assert all(x.size()[1:] == xs[0].size()[1:] for x in xs)

        _lens = [x.size(0) for x in xs]
        self._data = xs[0].new_empty(
            ((sum(_lens),)+xs[0].size()[1:]))
        pref = 0
        for x, l in zip(xs, _lens):
            self._data[pref:pref+l] = x
            pref += l
        self._lens = torch.LongTensor(_lens)

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def batch_sizes(self) -> torch.LongTensor:
        return self._lens

    def to(self, device):
        newpack = PackedSequence()
        newpack._data = self._data.to(device)
        newpack._lens = self._lens.to(device)
        return newpack

    def set(self, data: torch.Tensor):
        assert data.size(0) == self._data.size(0)
        self._data = data

    def unpack(self) -> Tuple[List[torch.Tensor], torch.LongTensor]:
        out = []

        pref = 0
        for l in self._lens:
            out.append(self._data[pref:pref+l])
            pref += l
        return out, self._lens


def print_peak_memory(prefix, device):
    print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


class JointNet(nn.Module):
    """
    Joint `encoder_output` and `decoder_output`.
    Args:
        encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size ``(batch, time_steps, dimensionA)``
        decoder_output (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size ``(batch, label_length, dimensionB)``
    Returns:
        outputs (torch.FloatTensor): outputs of joint `encoder_output` and `decoder_output`. `FloatTensor` of size ``(batch, time_steps, label_length, dimensionA + dimensionB)``
    """

    def __init__(self, odim_encoder: int, odim_decoder: int, num_classes: int, gather: bool = True, HAT: bool = False, act: Union[Literal['tanh'], Literal['relu']] = 'tanh'):
        super().__init__()
        in_features = odim_encoder+odim_decoder
        self.fc_enc = nn.Linear(odim_encoder, in_features)
        self.fc_dec = nn.Linear(odim_decoder, in_features)
        self._isHAT = HAT
        self._gather = gather
        if act == 'tanh':
            act_layer = nn.Tanh()
        elif act == 'relu':
            act_layer = nn.ReLU()
        else:
            raise NotImplementedError(f"Unknown activation layer type: {act}")

        if HAT:
            """
            Implementation of Hybrid Autoregressive Transducer (HAT)
            https://arxiv.org/abs/2003.07705
            """
            self.fc = nn.Sequential(
                act_layer,
                nn.Linear(in_features, num_classes-1)
            )
            self.distr_blk = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                act_layer,
                nn.Linear(in_features, num_classes)
            )

    def forward(self, encoder_output: Union[torch.FloatTensor, PackedSequence], decoder_output: Union[torch.FloatTensor, PackedSequence]) -> torch.FloatTensor:
        if self._gather:
            assert isinstance(encoder_output, PackedSequence) and isinstance(
                decoder_output, PackedSequence)

            packed_encoder_output = self.fc_enc(encoder_output.data)
            packed_decoder_output = self.fc_dec(decoder_output.data)

            L_enc = encoder_output.batch_sizes
            L_dec = decoder_output.batch_sizes

            # pad into batch layout
            perf_enc = 0
            perf_dec = 0
            img_xs = []
            for b in range(L_enc.size(0)):
                # type:torch.Tensor
                x_t = packed_encoder_output[perf_enc:perf_enc+L_enc[b]]
                perf_enc += L_enc[b]

                # type:torch.Tensor
                x_u = packed_decoder_output[perf_dec:perf_dec+L_dec[b]]
                perf_dec += L_dec[b]

                expanded_x = x_t.unsqueeze(1) + x_u.unsqueeze(0)
                img_xs.append(expanded_x.view(-1, expanded_x.size(-1)))
            img_xs = PackedSequence(img_xs)

            img_xs.set(self.fc(img_xs.data).log_softmax(dim=-1))

            img_xlist, _ = img_xs.unpack()
            MaxPad = torch.max(L_enc), torch.max(L_dec)

            for i in range(len(img_xlist)):
                img_xlist[i] = F.pad(img_xlist[i].view(L_enc[i], L_dec[i], -1), pad=(
                    0, 0, 0, MaxPad[1]-L_dec[i], 0, MaxPad[0]-L_enc[i]), value=0.)

            img = torch.stack(img_xlist)

            return img

        encoder_output = self.fc_enc(encoder_output)
        decoder_output = self.fc_dec(decoder_output)

        if encoder_output.dim() == 3:
            # expand the outputs
            T_max = encoder_output.size(1)
            U_max = decoder_output.size(1)

            encoder_output = encoder_output.unsqueeze(2)
            decoder_output = decoder_output.unsqueeze(1)

            encoder_output = encoder_output.repeat(
                [1, 1, U_max, 1])
            decoder_output = decoder_output.repeat(
                [1, T_max, 1, 1])
        elif encoder_output.dim() == 2 and self._gather:
            T_max = encoder_output.size(0)
            U_max = decoder_output.size(0)

        if self._isHAT:
            conbined_input = encoder_output + decoder_output

            prob_blk = self.distr_blk(conbined_input)
            vocab_logits = self.fc(conbined_input)
            vocab_log_probs = torch.log(
                1-prob_blk)+torch.log_softmax(vocab_logits, dim=-1)
            outputs = torch.cat([torch.log(prob_blk), vocab_log_probs], dim=-1)
        else:
            outputs = self.fc(
                encoder_output+decoder_output).log_softmax(dim=-1)

        return outputs


def SetRandomSeed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    SetRandomSeed(0)

    device = 'cuda:0'

    batchsize = 8
    dim_T, dim_U = 512, 512
    gather = False

    print("{0} Test with gather={1} {0}".format('================', gather))

    model = JointNet(dim_T, dim_U, 1024, gather=gather, act='relu')
    model.cuda(device)

    x_t = [torch.randn((torch.randint(50, 450, (1,)), dim_T))
           for _ in range(batchsize)]
    if gather:
        x_t = PackedSequence(x_t).to(device)
        print(x_t.data.size())
    else:
        x_t_lens = [x.size(0) for x in x_t]
        x_t = pad_sequence(x_t, batch_first=True).to(device)
        print(x_t.size())

    x_u = [torch.randn((torch.randint(10, 50, (1,)), dim_U))
           for _ in range(batchsize)]
    if gather:
        x_u = PackedSequence(x_u).to(device)
        print(x_u.data.size())
    else:
        x_u_lens = [x.size(0) for x in x_u]
        x_u = pad_sequence(x_u, batch_first=True).to(device)
        print(x_u.size())

    print_peak_memory("Memory allocated since data and model init", device)
    t0 = time.time()

    nested = model(x_t, x_u)
    print(nested.size())

    if not gather:
        loss = sum([torch.sum(nested[i][:x_t_lens[i], :x_u_lens[i], :])
                   for i in range(nested.size(0))])
    else:
        loss = torch.sum(nested)

    loss.backward()
    t1 = time.time()
    print("Loss = {:.4e}".format(loss))

    print_peak_memory("Memory allocated after loss backward", device)
    print("Computation time: {:.4f}s".format(t1-t0))
