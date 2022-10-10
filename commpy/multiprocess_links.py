# Authors: CommPy contributors
# License: BSD 3-Clause

"""
============================================
Multiprocess Links (:mod:`commpy.links`)
============================================

.. autosummary::
   :toctree: generated/

   LinkModel            -- Multiprocess Link model object.
   Wifi80211            -- Multiprocess class to simulate the transmissions and receiving parameters of physical layer 802.11

"""
from __future__ import division  # Python 2 compatibility

from itertools import product, cycle
from multiprocessing import Pool
from typing import Iterable, List, Optional

import numpy as np

from commpy.channels import _FlatChannel
from commpy.links import LinkModel as SPLinkModel
from commpy.wifi80211 import Wifi80211 as SPWifi80211

__all__ = ['LinkModel', 'Wifi80211']


class LinkModel(SPLinkModel):

    def __init__(self, modulate, channel, receive, num_bits_symbol, constellation, Es=1., decoder=None, rate=1.,
                 number_of_process: int = -1):
        self.params_builder = _RunParamsBuilder(modulate, channel, receive, num_bits_symbol, constellation, Es, decoder,
                                                rate)
        self.full_simulation_results = []
        self.number_of_process = number_of_process

    def link_performance_full_metrics(self, SNRs: Iterable, tx_max, err_min, send_chunk=None, code_rate: float = 1.,
                                      number_chunks_per_send=1, stop_on_surpass_error=True):
        pool = Pool(self.number_of_process if self.number_of_process > 0 else None)
        results = pool.map(_run_link_performance_full_metrics,
                           [self.params_builder.build_to_run([SNR],
                                                             tx_max, err_min, send_chunk,
                                                             code_rate, number_chunks_per_send,
                                                             stop_on_surpass_error)
                            for SNR in SNRs])
        tmp_res = {}
        for SNR, res in results:
            tmp_res[SNR] = res
        tmp_res_keys = sorted(tmp_res.keys())
        self.full_simulation_results = [[], [], [], []]
        for SNR in tmp_res_keys:
            BERs, BEs, CEs, NCs = tmp_res[SNR]
            self.full_simulation_results[0].append(BERs)
            self.full_simulation_results[1].append(BEs)
            self.full_simulation_results[2].append(CEs)
            self.full_simulation_results[3].append(NCs)

        return self.full_simulation_results

    def link_performance(self, SNRs, send_max, err_min, send_chunk=None, code_rate=1):
        pool = Pool(self.number_of_process if self.number_of_process > 0 else None)
        results = pool.map(_run_link_performance,
                           [self.params_builder.build_to_run([SNR],
                                                             send_max, err_min, send_chunk,
                                                             code_rate)
                            for SNR in SNRs])
        tmp_res = {}
        for SNR, BERs in results:
            tmp_res[SNR] = BERs
        tmp_res_keys = sorted(tmp_res.keys())
        self.full_simulation_results = []
        for SNR in tmp_res_keys:
            self.full_simulation_results.extend(tmp_res[SNR])
        return self.full_simulation_results


class _RunParamsBuilder:
    def __init__(self, modulate, channel, receive, num_bits_symbol, constellation, Es, decoder, rate):
        self.modulate = modulate
        self.channel = channel
        self.receive = receive
        self.num_bits_symbol = num_bits_symbol
        self.constellation = constellation
        self.Es = Es
        self.rate = rate
        self.decoder = decoder

    def build_to_run(self, SNR, tx_max, err_min, send_chunk, code_rate,
                     number_chunks_per_send=1, stop_on_surpass_error=True):
        return _RunParams(self.modulate,
                          self.channel,
                          self.receive,
                          self.num_bits_symbol,
                          self.constellation,
                          self.Es,
                          self.decoder,
                          self.rate,
                          SNR, tx_max, err_min, send_chunk, code_rate,
                          number_chunks_per_send, stop_on_surpass_error
                          )


class _RunParams:
    def __init__(self, modulate, channel, receive, num_bits_symbol, constellation, Es, decoder, rate,
                 SNRs, tx_max, err_min, send_chunk, code_rate,
                 number_chunks_per_send, stop_on_surpass_error
                 ):
        self.modulate = modulate
        self.channel = channel
        self.receive = receive
        self.num_bits_symbol = num_bits_symbol
        self.constellation = constellation
        self.Es = Es
        self.rate = rate
        self.decoder = decoder
        self.SNRs = SNRs
        self.tx_max = tx_max
        self.err_min = err_min
        self.send_chunk = send_chunk
        self.code_rate = code_rate
        self.number_chunks_per_send = number_chunks_per_send
        self.stop_on_surpass_error = stop_on_surpass_error


def _run_link_performance_full_metrics(run_params: _RunParams):
    link_model = SPLinkModel(run_params.modulate, run_params.channel, run_params.receive, run_params.num_bits_symbol,
                             run_params.constellation, run_params.Es, run_params.decoder, run_params.rate)
    return run_params.SNRs[0], [x[0] for x in
                                link_model.link_performance_full_metrics(run_params.SNRs, run_params.tx_max,
                                                                         run_params.err_min,
                                                                         run_params.send_chunk, run_params.code_rate,
                                                                         run_params.number_chunks_per_send,
                                                                         run_params.stop_on_surpass_error)]


def _run_link_performance(run_params: _RunParams):
    link_model = SPLinkModel(run_params.modulate, run_params.channel, run_params.receive, run_params.num_bits_symbol,
                             run_params.constellation, run_params.Es, run_params.decoder, run_params.rate)
    return run_params.SNRs[0], [x[0] if isinstance(x, np.ndarray) else x for x in
                                link_model.link_performance(run_params.SNRs, run_params.tx_max,
                                                            run_params.err_min,
                                                            run_params.send_chunk, run_params.code_rate)]


class Wifi80211(SPWifi80211):
    def __init__(self, mcs: int, number_of_processes=-1):
        self.mcs = mcs
        self.number_of_processes = number_of_processes

    def link_performance(self, channel: _FlatChannel, SNRs: Iterable, tx_max, err_min, send_chunk=None,
                         frame_aggregation=1, receiver=None, stop_on_surpass_error=True):
        return self.link_performance_mp_mcs([self.mcs], [SNRs], channel, tx_max, err_min, send_chunk, frame_aggregation,
                                            [receiver], stop_on_surpass_error)[self.mcs]

    def link_performance_mp_mcs(self, mcss: List[int], SNRss: Iterable[Iterable],
                                channel: _FlatChannel, tx_max, err_min, send_chunk=None,
                                frame_aggregation=1,
                                receivers: Optional[Iterable] = None,
                                stop_on_surpass_error=True):
        """
        Explicit multiprocess of multiple MCSs link performance call

        Parameters
        ----------
        mcss    : list of MCSs to run
        SNRss   : SNRs to test
        channel : Channel to test the MCSs at each SNR at
        tx_max  : maximum number of transmissions to test
        err_min : minimum error be
        send_chunk : amount of bits to send at each frame
        frame_aggregation : number of frames to send at each transmission
        receivers  : function to handle receiving
        stop_on_surpass_error : flag to stop when err_min was surpassed

        Returns
        -------

        """
        pool = Pool(self.number_of_processes if self.number_of_processes > 0 else None)

        if not receivers:
            receivers = [None] * len(mcss)

        results = pool.map(_run_wifi80211_link_performance,
                           [[[SNR], mcs, channel, tx_max, err_min, send_chunk, frame_aggregation,
                             receiver, stop_on_surpass_error]
                            for _SNRs, mcs, receiver in zip(SNRss, mcss, receivers)
                            for SNR in _SNRs])
        tmp_res = {}
        for SNR, mcs, res in results:
            tmp_res.setdefault(mcs, {})[SNR] = res
        tmp_res_keys = sorted(tmp_res.keys())
        full_simulation_results = {}
        for mcs in tmp_res_keys:
            full_simulation_results[mcs] = [[], [], [], []]
            for snr in sorted(tmp_res[mcs].keys()):
                BERs, BEs, CEs, NCs = tmp_res[mcs][snr]
                full_simulation_results[mcs][0].append(BERs[0])
                full_simulation_results[mcs][1].append(BEs)
                full_simulation_results[mcs][2].append(CEs)
                full_simulation_results[mcs][3].append(NCs)
        return full_simulation_results


def _run_wifi80211_link_performance(args: List):
    SNRs, mcs, channel, tx_max, err_min, send_chunk, frame_aggregation, receiver, stop_on_surpass_error = args
    sp_wifi80211 = SPWifi80211(mcs)
    res = sp_wifi80211.link_performance(channel, SNRs, tx_max, err_min, send_chunk, frame_aggregation,
                                        receiver, stop_on_surpass_error)
    return SNRs[0], mcs, res
