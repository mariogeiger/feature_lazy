# pylint: disable=no-member, not-callable, missing-docstring, line-too-long, invalid-name
import os
import sys

import numpy as np
import requests
import tables
import torch


class Higgs2BB(torch.utils.data.Dataset):
    url = 'http://opendata.cern.ch/eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/test/ntuple_merged_{}.h5'

    # 27 features
    features = ['fj_jetNTracks',
                'fj_nSV',
                'fj_tau0_trackEtaRel_0',
                'fj_tau0_trackEtaRel_1',
                'fj_tau0_trackEtaRel_2',
                'fj_tau1_trackEtaRel_0',
                'fj_tau1_trackEtaRel_1',
                'fj_tau1_trackEtaRel_2',
                'fj_tau_flightDistance2dSig_0',
                'fj_tau_flightDistance2dSig_1',
                'fj_tau_vertexDeltaR_0',
                'fj_tau_vertexEnergyRatio_0',
                'fj_tau_vertexEnergyRatio_1',
                'fj_tau_vertexMass_0',
                'fj_tau_vertexMass_1',
                'fj_trackSip2dSigAboveBottom_0',
                'fj_trackSip2dSigAboveBottom_1',
                'fj_trackSip2dSigAboveCharm_0',
                'fj_trackSipdSig_0',
                'fj_trackSipdSig_0_0',
                'fj_trackSipdSig_0_1',
                'fj_trackSipdSig_1',
                'fj_trackSipdSig_1_0',
                'fj_trackSipdSig_1_1',
                'fj_trackSipdSig_2',
                'fj_trackSipdSig_3',
                'fj_z_ratio']

    # spectators to define mass/pT window
    spectators = ['fj_sdmass',
                  'fj_pt']

    # 2 labels: QCD or Hbb
    labels = ['fj_isQCD*sample_isQCD',
              'fj_isH*fj_isBB']

    nfeatures = len(features)
    nspectators = len(spectators)
    nlabels = len(labels)

    def __init__(self, root, files=None, transform=None):
        if files is None:
            files = [0]

        self.root = os.path.expanduser(root)
        self.files = files
        self.transform = transform
        self.x = torch.empty(0, 27, dtype=torch.float64)
        self.y = torch.empty(0, dtype=torch.float64)
        self.download()
        self.load()
        self.normalize()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def download(self):
        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        for i in self.files:
            path = os.path.join(self.root, "ntuple_merged_{}.h5".format(i))
            url = self.url.format(i)

            if not os.path.exists(path):
                print('wget -c {} -O {}'.format(url, path), flush=True)

                r = requests.get(url, stream=True)
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            f.flush()
                        print(".", end="")
                        sys.stdout.flush()

    def load(self):
        for i in self.files:
            path = os.path.join(self.root, "ntuple_merged_{}.h5".format(i))
            x, y = self.get_features_labels(path)
            self.x = torch.cat([self.x, x], dim=0)
            self.y = torch.cat([self.y, y], dim=0)

    def normalize(self):
        missing = []

        for i in range(15):
            f = self.x[:, i]
            missing.append(f == -1)
            f[f == -1] = f[f != -1].mean()
            self.x[:, i] = f

        missing = torch.stack(missing, dim=1)
        self.x = (self.x - self.x.mean(0)) / self.x.std(0)
        self.x = torch.cat([self.x, missing.double().mul(2).sub(1)], dim=1)

    def get_features_labels(self, file_name, remove_mass_pt_window=True):
        # load file
        h5file = tables.open_file(file_name, 'r')
        njets = getattr(h5file.root, self.features[0]).shape[0]

        # allocate arrays
        feature_array = np.zeros((njets, self.nfeatures))
        spec_array = np.zeros((njets, self.nspectators))
        label_array = np.zeros((njets, self.nlabels))

        # load feature arrays
        for (i, feat) in enumerate(self.features):
            feature_array[:, i] = getattr(h5file.root, feat)[:]

        # load spectator arrays
        for (i, spec) in enumerate(self.spectators):
            spec_array[:, i] = getattr(h5file.root, spec)[:]

        # load labels arrays
        for (i, label) in enumerate(self.labels):
            prods = label.split('*')
            prod0 = prods[0]
            prod1 = prods[1]
            fact0 = getattr(h5file.root, prod0)[:]
            fact1 = getattr(h5file.root, prod1)[:]
            label_array[:, i] = np.multiply(fact0, fact1)

        # remove samples outside mass/pT window
        if remove_mass_pt_window:
            feature_array = feature_array[(spec_array[:, 0] > 40) & (spec_array[:, 0] < 200) & (spec_array[:, 1] > 300) & (spec_array[:, 1] < 2000)]
            label_array = label_array[(spec_array[:, 0] > 40) & (spec_array[:, 0] < 200) & (spec_array[:, 1] > 300) & (spec_array[:, 1] < 2000)]

        feature_array = feature_array[np.sum(label_array, axis=1) == 1]
        label_array = label_array[np.sum(label_array, axis=1) == 1]

        feature_array = torch.tensor(feature_array)
        label_array = torch.tensor(label_array)
        assert label_array.sum(1).eq(1).all()
        label_array = label_array[:, 0].mul(2).sub(1)

        h5file.close()

        return feature_array, label_array
