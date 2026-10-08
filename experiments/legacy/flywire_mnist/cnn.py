"""Traditional two-convolution MNIST baseline with a locked official test."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from . import dataset
from .protocol import atomic_json, file_sha, source_identity, training_data, test_report


class DigitCNN(nn.Module):
    """LeNet-style feature extraction; 28x28 -> 24 -> 12 -> 8 -> 4."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(1,16,5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(32*4*4,128), nn.ReLU(), nn.Linear(128,10))

    def forward(self, images):
        return self.layers(images)


def epoch(model, optimizer, images, labels, ids, batch_size, seed, device):
    model.train()
    order = np.random.default_rng(seed).permutation(ids)
    total_loss = 0.
    for start in range(0,len(order),batch_size):
        chosen = order[start:start+batch_size]
        x = torch.from_numpy(images[chosen]).unsqueeze(1).to(device=device,dtype=torch.float32)/255
        y = torch.from_numpy(labels[chosen].astype(np.int64)).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x),y)
        loss.backward(); optimizer.step()
        total_loss += float(loss.detach().cpu())*len(chosen)
    return total_loss/len(order)


def infer(model, images, ids, batch_size, device):
    model.eval()
    predicted = []
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            x = torch.from_numpy(images[ids[start:start+batch_size]]).unsqueeze(1)
            x = x.to(device=device,dtype=torch.float32)/255
            predicted.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(predicted)


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError('epochs, batch size and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but no Apple GPU is available')
    device = torch.device(args.device)
    images, labels, ids, fit, val, split = training_data(args.data,args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output/'split.json',split)
    config = {'schema':'mnist-cnn-v1','seed':args.seed,'max_epochs':args.epochs,
              'batch_size':args.batch_size,'optimizer':'Adam','learning_rate':.001,
              'architecture':'conv5-16/pool2/conv5-32/pool2/fc128/fc10, ReLU',
              'pixel_scaling':'uint8 / 255, no augmentation',
              'device':str(device),'threads':args.threads,'torch_version':torch.__version__,
              'split_sha256':split['sha256'],'sources':source_identity(),
              'selection':'best validation accuracy; earliest epoch wins ties',
              'final_fit':'reinitialize then fit all official 60,000 at selected epoch count'}
    atomic_json(args.output/'protocol.json',config)
    model=DigitCNN().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    history=[];best=(-1.,0);started=time.perf_counter()
    for k in range(1,args.epochs+1):
        loss=epoch(model,optimizer,images,labels,fit,args.batch_size,args.seed+k,device)
        accuracy=float(np.mean(infer(model,images,val,args.batch_size,device)==labels[val]))
        if accuracy>best[0]:best=(accuracy,k)
        history.append({'epoch':k,'loss':loss,'validation_accuracy':accuracy})
        atomic_json(args.output/'progress.json',{'phase':'validation-selection','history':history,
                                                'elapsed_seconds':time.perf_counter()-started})
        print(json.dumps(history[-1]),flush=True)
    # The epoch budget is immutable before refitting or opening official test.
    locked={'selected_epochs':best[1],'validation_accuracy':best[0],
            'protocol_sha256':file_sha(args.output/'protocol.json'),'official_test_used':False}
    atomic_json(args.output/'selection-lock.json',locked)
    torch.manual_seed(args.seed)
    model=DigitCNN().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for k in range(1,best[1]+1):
        loss=epoch(model,optimizer,images,labels,ids,args.batch_size,args.seed+k,device)
        torch.save(model.state_dict(),args.output/'refit-last.pt')
        progress={'phase':'refit-60000','epoch':k,'epochs':best[1],'loss':loss,
                  'elapsed_seconds':time.perf_counter()-started}
        atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    torch.save(model.cpu().state_dict(),args.output/'model.pt')
    locked['model_sha256']=file_sha(args.output/'model.pt')
    atomic_json(args.output/'test-lock.json',locked)
    model.to(device)
    test_images,test_labels,test_ids=dataset.load(args.data,test=True)
    predicted=infer(model,test_images,np.arange(len(test_ids)),args.batch_size,device)
    report={'model':'traditional CNN','fit_count':60000,'test_count':10000,
            'selection':locked,'test':test_report(test_labels,predicted),
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'elapsed_seconds':time.perf_counter()-started,
            'test_files':{name:file_sha(args.data/name) for name in dataset.FILES if name.startswith('t10k-')}}
    np.savez_compressed(args.output/'test-predictions.npz',sample_ids=test_ids,labels=test_labels,predicted=predicted)
    atomic_json(args.output/'report.json',report)
    atomic_json(args.output/'progress.json',{'phase':'complete','test_accuracy':report['test']['accuracy']})
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--seed',type=int,default=783)
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    run(parser.parse_args())


if __name__=='__main__':main()
