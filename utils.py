import torch
import importlib
import numpy as np

ARCHS_DIR = 'archs'
EXPERIMENTS_DIR = 'experiments'
DEVICE = 'cpu'

def save_checkpoint(config, epoch, model, optimizer, 
                    trn_losses, val_losses, min_val_loss,
                    trn_acc, val_acc, max_val_acc,
                    experiment_dir, fn="", v=True):
    torch.save({
        'config': config,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'trn_losses': trn_losses,
        'val_losses': val_losses,
        'min_val_loss': min_val_loss,
        'trn_acc': trn_acc,
        'val_acc': val_acc,
        'max_val_acc': max_val_acc
    }, experiment_dir + "/" + fn or "latest.pt")
    if v: print("Checkpoint saved")

def load_checkpoint(experiment_dir, fn="", device="cpu", v=True):
    cp = torch.load(experiment_dir + "/" + fn or "latest.pt", map_location=device)
    if v: print("Checkpoint loaded")
    return cp

def load_model(experiment_id, checkpoint, v=True):
    """returns model and checkpoint data"""
    experiment_dir = EXPERIMENTS_DIR + '/' + experiment_id
    cp = load_checkpoint(experiment_dir, checkpoint, v=v)
    if v: print("Epochs:", cp['epoch'])
    config = cp['config']
    if v: print(config)

    # init Net
    arch_mod = importlib.import_module("." + config['ARCH'], ARCHS_DIR)
    importlib.reload(arch_mod) # ensure changes are imported
    args = {
        "ip_seq_len": config['DS']['IP_SEQ_LEN'], 
        "op_seq_len": config['DS']['OP_SEQ_LEN'], 
        "hidden_size": config['HIDDEN_SIZE'], 
        "num_layers": config['NUM_LAYERS'],
        "ip_size": len(config['IP_FEATURES']),
        "op_size": len(config['OP_FEATURES'])
    }
    if config['ARCH'] == 'v2':
        args['dropout'] = config['DROPOUT']
    model = arch_mod.CovidNet(**args)
    model = model.to(DEVICE)
    if v: print ("Model initialised")
    
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    
    return model, cp

def fix_anomalies(df):
    # MH data fix: spread 17 Juns deaths over last 60 days
    if 'new_deaths' in df.columns:
        df.loc[(df['date']=='2020-06-17') & (df['location']=='India'), 'new_deaths'] = 353 #actual=2003
        t = np.random.rand(60)
        t = (2003-353) * t / sum(t)
        df.loc[(df['date']>='2020-04-18') & (df['date']<'2020-06-17') & (df['location']=='India'), 'new_deaths'] += np.int32(t)

    return df
