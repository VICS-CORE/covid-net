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
        "op_size": len(config['OP_FEATURES']),
        "dropout": config.get('DROPOUT', 0),
        "ip_aux_size": len(config.get('AUX_FEATURES', []))
    }
    model = arch_mod.CovidNet(**args)
    model = model.to(DEVICE)
    if v: print ("Model initialised")
    
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    
    return model, cp
