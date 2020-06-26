import torch

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
