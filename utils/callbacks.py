import pytorch_lightning as pl
import time
from optuna.integration import PyTorchLightningPruningCallback


class TimerCallback(pl.Callback):
    """Custom callback to track training time."""
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
    
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
    
    def on_train_end(self, trainer, pl_module):
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        trainer.logger.log_metrics({
            'total_training_hours': hours,
            'total_training_time': f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        })
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        trainer.logger.log_metrics({
            'epoch_time_seconds': epoch_time,
            'epoch_time': f"{epoch_time:.2f}s"
        })

class ClearMemoryCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        gc.collect()

import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pathlib import Path
class DeadlockDetectionCallback(pl.Callback):
    def __init__(self, timeout_mins=10):
        self.timeout = timeout_mins * 60
        self.last_batch_time = time.time()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()
        time_since_last = current_time - self.last_batch_time
        
        if time_since_last > self.timeout:
            print(f"Potential deadlock detected! No progress for {time_since_last/60:.2f} minutes")
            print("Current GPU Memory Usage:")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.2f}GB")
            raise Exception("Training deadlock detected")
            
        self.last_batch_time = current_time
class MemoryDebugCallback(pl.Callback):
    def __init__(self):
        self.prev_allocated = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:  # Check every 10 batches
            allocated = torch.cuda.memory_allocated()
            delta = allocated - self.prev_allocated
            print(f"Batch {batch_idx}")
            print(f"Current memory: {allocated/1e9:.2f}GB")
            print(f"Delta: {delta/1e9:.2f}GB")
            self.prev_allocated = allocated
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
class CustomCallbackManager:
    """
    A class to manage and dynamically select PyTorch Lightning callbacks.
    """
    def __init__(self, save_dir=None, trial=None):
        """
        Initialize the callback manager.

        Args:
            save_dir (str or Path, optional): Directory for saving model checkpoints.
            trial (optuna.trial.Trial, optional): Optuna trial for pruning callback.
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.trial = trial
        self.callbacks = [MemoryDebugCallback(), DeadlockDetectionCallback()]
    def add_clear_memory_callback(self):
        self.callbacks.append(ClearMemoryCallback())
    def add_timer_callback(self):
        """
        Add a TimerCallback to track training time.
        """
        self.callbacks.append(TimerCallback())
    
    def add_model_checkpoint(self, monitor='val_iou_macro', mode='max', save_top_k=1, verbose=True):
        """
        Add a ModelCheckpoint callback.

        Args:
            monitor (str): Metric to monitor.
            mode (str): Optimization mode, 'min' or 'max'.
            save_top_k (int): Number of top models to save.
            verbose (bool): Whether to print logs about model checkpointing.
        """
        if not self.save_dir:
            raise ValueError("Save directory is required for ModelCheckpoint callback.")
        self.callbacks.append(pl.callbacks.ModelCheckpoint(
            dirpath=self.save_dir / "checkpoints",
            filename="{epoch}-{val_iou_macro:.3f}",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            verbose=verbose
        ))
    
    def add_early_stopping(self, monitor='val_iou_macro', mode='max', patience=10, verbose=True):
        """
        Add an EarlyStopping callback.

        Args:
            monitor (str): Metric to monitor.
            mode (str): Optimization mode, 'min' or 'max'.
            patience (int): Number of epochs with no improvement to stop training.
            verbose (bool): Whether to print logs about early stopping.
        """
        self.callbacks.append(pl.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=verbose
        ))
    
    def add_pruning(self, monitor='val_iou_macro'):
        """
        Add a PyTorchLightningPruningCallback for Optuna pruning.

        Args:
            monitor (str): Metric to monitor for pruning.
        """
        if not self.trial:
            raise ValueError("Optuna trial is required for the Pruning callback.")
        self.callbacks.append(PyTorchLightningPruningCallback(
            trial=self.trial,
            monitor=monitor
        ))
    
    def add_learning_rate_monitor(self, logging_interval='step'):
        """
        Add a LearningRateMonitor callback.

        Args:
            logging_interval (str): Interval for logging learning rate ('step' or 'epoch').
        """
        self.callbacks.append(pl.callbacks.LearningRateMonitor(
            logging_interval=logging_interval
        ))

    def get_callbacks(self):
        """
        Get the list of configured callbacks.

        Returns:
            list: List of PyTorch Lightning callbacks.
        """
        return self.callbacks
    def clear(self):
        self.callbacks=[]
