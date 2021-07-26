import os
import signal

import torch
import ttools
from ttools.utils import get_logger

LOG = get_logger(__name__)


class CustomTrainer(object):
    """Implements a simple training loop with hooks for callbacks. Same as ttools.training.Trainer
     except the methods are not private.

        Args:
          interface (ModelInterface): adapter to run forward and backward
            pass on the model being trained.

        Attributes:
          callbacks (list of Callbacks): hooks that will be called while training
            progresses.
        """

    def __init__(self, interface):
        super(CustomTrainer, self).__init__()
        self.callbacks = []
        self.interface = interface
        LOG.debug("Creating {}".format(self))

        signal.signal(signal.SIGINT, self.interrupt_handler)

        self._keep_running = True

    def interrupt_handler(self, signo, frame):
        """Stop the training process upon receiving a SIGINT (Ctrl+C)."""
        LOG.debug("interrupting run")
        self._keep_running = False

    def _stop(self):
        # Reset the run flag
        self._keep_running = True
        self.training_end()

    def add_callback(self, callback):
        """Adds a callback to the list of training hooks.

        Args:
            callback(ttools.Callback): callback to add.
        """
        LOG.debug("Adding callback {}".format(callback))
        # pass an interface reference to the callback
        callback.model_interface = self.interface
        self.callbacks.append(callback)

    def train(self, dataloader, starting_epoch=None, num_epochs=None,
              val_dataloader=None):
        """Main training loop. This starts the training procedure.

        Args:
          dataloader (DataLoader): loader that yields training batches.
          starting_epoch (int, optional): index of the epoch we are starting from.
          num_epochs (int, optional): max number of epochs to run.
          val_dataloader (DataLoader, optional): loader that yields validation
            batches
        """
        self.training_start(dataloader)
        if starting_epoch is None:
            starting_epoch = 0

        LOG.info("Starting taining from epoch %d", starting_epoch)

        epoch = starting_epoch

        while num_epochs is None or epoch < starting_epoch + num_epochs:
            self.epoch_start(epoch)
            for batch_idx, batch in enumerate(dataloader):
                if not self._keep_running:
                    self._stop()
                    return
                self.batch_start(batch_idx, batch)
                train_step_data = self.training_step(batch)
                self.batch_end(batch, train_step_data)
            self.epoch_end()

            # TODO: allow validation at intermediate steps during one epoch

            # Validate
            if val_dataloader:
                with torch.no_grad():
                    running_val_data = self.validation_start(val_dataloader)
                    for batch_idx, batch in enumerate(val_dataloader):
                        if not self._keep_running:
                            self._stop()
                            return
                        self.val_batch_start(batch_idx, batch)
                        running_val_data = self.validation_step(batch, running_val_data)
                        self.val_batch_end(batch, running_val_data)
                    self.validation_end(running_val_data)

            epoch += 1

            if not self._keep_running:
                self._stop()
                return

        self._stop()

    def __repr__(self):
        return "Trainer({}, {} callbacks)".format(
            self.interface, len(self.callbacks))

    def training_start(self, dataloader):
        for cb in self.callbacks:
            cb.training_start(dataloader)

    def training_end(self):
        for cb in self.callbacks:
            cb.training_end()

    def epoch_start(self, epoch_idx):
        for cb in self.callbacks:
            cb.epoch_start(epoch_idx)

    def epoch_end(self):
        for cb in self.callbacks:
            cb.epoch_end()

    def batch_start(self, batch_idx, batch):
        for cb in self.callbacks:
            cb.batch_start(batch_idx, batch)

    def batch_end(self, batch, train_step_data):
        for cb in self.callbacks:
            cb.batch_end(batch, train_step_data)

    def val_batch_start(self, batch_idx, batch):
        for cb in self.callbacks:
            cb.val_batch_start(batch_idx, batch)

    def val_batch_end(self, batch, running_val_data):
        for cb in self.callbacks:
            cb.val_batch_end(batch, running_val_data)

    def validation_start(self, dataloader):
        for cb in self.callbacks:
            cb.validation_start(dataloader)
        return self.interface.init_validation()

    def validation_end(self, running_val_data):
        for cb in self.callbacks:
            cb.validation_end(running_val_data)

    def training_step(self, batch):
        return self.interface.training_step(batch)

    def validation_step(self, batch, running_val_data):
        return self.interface.validation_step(batch, running_val_data)


class SchedulerTrainer(CustomTrainer):
    """Implements a simple training loop with hooks for callbacks.
     Use this if you have a Learning Rate Scheduler which needs to updated after validation

    Args:
      interface (ModelInterface): adapter to run forward and backward
        pass on the model being trained.

    Attributes:
      callbacks (list of Callbacks): hooks that will be called while training
        progresses.
    """

    def validation_end(self, running_val_data):
        super().validation_end(running_val_data)
        self.interface.epoch_num += 1
        self.interface.scheduler.step()
        print(f"Validation loss: {running_val_data['loss']}")
        if running_val_data['loss'] < self.interface.best_val_loss:
            path = os.path.join(self.interface.args.checkpoint_dir,
                                f'best_model.pth')
            torch.save({
                'model_state_dict': self.interface.model.state_dict(),
                'optimizer_state_dict': self.interface.optimizer.state_dict(),
                'loss': running_val_data['loss'],
                'epoch': self.interface.epoch_num,
                'scheduler': self.interface.scheduler.state_dict()
            }, path)
            self.interface.best_val_loss = running_val_data['loss']
            print(f"Best Val Loss {self.interface.best_val_loss}. Saving Model.")


class GeneralTrainer(CustomTrainer):
    """

    """

    def validation_end(self, running_val_data):
        super().validation_end(running_val_data)
        self.interface.epoch_num += 1
        print(f"Validation loss: {running_val_data['loss']}")

        if running_val_data['loss'] < self.interface.best_val_loss:
            path = os.path.join(self.interface.args.checkpoint_dir,
                                f'best_model.pth')
            torch.save({
                'model_state_dict': self.interface.model.state_dict(),
                'optimizer_state_dict': self.interface.optimizer.state_dict(),
                'loss': running_val_data['loss'],
                'epoch': self.interface.epoch_num
            }, path)
            self.interface.best_val_loss = running_val_data['loss']
            print(f"Best Val Loss {self.interface.best_val_loss}. Saving Model.")


class BatchScheduleTrainer(SchedulerTrainer):
    def batch_end(self, batch, train_step_data):
        super(BatchScheduleTrainer, self).batch_end(batch, train_step_data)
        self.interface.scheduler.step()
