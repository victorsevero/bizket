from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import torch


class ModelArchCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ModelArchCallback, self).__init__(verbose)

    def _on_training_start(self):
        input_data = torch.zeros(1, 3, 84, 84, device="cuda")
        input_data = torch.clamp(input_data, min=0, max=1)

        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )
        self.tb_formatter.writer.add_graph(self.model.policy, input_data)

        return True

    def _on_step(self) -> bool:
        return True