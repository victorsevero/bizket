from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.utils import safe_mean
import torch


class ModelArchCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ModelArchCallback, self).__init__(verbose)

    def _on_training_start(self):
        input_data = torch.zeros(1, 5, 84, 84, device="cuda")

        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )
        self.model.policy.eval()
        with torch.no_grad():
            self.tb_formatter.writer.add_graph(self.model.policy, input_data)
        self.model.policy.train()
        return True

    def _on_step(self) -> bool:
        return True


class HpLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(HpLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if (
            len(self.model.ep_info_buffer) > 0
            and len(self.model.ep_info_buffer[0]) > 0
        ):
            self.logger.record(
                "rollout/hp_player",
                safe_mean(
                    [
                        ep_info["player_hp"]
                        for ep_info in self.model.ep_info_buffer
                    ]
                ),
            )
            self.logger.record(
                "rollout/hp_boss",
                safe_mean(
                    [
                        ep_info["boss_hp"]
                        for ep_info in self.model.ep_info_buffer
                    ]
                ),
            )
