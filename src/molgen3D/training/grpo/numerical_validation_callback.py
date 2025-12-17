"""
Callback for numerical validation during GRPO training.
"""

from typing import Optional

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NumericalValidationCallback(TrainerCallback):
    """
    Callback to run numerical validation during GRPO training.

    This callback runs numerical validation at specified intervals during training,
    providing detailed metrics on model performance beyond the standard reward metrics.
    """

    def __init__(
        self,
        validator,
        stats,
        validation_steps: Optional[int] = None,
        max_seq_len: int = 4096,
    ):
        """
        Initialize the numerical validation callback.

        Args:
            validator: The numerical validator instance
            stats: Statistics tracker
            validation_steps: Run validation every N steps. If None, only run at end.
            max_seq_len: Maximum sequence length for generation
        """
        self.validator = validator
        self.stats = stats
        self.validation_steps = validation_steps
        self.max_seq_len = max_seq_len
        self.last_validation_step = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ) -> TrainerControl:
        """Run numerical validation at the beginning of training (step 0)."""
        current_step = state.global_step
        logger.info(
            f"Running numerical validation at train start (step {current_step})"
        )
        metrics = self.validator.run_validation(
            model=kwargs.get("model"),
            step=current_step,
            max_seq_len=self.max_seq_len,
        )

        if metrics:
            self.stats.update_numerical_validation_stats(metrics)
            logger.info(
                f"Numerical validation completed at train start (step {current_step})"
            )

        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ) -> TrainerControl:
        """Run numerical validation at specified intervals."""
        if self.validation_steps is None:
            return control

        current_step = state.global_step
        if current_step - self.last_validation_step >= self.validation_steps:
            logger.info(f"Running numerical validation at step {current_step}")
            metrics = self.validator.run_validation(
                model=kwargs.get("model"),
                step=current_step,
                max_seq_len=self.max_seq_len,
            )

            if metrics:
                self.stats.update_numerical_validation_stats(metrics)
                logger.info(f"Numerical validation completed at step {current_step}")

            self.last_validation_step = current_step

        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ) -> TrainerControl:
        """Run final numerical validation at the end of training."""
        current_step = state.global_step
        logger.info(f"Running final numerical validation at step {current_step}")
        metrics = self.validator.run_validation(
            model=kwargs.get("model"),
            step=current_step,
            max_seq_len=self.max_seq_len,
        )

        if metrics:
            self.stats.update_numerical_validation_stats(metrics)
            logger.info(f"Final numerical validation completed: {metrics}")

        return control
