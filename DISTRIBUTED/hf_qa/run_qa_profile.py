import os

import run_qa
from profiler_callback import TorchProfilerCallback
from trainer_qa import QuestionAnsweringTrainer


class ProfiledQuestionAnsweringTrainer(QuestionAnsweringTrainer):
    def __init__(self, *args, **kwargs):
        callbacks = list(kwargs.pop("callbacks", []) or [])

        callbacks.append(
            TorchProfilerCallback(
                os.environ["PROFILER_DIR"]
            )
        )

        kwargs["callbacks"] = callbacks
        super().__init__(*args, **kwargs)


run_qa.QuestionAnsweringTrainer = ProfiledQuestionAnsweringTrainer


if __name__ == "__main__":
    run_qa.main()
